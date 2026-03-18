# eval_pqr_bge_m3.py
# -*- coding: utf-8 -*-
# Baseline: PQR-lite on top of BGE-m3
# 依赖：
#   pip install FlagEmbedding scikit-learn jsonlines
#
# 运行示例：
#   CUDA_VISIBLE_DEVICES=0 python eval_pqr_bge_m3.py \
#     --data_dir ./dataset/ \
#     --pqr_jsonl ./generated/pqr_qwen25_7b.train.merged.dedup.jsonl

import os
import argparse
from datetime import datetime
from typing import Dict, List

import math
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import jsonlines

from FlagEmbedding import BGEM3FlagModel

from src.ChunQiuDataset import ChunqiuEvalDataset, load_splits, build_corpus_index
from src.retrieval_utils import compute_retrieval_metrics  # 保留引用以保持风格一致（未直接使用）
from src.method_eval_utils import pretty_print_summary


DEFAULT_META    = "chunqiu_meta_sid_fixed.json"
DEFAULT_QUERIES = "queries_all_labeledv3.jsonl"
DEFAULT_SPLITS  = "time_splits_by_month_v1.json"

DEFAULT_MODEL_NAME = "/amax/wangyh/pretrained/bge-m3"  # or "BAAI/bge-m3"
DEFAULT_MAX_Q_LEN = 128
DEFAULT_MAX_D_LEN = 256

# PQR config
DEFAULT_PQR_MAX_Q_PER_SID = 20          # 对每个 sid 最多用多少条生成 query
DEFAULT_PQR_K = 4                       # 每个 doc 的聚类簇数
DEFAULT_PQR_BATCH_SIZE = 32             # 编码生成 query 的 batch size

WRITE_LOG = True
OUT_ROOT = "model_outputs_results"
MODEL_TAG = "bge3_pqr"


def encode_bge_m3_dense(texts, model: BGEM3FlagModel, batch_size: int, max_length: int):
    """
    返回 torch.FloatTensor (N, D)，L2-normalized
    """
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        out = model.encode(
            batch,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = out["dense_vecs"]  # numpy (B, D) or torch
        if isinstance(dense, torch.Tensor):
            embs = dense.float()
        else:
            embs = torch.from_numpy(dense).float()
        embs = F.normalize(embs, p=2, dim=1)
        all_embs.append(embs)
    if not all_embs:
        # 理论上不会走到这里
        return torch.empty(0, 0)
    return torch.cat(all_embs, dim=0)


def compute_retrieval_metrics_from_scores(
    sims: torch.Tensor,          # (Q, G_docs)
    gold_indices_list,           # List[List[int]] over doc indices
    ks=(1, 5, 10),
):
    """
    复制 retrieval_utils.compute_retrieval_metrics 的逻辑，
    但直接使用预先计算好的 sims 矩阵（Q x #docs）。
    额外算一个 nDCG@10，方便和原来的 summary 对齐。
    """
    Q = sims.size(0)

    recalls = {k: 0 for k in ks}
    mrr_at_10 = 0.0
    ndcg_at_10 = 0.0

    with torch.no_grad():
        for qi in range(Q):
            sim_row = sims[qi]
            gold_indices = gold_indices_list[qi]
            if not gold_indices:
                continue

            topv, topi = torch.topk(sim_row, k=min(10, sim_row.size(0)))
            topi = topi.tolist()

            # Recall@k
            for k in ks:
                k_cut = min(k, len(topi))
                if any(idx in topi[:k_cut] for idx in gold_indices):
                    recalls[k] += 1

            # MRR@10
            rank = None
            for r, idx in enumerate(topi):
                if idx in gold_indices:
                    rank = r + 1
                    break
            if rank is not None:
                mrr_at_10 += 1.0 / rank

            # nDCG@10（binary relevance）
            dcg = 0.0
            for r, idx in enumerate(topi):
                if idx in gold_indices:
                    dcg += 1.0 / math.log2(r + 2)
            ideal_hits = min(len(gold_indices), len(topi))
            if ideal_hits > 0:
                idcg = 0.0
                for r in range(ideal_hits):
                    idcg += 1.0 / math.log2(r + 2)
                if idcg > 0:
                    ndcg_at_10 += dcg / idcg

    metrics = {}
    for k in ks:
        metrics[f"Recall@{k}"] = recalls[k] / Q if Q > 0 else 0.0
    metrics["MRR@10"] = mrr_at_10 / Q if Q > 0 else 0.0
    metrics["nDCG@10"] = ndcg_at_10 / Q if Q > 0 else 0.0
    return metrics


def build_pqr_gallery_bge(
    corpus,
    full_gallery_sids: List[int],
    pqr_jsonl_path: str,
    model: BGEM3FlagModel,
    max_queries_per_sid: int,
    k_components: int,
    max_length: int,
    batch_size: int,
):
    """
    从 PQR 生成的 jsonl 中构建句级 doc 的 multi-vector 表示：
    - 对每个 sid，收集 generated_queries（如果没有则 fallback 原始句子文本）；
    - 用 BGE 编码；
    - 若条数 > k_components，用 KMeans(K) 得到 K 个中心；否则直接用所有向量；
    - 返回：
        doc_comp_embs: FloatTensor [N_comp, D]
        sid2comp_indices: Dict[sid, List[int]]
    """
    # 1) 读取 jsonl -> sid2queries
    sid2queries: Dict[int, List[str]] = {}
    if not os.path.isfile(pqr_jsonl_path):
        raise FileNotFoundError(f"PQR jsonl not found: {pqr_jsonl_path}")

    with jsonlines.open(pqr_jsonl_path, "r") as r:
        for obj in r:
            # 你自己的生成脚本里是 "sid"，如果叫 "id" 可以自己改一下这里
            sid = int(obj.get("sid"))
            gq = obj.get("generated_queries") or obj.get("generated_queries_text") or []
            if not isinstance(gq, list):
                continue
            # 简单去重 & 截断
            uniq = []
            seen = set()
            for q in gq:
                q = (q or "").strip()
                if not q:
                    continue
                if q in seen:
                    continue
                seen.add(q)
                uniq.append(q)
                if len(uniq) >= max_queries_per_sid:
                    break
            if uniq:
                sid2queries[sid] = uniq

    print(f"[PQR] loaded generated queries for {len(sid2queries)} sids from: {pqr_jsonl_path}")

    all_comp_embs: List[torch.Tensor] = []
    sid2comp_indices: Dict[int, List[int]] = {}
    current_idx = 0

    # 遍历完整 gallery 的 sid，保证每个 sid 至少有一个向量（即使没生成 query）
    for sid in tqdm(full_gallery_sids, desc="[PQR] build doc multi-vectors"):
        # 对每个 sid，若有 generated_queries 就用；否则 fallback 原始文本
        if sid in sid2queries:
            texts = sid2queries[sid]
        else:
            texts = [corpus.sid2text[sid]]

        # 编码
        embs = encode_bge_m3_dense(texts, model, batch_size=batch_size, max_length=max_length)
        if embs.size(0) == 0:
            # 极端情况：编码失败，跳过
            continue

        n = embs.size(0)
        if n <= k_components:
            means = embs
        else:
            # KMeans 聚类
            X = embs.cpu().numpy()
            km = KMeans(
                n_clusters=k_components,
                random_state=0,
                n_init="auto",
            )
            km.fit(X)
            centers = torch.from_numpy(km.cluster_centers_).float()
            centers = F.normalize(centers, p=2, dim=1)
            means = centers.to(embs.device)

        all_comp_embs.append(means)
        n_comp = means.size(0)
        sid2comp_indices[sid] = list(range(current_idx, current_idx + n_comp))
        current_idx += n_comp

    if all_comp_embs:
        doc_comp_embs = torch.cat(all_comp_embs, dim=0)
    else:
        # 理论上不会走到这里
        doc_comp_embs = torch.empty(0, 0).float()

    print(f"[PQR] total components = {doc_comp_embs.size(0)}, dim = {doc_comp_embs.size(1)}")
    return doc_comp_embs, sid2comp_indices


def compute_pqr_scores_for_queries(
    query_embs: torch.Tensor,          # (Q, D)
    doc_comp_embs: torch.Tensor,       # (N_comp, D)
    sid2comp_indices: Dict[int, List[int]],
    gallery_sids: List[int],
    device: torch.device,
) -> torch.Tensor:
    """
    对一批 query，计算基于 PQR 的 doc-level 相似度：
      sim(q, d) = max_k <v_q, v_{d,k}>
    返回 scores: (Q, G_docs)，列顺序与 gallery_sids 一致。
    """
    if query_embs.device != device:
        query_embs = query_embs.to(device)

    # 收集这批 gallery_sids 的 component index
    eff_comp_indices: List[int] = []
    for sid in gallery_sids:
        comp_idxs = sid2comp_indices.get(sid, [])
        eff_comp_indices.extend(comp_idxs)

    if not eff_comp_indices:
        return torch.empty(query_embs.size(0), 0, device=device)

    # 去重 & 构建映射（理论上不会有重复，但这样更稳）
    eff_comp_indices = sorted(set(eff_comp_indices))
    comp_idx2pos = {idx: j for j, idx in enumerate(eff_comp_indices)}

    comp_embs_eff = doc_comp_embs[eff_comp_indices].to(device)  # (G_eff, D)
    sims_qc = torch.matmul(query_embs, comp_embs_eff.t())       # (Q, G_eff)

    Q = sims_qc.size(0)
    num_docs = len(gallery_sids)
    # 用 -inf 作为初始值，便于做 max
    sid_scores = torch.full((Q, num_docs), -1e9, device=device)

    # 每个 doc 做一次 max over components
    for local_sid_idx, sid in enumerate(gallery_sids):
        comp_idxs = sid2comp_indices.get(sid, [])
        if not comp_idxs:
            continue
        pos_list = [comp_idx2pos[c] for c in comp_idxs if c in comp_idx2pos]
        if not pos_list:
            continue
        comp_scores = sims_qc[:, pos_list]           # (Q, #comp_for_sid)
        max_scores, _ = comp_scores.max(dim=1)       # (Q,)
        sid_scores[:, local_sid_idx] = max_scores

    return sid_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default="dataset", help="dataset directory")
    ap.add_argument("--data_dir", dest="data_dir_opt", default=None, help="same as positional data_dir")

    ap.add_argument(
        "--pqr_jsonl",
        type=str,
        required=True,
        help="Path to merged PQR generated queries jsonl (one line per sid).",
    )

    ap.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    ap.add_argument("--max_q_len", type=int, default=DEFAULT_MAX_Q_LEN)
    ap.add_argument("--max_d_len", type=int, default=DEFAULT_MAX_D_LEN)

    ap.add_argument("--pqr_max_queries", type=int, default=DEFAULT_PQR_MAX_Q_PER_SID)
    ap.add_argument("--pqr_k", type=int, default=DEFAULT_PQR_K)
    ap.add_argument("--pqr_bs", type=int, default=DEFAULT_PQR_BATCH_SIZE)

    args = ap.parse_args()
    data_dir = args.data_dir_opt or args.data_dir

    meta_path    = os.path.join(data_dir, DEFAULT_META)
    queries_path = os.path.join(data_dir, DEFAULT_QUERIES)
    splits_path  = os.path.join(data_dir, DEFAULT_SPLITS)

    assert os.path.isfile(meta_path),    f"meta not found: {meta_path}"
    assert os.path.isfile(queries_path), f"queries not found: {queries_path}"
    assert os.path.isfile(splits_path),  f"splits not found: {splits_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    now_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(OUT_ROOT, f"outputs_{MODEL_TAG}")
    os.makedirs(out_dir, exist_ok=True)
    model_name = os.path.basename(args.model_name.rstrip("/")).replace("/", "_")
    log_path = os.path.join(out_dir, f"eval_{model_name}_{now_tag}.log") if WRITE_LOG else None

    print("[EVAL-PQR] device =", device)
    print("[EVAL-PQR] model  =", args.model_name)
    print("[EVAL-PQR] max_q_len =", args.max_q_len, "max_d_len =", args.max_d_len)
    print("[EVAL-PQR] pqr_max_queries =", args.pqr_max_queries, "pqr_k =", args.pqr_k)

    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"[RUN] {now_tag}\n")
            f.write(f"device={device}\n")
            f.write(f"model={args.model_name}\n")
            f.write(f"pqr_jsonl={args.pqr_jsonl}\n")
            f.write(f"max_q_len={args.max_q_len}, max_d_len={args.max_d_len}\n")
            f.write(f"pqr_max_queries={args.pqr_max_queries}, pqr_k={args.pqr_k}\n")
            f.write("=" * 80 + "\n")

    # 1) corpus
    splits_by_sort_key = load_splits(splits_path)
    corpus = build_corpus_index(meta_path, splits_by_sort_key)

    # FULL gallery
    full_gallery_sids = sorted(corpus.sid2text.keys())
    print(f"[EVAL-PQR] FULL gallery size = {len(full_gallery_sids)}")

    # 2) init BGE encoder
    bge = BGEM3FlagModel(args.model_name, use_fp16=(device.type == "cuda"))

    # 3) build PQR multi-vector gallery
    doc_comp_embs, sid2comp_indices = build_pqr_gallery_bge(
        corpus=corpus,
        full_gallery_sids=full_gallery_sids,
        pqr_jsonl_path=args.pqr_jsonl,
        model=bge,
        max_queries_per_sid=args.pqr_max_queries,
        k_components=args.pqr_k,
        max_length=args.max_d_len,
        batch_size=args.pqr_bs,
    )

    # 4) encode FULL queries for val/test
    eval_sets = {}
    for split in ("val", "test"):
        eval_ds = ChunqiuEvalDataset(
            meta_path=meta_path,
            queries_path=queries_path,
            splits_path=splits_path,
            split=split,
            include_no_event_queries=True,
        )

        raw_queries, gold_sids_list, is_pure_flags, q_types = [], [], [], []
        for q in eval_ds.queries:
            raw_queries.append(q["query"])
            gold_sids_list.append(q["pos_sids"])
            is_pure_flags.append(q.get("is_pure_no_event", False))
            q_types.append(q.get("type", "point"))

        print(f"[EVAL-PQR] Loaded {len(raw_queries)} FULL {split} queries.")

        query_embs_full = encode_bge_m3_dense(
            raw_queries,
            bge,
            batch_size=args.pqr_bs,
            max_length=args.max_q_len,
        ).to(device)

        eval_sets[split] = {
            "gold_sids_list": gold_sids_list,
            "is_pure_no_event": is_pure_flags,
            "q_types": q_types,
            "query_embs_full": query_embs_full,
        }

    # 5) combos + summary
    combos = []
    for include_neg in (False, True):
        for include_no_event_sids in (False, True):
            for drop_no_event_q in (False, True):
                if (not include_no_event_sids) and (not drop_no_event_q):
                    continue
                combos.append((include_neg, include_no_event_sids, drop_no_event_q))

    summary_metrics = {}

    for include_neg, include_no_event_sids, drop_no_event_q in combos:
        eff_gallery_sids = []
        for sid in full_gallery_sids:
            t = corpus.sid2_type.get(sid, "event")
            if (t == "neg_comment") and (not include_neg):
                continue
            if (t == "no_event") and (not include_no_event_sids):
                continue
            eff_gallery_sids.append(sid)

        if not eff_gallery_sids:
            continue

        sid2effidx = {sid: i for i, sid in enumerate(eff_gallery_sids)}

        mode_name = f"neg{int(include_neg)}_ne{int(include_no_event_sids)}_dq{int(drop_no_event_q)}"
        print("\n" + "=" * 80)
        print(f"[EVAL-PQR] MODE = {mode_name} | gallery_size = {len(eff_gallery_sids)}")

        for split in ("val", "test"):
            gold_sids_list = eval_sets[split]["gold_sids_list"]
            is_pure_flags = eval_sets[split]["is_pure_no_event"]
            q_types = eval_sets[split]["q_types"]
            query_embs_full = eval_sets[split]["query_embs_full"]

            valid_entries = []
            for i, (sids, is_pure, q_type) in enumerate(zip(gold_sids_list, is_pure_flags, q_types)):
                if drop_no_event_q and is_pure:
                    continue
                # map gold sids -> eff gallery indices
                local_idx = [sid2effidx[sid] for sid in sids if sid in sid2effidx]
                if not local_idx:
                    continue
                valid_entries.append({"orig_idx": i, "gold_indices": local_idx, "q_type": q_type})

            if not valid_entries:
                print(f"[EVAL-PQR] split={split}: 0 valid queries, skip.")
                continue

            base_all = [e["orig_idx"] for e in valid_entries]
            gold_all = [e["gold_indices"] for e in valid_entries]
            q_all = query_embs_full[base_all]

            # PQR scoring: sim(q, d) = max_k <q, v_{d,k}>
            sid_scores_all = compute_pqr_scores_for_queries(
                query_embs=q_all,
                doc_comp_embs=doc_comp_embs,
                sid2comp_indices=sid2comp_indices,
                gallery_sids=eff_gallery_sids,
                device=device,
            )

            metrics_all = compute_retrieval_metrics_from_scores(
                sid_scores_all,
                gold_all,
                ks=(1, 5, 10),
            )
            summary_metrics[(mode_name, "all", split)] = metrics_all

            # 按 query 类型拆成 point / window
            for family in ("point", "window"):
                if family == "point":
                    sub = [e for e in valid_entries if e["q_type"] == "point"]
                else:
                    sub = [e for e in valid_entries if e["q_type"] != "point"]
                if not sub:
                    continue
                base_sub = [e["orig_idx"] for e in sub]
                gold_sub = [e["gold_indices"] for e in sub]
                q_sub = query_embs_full[base_sub]
                sid_scores_sub = compute_pqr_scores_for_queries(
                    query_embs=q_sub,
                    doc_comp_embs=doc_comp_embs,
                    sid2comp_indices=sid2comp_indices,
                    gallery_sids=eff_gallery_sids,
                    device=device,
                )
                metrics_sub = compute_retrieval_metrics_from_scores(
                    sid_scores_sub,
                    gold_sub,
                    ks=(1, 5, 10),
                )
                summary_metrics[(mode_name, family, split)] = metrics_sub

            m = metrics_all
            print(
                f"[EVAL-PQR] {split} | N={len(base_all)} | "
                f"R1={m['Recall@1']:.4f} R5={m['Recall@5']:.4f} "
                f"R10={m['Recall@10']:.4f} MRR10={m['MRR@10']:.4f} "
                f"nDCG10={m['nDCG@10']:.4f}"
            )

    pretty_print_summary(summary_metrics, log_path=log_path)
    if log_path:
        print(f"\n[EVAL-PQR] log saved to: {log_path}")


if __name__ == "__main__":
    main()
