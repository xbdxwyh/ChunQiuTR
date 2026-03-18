# method_eval_colbert_jina.py
# Requires:
#   pip install "pylate[eval]" einops "flash-attn>=2"
#
# Run (single GPU recommended):
#   CUDA_VISIBLE_DEVICES=0 python ./method_eval_colbert_jina.py \
#       --data_dir ./dataset \
#       --output_dir ./model_outputs_results/outputs_colbert_jina \
#       --model_name_or_path jinaai/jina-colbert-v2
# CUDA_VISIBLE_DEVICES=0 python ./method_eval_colbert_jina.py --output_dir ./model_outputs_results/outputs_colbert_jina --model_name_or_path ../pretrained/jina-colbert-v2/

import os
import argparse
from datetime import datetime

import torch

from pylate import indexes, models, retrieve

from src.ChunQiuDataset import (
    ChunqiuEvalDataset,
    load_splits,
    build_corpus_index,
    build_eval_gallery,
)
from src.method_eval_utils import pretty_print_summary, export_ranked_results_jsonl


DEFAULT_META    = "chunqiu_meta_sid_fixed.json"
DEFAULT_QUERIES = "queries_all_labeledv3.jsonl"
DEFAULT_SPLITS  = "time_splits_by_month_v1.json"

DEFAULT_MODEL_NAME = "jinaai/jina-colbert-v2"
DEFAULT_Q_BS = 32
DEFAULT_D_BS = 32

WRITE_LOG = True
OUT_ROOT  = "model_outputs_results"
MODEL_TAG = "colbert_lfm2"


# ======== metrics: 根据 ranked doc ids 计算 R@k / MRR / nDCG ========

def _compute_metrics_from_ranked_ids(
    ranked_ids_list,
    gold_ids_list,
    ks=(1, 5, 10),
):
    """
    ranked_ids_list: List[List[str]]，每个 query 的检索结果 doc_id 排序列表
    gold_ids_list:   List[List[str]]，每个 query 的 gold doc_id 列表
    """
    assert len(ranked_ids_list) == len(gold_ids_list)
    n = len(ranked_ids_list)
    ks = sorted(ks)
    max_k = ks[-1]

    total_recall = {k: 0.0 for k in ks}
    total_mrr = 0.0
    total_ndcg = 0.0

    import math

    for preds, gold in zip(ranked_ids_list, gold_ids_list):
        gold_set = set(gold)
        if not gold_set:
            # 理论上我们已经滤掉了没有正例的 query，这里只是兜底
            continue

        # ----- Recall@k -----
        for k in ks:
            topk = preds[:k]
            hit = any(doc_id in gold_set for doc_id in topk)
            if hit:
                total_recall[k] += 1.0

        # ----- MRR@max_k -----
        mrr_val = 0.0
        for rank, doc_id in enumerate(preds[:max_k], start=1):
            if doc_id in gold_set:
                mrr_val = 1.0 / rank
                break
        total_mrr += mrr_val

        # ----- nDCG@max_k (binary relevance) -----
        # DCG
        dcg = 0.0
        for rank, doc_id in enumerate(preds[:max_k], start=1):
            rel = 1.0 if doc_id in gold_set else 0.0
            if rel > 0:
                dcg += rel / math.log2(rank + 1)

        # IDCG：理想情况是前 min(len(gold), max_k) 个都 relevant
        ideal_rels = min(len(gold_set), max_k)
        idcg = 0.0
        for rank in range(1, ideal_rels + 1):
            idcg += 1.0 / math.log2(rank + 1)
        if idcg > 0:
            ndcg = dcg / idcg
        else:
            ndcg = 0.0

        total_ndcg += ndcg

    # 平均
    metrics = {}
    for k in ks:
        metrics[f"Recall@{k}"] = total_recall[k] / n if n > 0 else 0.0
    metrics["MRR@10"] = total_mrr / n if n > 0 else 0.0
    metrics["nDCG@10"] = total_ndcg / n if n > 0 else 0.0
    return metrics


def eval_one_split_with_colbert(
    eval_ds: ChunqiuEvalDataset,
    model: models.ColBERT,
    retriever: retrieve.ColBERT,
    gallery_sid_set,
    drop_no_event_q: bool,
    batch_size: int = 32,
    ks=(1, 5, 10),
    export_path: str = None,      # 新增：如果不为 None，就导出 JSONL
    export_top_k: int = 20,       # 新增：导出多少个 top-K
    gallery_mode: str = "neg1_ne1_dq0",     # 新增：记录当前 combo，比如 "neg0_ne1_dq1"
):
    """
    和 Sentence-T5 那个 eval_one_split_* 类似，但这里是：
      - 用 PyLate ColBERT encode queries -> embeddings
      - retriever.retrieve() -> ranked doc ids
      - 自己算 metrics
    """
    raw_queries, gold_sids_list, is_pure_flags, q_types = [], [], [], []
    for q in eval_ds.queries:
        raw_queries.append(q["query"])
        gold_sids_list.append(q["pos_sids"])
        is_pure_flags.append(q.get("is_pure_no_event", False))
        q_types.append(q.get("type", "point"))

    kept_q_texts = []
    kept_gold_ids = []
    kept_q_types = []

    for q_text, gold_sids, is_pure, q_type in zip(
        raw_queries, gold_sids_list, is_pure_flags, q_types
    ):
        if drop_no_event_q and is_pure:
            continue
        # 只保留当前 gallery 里存在的 gold sids
        valid_sids = [sid for sid in gold_sids if sid in gallery_sid_set]
        if not valid_sids:
            continue
        kept_q_texts.append(q_text)
        kept_gold_ids.append([str(sid) for sid in valid_sids])
        kept_q_types.append(q_type)

    if not kept_q_texts:
        return {}

    # 编码 queries
    q_embs = model.encode(
        kept_q_texts,
        batch_size=batch_size,
        is_query=True,
        show_progress_bar=False,
    )

    # 检索 top-k（k 至少 >= max(ks)）
    max_k = max(ks)
    scores = retriever.retrieve(
        queries_embeddings=q_embs,
        k=max_k,
    )
    # scores: List[List[{"id": str, "score": float}]]

    ranked_ids_list = [[item["id"] for item in row] for row in scores]
    ranked_scores_list = [[float(item["score"]) for item in row] for row in scores]

    # ===== all =====
    metrics_all = _compute_metrics_from_ranked_ids(
        ranked_ids_list, kept_gold_ids, ks=ks
    )
    metrics_all["N"] = len(kept_q_texts)

    # ===== point / window =====
    idx_point = [i for i, t in enumerate(kept_q_types) if t == "point"]
    idx_window = [i for i, t in enumerate(kept_q_types) if t != "point"]

    def pack_family(indices):
        if not indices:
            return {
                "Recall@1": 0.0,
                "Recall@5": 0.0,
                "Recall@10": 0.0,
                "MRR@10": 0.0,
                "nDCG@10": 0.0,
                "N": 0,
            }
        sub_ranked = [ranked_ids_list[i] for i in indices]
        sub_gold = [kept_gold_ids[i] for i in indices]
        m = _compute_metrics_from_ranked_ids(sub_ranked, sub_gold, ks=ks)
        m["N"] = len(indices)
        return m

    metrics_point = pack_family(idx_point)
    metrics_window = pack_family(idx_window)

        # ==== 可选：导出 per-query top-K 结果 ====
    if export_path is not None:
        # 注意：kept_* 列表已经和 ranked_* 对齐
        export_ranked_results_jsonl(
            output_path=export_path,
            ranked_ids_list=ranked_ids_list,
            ranked_scores_list=ranked_scores_list,
            gold_ids_list=kept_gold_ids,
            raw_queries=kept_q_texts,
            q_types=kept_q_types,
            is_pure_no_event=None,    # 这里如果想要，可以把 kept_is_pure 也记录下来
            qids=None,                # ColBERT 这条路暂时不管 qid
            gallery_mode=gallery_mode,
            top_k=export_top_k,
        )

    return {"all": metrics_all, "point": metrics_point, "window": metrics_window}



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default="dataset", help="dataset directory")
    ap.add_argument("--data_dir", dest="data_dir_opt", default=None)

    ap.add_argument("--output_dir", default=None)
    ap.add_argument(
        "--model_name_or_path",
        default=DEFAULT_MODEL_NAME,
        help="jina-colbert-v2 checkpoint 或本地路径",
    )
    ap.add_argument("--q_batch_size", type=int, default=DEFAULT_Q_BS)
    ap.add_argument("--d_batch_size", type=int, default=DEFAULT_D_BS)
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    ap.add_argument(
        "--export_topk_dir",
        type=str,
        default=None,
        help="如果设置，则在 TEST split 上导出 per-query top-K 结果 JSONL 到该目录。",
    )
    ap.add_argument(
        "--export_topk_k",
        type=int,
        default=20,
        help="导出的 top-K 数量。",
    )


    args = ap.parse_args()
    data_dir = args.data_dir_opt or args.data_dir

    meta_path    = os.path.join(data_dir, DEFAULT_META)
    queries_path = os.path.join(data_dir, DEFAULT_QUERIES)
    splits_path  = os.path.join(data_dir, DEFAULT_SPLITS)

    assert os.path.isfile(meta_path),    f"meta not found: {meta_path}"
    assert os.path.isfile(queries_path), f"queries not found: {queries_path}"
    assert os.path.isfile(splits_path),  f"splits not found: {splits_path}"

    device = torch.device(args.device)
    now_tag = datetime.now().strftime("%Y%m%d-%H%M%S")

    out_root = OUT_ROOT
    if args.output_dir is not None:
        out_root = args.output_dir
    out_dir = os.path.join(out_root, f"outputs_{MODEL_TAG}")
    os.makedirs(out_dir, exist_ok=True)

    model_name = os.path.basename(str(args.model_name_or_path).rstrip("/")).replace("/", "_")
    log_path = os.path.join(out_dir, f"eval_{model_name}_{now_tag}.log") if WRITE_LOG else None

    print("[EVAL-ColBERT] device =", device)
    print("[EVAL-ColBERT] model  =", args.model_name_or_path)
    print("[EVAL-ColBERT] q_bs   =", args.q_batch_size, "d_bs =", args.d_batch_size)

    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"[RUN] {now_tag}\n")
            f.write(f"device={device}\n")
            f.write(f"model={args.model_name_or_path}\n")
            f.write(f"q_bs={args.q_batch_size}, d_bs={args.d_batch_size}\n")
            f.write("=" * 80 + "\n")

    # 1) corpus / splits
    splits_by_sort_key = load_splits(splits_path)
    corpus = build_corpus_index(meta_path, splits_by_sort_key)

    # 2) 预加载 eval datasets（val / test）
    eval_ds_map = {}
    for split in ("val", "test"):
        eval_ds_map[split] = ChunqiuEvalDataset(
            meta_path=meta_path,
            queries_path=queries_path,
            splits_path=splits_path,
            split=split,
            include_no_event_queries=True,
        )

    # 3) 加载 ColBERT 模型（PyLate）
    print("[EVAL-ColBERT] Loading PyLate ColBERT model ...")
    model = models.ColBERT(
        model_name_or_path=args.model_name_or_path,
        # 如果需要可以设置 query_prefix / document_prefix 等
        trust_remote_code=True,  # JinaColBERT 需要
    )
    # ★ 关键：官方建议这样设置 pad_token
    if getattr(model, "tokenizer", None) is not None:
        if model.tokenizer.pad_token is None and getattr(model.tokenizer, "eos_token", None) is not None:
            model.tokenizer.pad_token = model.tokenizer.eos_token

    try:
        model.to(device)
    except Exception:
        # 某些版本可能不暴露 .to()，encode 内部会处理
        pass

    # combos
    combos = []
    for include_neg in (False, True):
        for include_no_event_sids in (False, True):
            for drop_no_event_q in (False, True):
                if (not include_no_event_sids) and (not drop_no_event_q):
                    continue
                combos.append((include_neg, include_no_event_sids, drop_no_event_q))

    # group by (include_neg, include_no_event_sids)，避免重复建 index
    group_keys = sorted({(a, b) for (a, b, _) in combos})
    summary_metrics = {}

    for include_neg, include_no_event_sids in group_keys:
        # 4) build gallery
        gallery_sids, gallery_texts, sid2idx = build_eval_gallery(
            corpus,
            include_neg_samples=include_neg,
            include_no_event_sids=include_no_event_sids,
        )
        gallery_sid_set = set(gallery_sids)

        print("\n" + "=" * 80)
        print(
            f"[EVAL-ColBERT] Build gallery: "
            f"include_neg={include_neg}, include_no_event_sids={include_no_event_sids}"
        )
        print(f"[EVAL-ColBERT] gallery_size={len(gallery_sids)}")

        # 5) 建立 PLAID index
        index_folder = os.path.join(
            out_dir,
            f"pylate_index_neg{int(include_neg)}_ne{int(include_no_event_sids)}",
        )
        os.makedirs(index_folder, exist_ok=True)

        index = indexes.PLAID(
            index_folder=index_folder,
            index_name="index",
            override=True,
        )

        # encode gallery
        print("[EVAL-ColBERT] Encoding gallery ...")
        doc_embs = model.encode(
            gallery_texts,
            batch_size=args.d_batch_size,
            is_query=False,
            show_progress_bar=True,
        )
        index.add_documents(
            documents_ids=[str(sid) for sid in gallery_sids],
            documents_embeddings=doc_embs,
        )

        retriever = retrieve.ColBERT(index=index)

        # 对所有 dq 组合跑一次
        dq_list = sorted({dq for (a, b, dq) in combos if a == include_neg and b == include_no_event_sids})

        for drop_no_event_q in dq_list:
            mode_name = (
                f"neg{int(include_neg)}_"
                f"ne{int(include_no_event_sids)}_"
                f"dq{int(drop_no_event_q)}"
            )

            print("\n" + "-" * 80)
            print(f"[EVAL-ColBERT] === Combo: {mode_name} ===")

            for split in ("val", "test"):
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # 是否导出 top-K
                export_path = None
                if args.export_topk_dir is not None and split == "test":
                    os.makedirs(args.export_topk_dir, exist_ok=True)
                    fname = f"{MODEL_TAG}_{model_name}_{split}_{mode_name}_top{args.export_topk_k}.jsonl"
                    export_path = os.path.join(args.export_topk_dir, fname)

                out = eval_one_split_with_colbert(
                    eval_ds_map[split],
                    model=model,
                    retriever=retriever,
                    gallery_sid_set=gallery_sid_set,
                    drop_no_event_q=drop_no_event_q,
                    batch_size=args.q_batch_size,
                    ks=(1, 5, 10),
                    export_path=export_path,
                    export_top_k=args.export_topk_k,
                    gallery_mode=mode_name,
                )

                if not out:
                    print(
                        f"[EVAL-ColBERT] WARNING: split={split}, combo={mode_name} "
                        f"has 0 valid queries, skip."
                    )
                    continue

                m_all = out["all"]
                r1 = m_all["Recall@1"]
                r5 = m_all["Recall@5"]
                r10 = m_all["Recall@10"]
                mrr10 = m_all["MRR@10"]
                ndcg10 = m_all["nDCG@10"]
                run_name = f"colbert_jina_{split}_{mode_name}"

                print("=" * 80)
                print(f"Run name       : {run_name}")
                print(f"Time           : {now}")
                print(f"Split          : {split}")
                print(f"Gallery size   : {len(gallery_sids)}")
                print(f"#Eval queries  : {m_all['N']}")
                print("- Metrics ------------------------------")
                print(f"Recall@1    : {r1:.6f}")
                print(f"Recall@5    : {r5:.6f}")
                print(f"Recall@10   : {r10:.6f}")
                print(f"MRR@10      : {mrr10:.6f}")
                print(f"nDCG@10     : {ndcg10:.6f}")
                print("- Config -------------------------------")
                print(f"model_name             = {args.model_name_or_path}")
                print(f"q_batch_size           = {args.q_batch_size}")
                print(f"d_batch_size           = {args.d_batch_size}")
                print(f"include_neg_samples    = {include_neg}")
                print(f"include_no_event_sids  = {include_no_event_sids}")
                print(f"drop_no_event_queries  = {drop_no_event_q}")
                print("=" * 80)

                if log_path:
                    with open(log_path, "a", encoding="utf-8") as f_log:
                        f_log.write(
                            f"[{now}] [EVAL] run={run_name}, "
                            f"R1={r1:.6f}, R5={r5:.6f}, R10={r10:.6f}, "
                            f"MRR10={mrr10:.6f}, nDCG10={ndcg10:.6f}, "
                            f"#queries={m_all['N']}, "
                            f"include_neg={include_neg}, "
                            f"include_no_event_sids={include_no_event_sids}, "
                            f"drop_no_event_q={drop_no_event_q}, "
                            f"model={args.model_name_or_path}, "
                            f"q_batch_size={args.q_batch_size}, "
                            f"d_batch_size={args.d_batch_size}\n"
                        )

                # 收集 summary（all/point/window）
                for fam in ("all", "point", "window"):
                    summary_metrics[(mode_name, fam, split)] = {
                        "Recall@1":  out[fam]["Recall@1"],
                        "Recall@5":  out[fam]["Recall@5"],
                        "Recall@10": out[fam]["Recall@10"],
                        "MRR@10":    out[fam]["MRR@10"],
                        "nDCG@10":   out[fam]["nDCG@10"],
                    }

    # ===== summary 表 =====
    if not summary_metrics:
        print("[EVAL-ColBERT] No metrics collected, skip summary table.")
        return

    pretty_print_summary(summary_metrics, log_path=log_path)
    if log_path:
        print(f"\n[EVAL-ColBERT] log saved to: {log_path}")


if __name__ == "__main__":
    main()
