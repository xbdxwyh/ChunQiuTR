# method_bge3_zh.py
# Zero-shot eval for BAAI/bge-large-zh-v1.5 on ChunQiu benchmark (same eval combos)
#
# Install:
#   pip install -U FlagEmbedding
#
# Run (single GPU recommended):
#   CUDA_VISIBLE_DEVICES=0 python ./method_eval_bge3_zh.py --data_dir ./dataset/

import os
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np

from FlagEmbedding import FlagModel

from src.ChunQiuDataset import (
    ChunqiuEvalDataset,
    load_splits,
    build_corpus_index,
)
from src.retrieval_utils import compute_retrieval_metrics
from src.method_eval_utils import pretty_print_summary


# ====== defaults: 你只需要改这里 ======
DEFAULT_META     = "chunqiu_meta_sid_fixed.json"
DEFAULT_QUERIES  = "queries_all_labeledv3.jsonl"
DEFAULT_SPLITS   = "time_splits_by_month_v1.json"

DEFAULT_MODEL_NAME = "/amax/wangyh/pretrained/bge-large-zh-v1.5"   # or "BAAI/bge-large-zh-v1.5"
QUERY_INSTRUCTION  = "为这个句子生成表示以用于检索相关文章："
# QUERY_INSTRUCTION  = "Given a classical Chinese query about the Spring and Autumn Annals, retrieve relevant passages that describe the corresponding historical events."

DEFAULT_MAX_Q_LEN = 128
DEFAULT_MAX_D_LEN = 256
DEFAULT_Q_BS = 32
DEFAULT_D_BS = 32

WRITE_LOG = True
OUT_ROOT = "model_outputs_results"
MODEL_TAG = "bge_zh"


def _to_torch_float32(x) -> torch.Tensor:
    """Accept numpy / torch, return CPU float32 tensor."""
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu().float()
    else:
        t = torch.from_numpy(np.asarray(x)).float()
    return t


def encode_queries_flagmodel(queries, model: FlagModel, batch_size: int, max_length: int) -> torch.Tensor:
    """
    encode_queries() 会自动把 query_instruction_for_retrieval 加到 query 上
    Return: (N, D) CPU float32, L2-normalized
    """
    embs = model.encode_queries(
        queries,
        batch_size=batch_size,
        max_length=max_length,
        convert_to_numpy=True,
    )
    embs = _to_torch_float32(embs)
    embs = F.normalize(embs, p=2, dim=1)
    return embs


def encode_corpus_flagmodel(texts, model: FlagModel, batch_size: int, max_length: int) -> torch.Tensor:
    """
    encode_corpus() 不加 instruction
    Return: (N, D) CPU float32, L2-normalized
    """
    embs = model.encode_corpus(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        convert_to_numpy=True,
    )
    embs = _to_torch_float32(embs)
    embs = F.normalize(embs, p=2, dim=1)
    return embs


def main():
    # “真·单线程”：不搞多进程，不搞多卡（多卡你用 CUDA_VISIBLE_DEVICES 控制）
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(1)

    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default="dataset", help="dataset directory")
    ap.add_argument("--data_dir", dest="data_dir_opt", default=None, help="same as positional data_dir")
    args = ap.parse_args()
    data_dir = args.data_dir_opt or args.data_dir

    meta_path    = os.path.join(data_dir, DEFAULT_META)
    queries_path = os.path.join(data_dir, DEFAULT_QUERIES)
    splits_path  = os.path.join(data_dir, DEFAULT_SPLITS)

    assert os.path.isfile(meta_path), f"meta not found: {meta_path}"
    assert os.path.isfile(queries_path), f"queries not found: {queries_path}"
    assert os.path.isfile(splits_path), f"splits not found: {splits_path}"

    now_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(OUT_ROOT, f"outputs_{MODEL_TAG}")
    os.makedirs(out_dir, exist_ok=True)
    model_tag = os.path.basename(DEFAULT_MODEL_NAME.rstrip("/")).replace("/", "_")
    log_path = os.path.join(out_dir, f"eval_{model_tag}_{now_tag}.log") if WRITE_LOG else None

    print("[EVAL] model =", DEFAULT_MODEL_NAME)
    print("[EVAL] query_instruction =", QUERY_INSTRUCTION)
    print("[EVAL] max_q_len =", DEFAULT_MAX_Q_LEN, "max_d_len =", DEFAULT_MAX_D_LEN)
    print("[EVAL] q_bs =", DEFAULT_Q_BS, "d_bs =", DEFAULT_D_BS)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"[RUN] {now_tag}\n")
            f.write(f"model={DEFAULT_MODEL_NAME}\n")
            f.write(f"query_instruction={QUERY_INSTRUCTION}\n")
            f.write(f"max_q_len={DEFAULT_MAX_Q_LEN}, max_d_len={DEFAULT_MAX_D_LEN}\n")
            f.write(f"q_bs={DEFAULT_Q_BS}, d_bs={DEFAULT_D_BS}\n")
            f.write("=" * 80 + "\n")

    # 1) corpus
    splits_by_sort_key = load_splits(splits_path)
    corpus = build_corpus_index(meta_path, splits_by_sort_key)

    full_gallery_sids = sorted(corpus.sid2text.keys())
    full_gallery_texts = [corpus.sid2text[sid] for sid in full_gallery_sids]
    sid2fullidx = {sid: i for i, sid in enumerate(full_gallery_sids)}
    print(f"[EVAL] FULL gallery size = {len(full_gallery_sids)}")

    # 2) model (FlagModel) — encode_queries/encode_corpus 自带推荐 pooling（不用你手写 CLS）
    use_fp16 = torch.cuda.is_available()
    model = FlagModel(
        DEFAULT_MODEL_NAME,
        query_instruction_for_retrieval=QUERY_INSTRUCTION,
        query_instruction_format="{}{}",
        use_fp16=use_fp16,
    )

    # 3) encode FULL gallery once
    full_gallery_embs = encode_corpus_flagmodel(
        full_gallery_texts, model, batch_size=DEFAULT_D_BS, max_length=DEFAULT_MAX_D_LEN
    )
    print(f"[EVAL] Encoded FULL gallery: {tuple(full_gallery_embs.shape)}")

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

        print(f"[EVAL] Loaded {len(raw_queries)} FULL {split} queries.")

        query_embs_full = encode_queries_flagmodel(
            raw_queries, model, batch_size=DEFAULT_Q_BS, max_length=DEFAULT_MAX_Q_LEN
        )

        eval_sets[split] = {
            "gold_sids_list": gold_sids_list,
            "is_pure_no_event": is_pure_flags,
            "q_types": q_types,
            "query_embs_full": query_embs_full,
        }

    # 5) combos + metrics
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

        eff_idx_full = [sid2fullidx[sid] for sid in eff_gallery_sids]
        eff_gallery_embs = full_gallery_embs[eff_idx_full]
        sid2effidx = {sid: i for i, sid in enumerate(eff_gallery_sids)}

        mode_name = f"neg{int(include_neg)}_ne{int(include_no_event_sids)}_dq{int(drop_no_event_q)}"
        print("\n" + "=" * 80)
        print(f"[EVAL] MODE = {mode_name} | gallery_size = {len(eff_gallery_sids)}")

        for split in ("val", "test"):
            gold_sids_list = eval_sets[split]["gold_sids_list"]
            is_pure_flags = eval_sets[split]["is_pure_no_event"]
            q_types = eval_sets[split]["q_types"]
            query_embs_full = eval_sets[split]["query_embs_full"]

            valid_entries = []
            for i, (sids, is_pure, q_type) in enumerate(zip(gold_sids_list, is_pure_flags, q_types)):
                if drop_no_event_q and is_pure:
                    continue
                indices = [sid2effidx[sid] for sid in sids if sid in sid2effidx]
                if not indices:
                    continue
                valid_entries.append({"orig_idx": i, "gold_indices": indices, "q_type": q_type})

            if not valid_entries:
                print(f"[EVAL] split={split}: 0 valid queries, skip.")
                continue

            base_all = [e["orig_idx"] for e in valid_entries]
            gold_all = [e["gold_indices"] for e in valid_entries]
            q_all = query_embs_full[base_all]

            metrics_all = compute_retrieval_metrics(q_all, eff_gallery_embs, gold_all, ks=(1, 5, 10))
            summary_metrics[(mode_name, "all", split)] = metrics_all

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
                metrics_sub = compute_retrieval_metrics(q_sub, eff_gallery_embs, gold_sub, ks=(1, 5, 10))
                summary_metrics[(mode_name, family, split)] = metrics_sub

            m = metrics_all
            print(f"[EVAL] {split} | N={len(base_all)} | R1={m['Recall@1']:.4f} R5={m['Recall@5']:.4f} R10={m['Recall@10']:.4f} MRR10={m['MRR@10']:.4f} nDCG10={m['nDCG@10']:.4f}")

    pretty_print_summary(summary_metrics, log_path=log_path)
    if log_path:
        print(f"\n[EVAL] log saved to: {log_path}")


if __name__ == "__main__":
    main()
