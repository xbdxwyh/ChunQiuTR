# eval_bge_m3.py
# pip install FlagEmbedding
# Run (single GPU recommended):
#   CUDA_VISIBLE_DEVICES=0 python ./method_eval_bge3.py --data_dir ./dataset/

import os
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
from FlagEmbedding import BGEM3FlagModel

from src.ChunQiuDataset import ChunqiuEvalDataset, load_splits, build_corpus_index
from src.retrieval_utils import compute_retrieval_metrics
from src.method_eval_utils import pretty_print_summary


DEFAULT_META    = "chunqiu_meta_sid_fixed.json"
DEFAULT_QUERIES = "queries_all_labeledv3.jsonl"
DEFAULT_SPLITS  = "time_splits_by_month_v1.json"

DEFAULT_MODEL_NAME = "/amax/wangyh/pretrained/bge-m3"  # or "BAAI/bge-m3"
DEFAULT_MAX_Q_LEN = 128
DEFAULT_MAX_D_LEN = 256
DEFAULT_Q_BS = 32
DEFAULT_D_BS = 32

WRITE_LOG = True
OUT_ROOT = "model_outputs_results"
MODEL_TAG = "bge3"

def encode_bge_m3_dense(texts, model: BGEM3FlagModel, batch_size: int, max_length: int):
    """
    返回 torch.FloatTensor (N, D)，L2-normalized
    """
    out = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    dense = out["dense_vecs"]  # usually numpy (N, D)
    if isinstance(dense, torch.Tensor):
        embs = dense.float()
    else:
        embs = torch.from_numpy(dense).float()
    embs = F.normalize(embs, p=2, dim=1)
    return embs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default="dataset", help="dataset directory")
    ap.add_argument("--data_dir", dest="data_dir_opt", default=None, help="same as positional data_dir")
    args = ap.parse_args()
    data_dir = args.data_dir_opt or args.data_dir

    meta_path   = os.path.join(data_dir, DEFAULT_META)
    queries_path = os.path.join(data_dir, DEFAULT_QUERIES)
    splits_path  = os.path.join(data_dir, DEFAULT_SPLITS)

    assert os.path.isfile(meta_path),   f"meta not found: {meta_path}"
    assert os.path.isfile(queries_path), f"queries not found: {queries_path}"
    assert os.path.isfile(splits_path),  f"splits not found: {splits_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    now_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(OUT_ROOT, f"outputs_{MODEL_TAG}")
    os.makedirs(out_dir, exist_ok=True)
    model_name = os.path.basename(DEFAULT_MODEL_NAME.rstrip("/")).replace("/", "_")
    log_path = os.path.join(out_dir, f"eval_{model_name}_{now_tag}.log") if WRITE_LOG else None

    print("[EVAL] device =", device)
    print("[EVAL] model  =", DEFAULT_MODEL_NAME)
    print("[EVAL] max_q_len =", DEFAULT_MAX_Q_LEN, "max_d_len =", DEFAULT_MAX_D_LEN)
    print("[EVAL] q_bs =", DEFAULT_Q_BS, "d_bs =", DEFAULT_D_BS)

    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"[RUN] {now_tag}\n")
            f.write(f"device={device}\n")
            f.write(f"model={DEFAULT_MODEL_NAME}\n")
            f.write(f"max_q_len={DEFAULT_MAX_Q_LEN}, max_d_len={DEFAULT_MAX_D_LEN}\n")
            f.write(f"q_bs={DEFAULT_Q_BS}, d_bs={DEFAULT_D_BS}\n")
            f.write("=" * 80 + "\n")

    # 1) corpus
    splits_by_sort_key = load_splits(splits_path)
    corpus = build_corpus_index(meta_path, splits_by_sort_key)

    # 2) FULL gallery
    full_gallery_sids = sorted(corpus.sid2text.keys())
    full_gallery_texts = [corpus.sid2text[sid] for sid in full_gallery_sids]
    sid2fullidx = {sid: i for i, sid in enumerate(full_gallery_sids)}
    print(f"[EVAL] FULL gallery size = {len(full_gallery_sids)}")

    # bge-m3 (official encoder)
    bge = BGEM3FlagModel(DEFAULT_MODEL_NAME, use_fp16=(device.type == "cuda"))

    full_gallery_embs = encode_bge_m3_dense(full_gallery_texts, bge, batch_size=DEFAULT_D_BS, max_length=DEFAULT_MAX_D_LEN)
    print(f"[EVAL] Encoded FULL gallery: {tuple(full_gallery_embs.shape)}")

    # 3) encode FULL queries for val/test
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

        query_embs_full = encode_bge_m3_dense(raw_queries, bge, batch_size=DEFAULT_Q_BS, max_length=DEFAULT_MAX_Q_LEN)

        eval_sets[split] = {
            "gold_sids_list": gold_sids_list,
            "is_pure_no_event": is_pure_flags,
            "q_types": q_types,
            "query_embs_full": query_embs_full,
        }

    # 4) combos + summary
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
