# eval_gte.py
# HF official usage for thenlper/gte-large-zh:
# embeddings = outputs.last_hidden_state[:, 0], then L2 normalize.
# pip install transformers torch
# python method_eval_gte.py dataset

import os
import argparse
from datetime import datetime
from contextlib import nullcontext
from typing import List
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from src.ChunQiuDataset import ChunqiuEvalDataset, load_splits, build_corpus_index
from src.retrieval_utils import compute_retrieval_metrics
from src.method_eval_utils import pretty_print_summary


DEFAULT_META    = "chunqiu_meta_sid_fixed.json"
DEFAULT_QUERIES = "queries_all_labeledv3.jsonl"
DEFAULT_SPLITS  = "time_splits_by_month_v1.json"

DEFAULT_MODEL_NAME = "/amax/wangyh/pretrained/gte-large-zh"  # 你也可以换成本地路径
DEFAULT_MAX_Q_LEN = 128
DEFAULT_MAX_D_LEN = 256
DEFAULT_Q_BS = 32
DEFAULT_D_BS = 32

WRITE_LOG = True
OUT_ROOT = "model_outputs_results"

def sanitize_model_tag(model_name_or_path: str) -> str:
    s = model_name_or_path.rstrip("/")
    if os.path.exists(s):
        s = os.path.basename(s)
    else:
        s = s.replace("/", "_")
    for ch in [" ", ":", "\\"]:
        s = s.replace(ch, "_")
    return s


@torch.inference_mode()
def encode_gte_cls(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    max_length: int,
    use_fp16: bool = False,
) -> torch.Tensor:
    """
    GTE 官方口径：取 outputs.last_hidden_state[:, 0] 作为 embedding，再做 L2 normalize
    返回 torch.FloatTensor (N, D)，在 CPU 上
    """
    model.eval()
    all_embs = []

    amp_ctx = torch.cuda.amp.autocast if (device.type == "cuda" and use_fp16) else nullcontext

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        batch_dict = tokenizer(
            batch,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

        with amp_ctx():
            outputs = model(**batch_dict)
            embs = outputs.last_hidden_state[:, 0]  # (B, H)  <-- 官方写法

        embs = embs.float()
        embs = F.normalize(embs, p=2, dim=1)
        all_embs.append(embs.detach().cpu())

    return torch.cat(all_embs, dim=0)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default="dataset", help="dataset directory")
    ap.add_argument("--data_dir", dest="data_dir_opt", default=None)

    ap.add_argument("--model_name_or_path", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--max_q_len", type=int, default=DEFAULT_MAX_Q_LEN)
    ap.add_argument("--max_d_len", type=int, default=DEFAULT_MAX_D_LEN)
    ap.add_argument("--q_bs", type=int, default=DEFAULT_Q_BS)
    ap.add_argument("--d_bs", type=int, default=DEFAULT_D_BS)

    ap.add_argument("--use_fp16", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true")
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

    model_tag = sanitize_model_tag(args.model_name_or_path)
    out_dir = os.path.join(OUT_ROOT, f"outputs-{model_tag}")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"eval_{model_tag}_{now_tag}.log") if WRITE_LOG else None

    log_f = open(log_path, "a", encoding="utf-8") if log_path else None

    def log_print(msg: str):
        print(msg)
        if log_f:
            log_f.write(msg + "\n")
            log_f.flush()

    log_print("[EVAL] device = " + str(device))
    log_print("[EVAL] model  = " + str(args.model_name_or_path))
    log_print(f"[EVAL] trust_remote_code={args.trust_remote_code} use_fp16={args.use_fp16}")
    log_print(f"[EVAL] max_q_len={args.max_q_len} max_d_len={args.max_d_len} q_bs={args.q_bs} d_bs={args.d_bs}")

    if log_f:
        log_f.write("=" * 80 + "\n")
        log_f.write(f"[RUN] {now_tag}\n")
        log_f.write(f"device={device}\n")
        log_f.write(f"model={args.model_name_or_path}\n")
        log_f.write(f"pooling=cls (official), trust_remote_code={args.trust_remote_code}, use_fp16={args.use_fp16}\n")
        log_f.write(f"max_q_len={args.max_q_len}, max_d_len={args.max_d_len}\n")
        log_f.write(f"q_bs={args.q_bs}, d_bs={args.d_bs}\n")
        log_f.write("=" * 80 + "\n")
        log_f.flush()

    # 1) corpus
    splits_by_sort_key = load_splits(splits_path)
    corpus = build_corpus_index(meta_path, splits_by_sort_key)

    # 2) FULL gallery
    full_gallery_sids = sorted(corpus.sid2text.keys())
    full_gallery_texts = [corpus.sid2text[sid] for sid in full_gallery_sids]
    sid2fullidx = {sid: i for i, sid in enumerate(full_gallery_sids)}
    log_print(f"[EVAL] FULL gallery size = {len(full_gallery_sids)}")

    # 3) load tokenizer/model
    log_print("[EVAL] Loading tokenizer/model ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code).to(device)
    log_print("[EVAL] Loaded.")

    # 4) encode FULL gallery once
    log_print("[EVAL] Encoding FULL gallery (CLS pooling) ...")
    full_gallery_embs = encode_gte_cls(
        full_gallery_texts, tokenizer, model, device,
        batch_size=args.d_bs,
        max_length=min(args.max_d_len, 512),  # 官方 max seq=512
        use_fp16=args.use_fp16,
    )
    log_print(f"[EVAL] Encoded FULL gallery: {tuple(full_gallery_embs.shape)}")

    # 5) encode FULL queries for val/test
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

        log_print(f"[EVAL] Loaded {len(raw_queries)} FULL {split} queries.")
        log_print(f"[EVAL] Encoding FULL {split} queries (CLS pooling) ...")

        query_embs_full = encode_gte_cls(
            raw_queries, tokenizer, model, device,
            batch_size=args.q_bs,
            max_length=min(args.max_q_len, 512),
            use_fp16=args.use_fp16,
        )
        log_print(f"[EVAL] Encoded FULL {split} queries: {tuple(query_embs_full.shape)}")

        eval_sets[split] = {
            "gold_sids_list": gold_sids_list,
            "is_pure_no_event": is_pure_flags,
            "q_types": q_types,
            "query_embs_full": query_embs_full,
        }

    # 6) combos + summary
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
        log_print("\n" + "=" * 80)
        log_print(f"[EVAL] MODE = {mode_name} | gallery_size = {len(eff_gallery_sids)}")

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
                log_print(f"[EVAL] split={split}: 0 valid queries, skip.")
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
            log_print(f"[EVAL] {split} | N={len(base_all)} | R1={m['Recall@1']:.4f} R5={m['Recall@5']:.4f} "
                      f"R10={m['Recall@10']:.4f} MRR10={m['MRR@10']:.4f} nDCG10={m['nDCG@10']:.4f}")

    pretty_print_summary(summary_metrics, log_path=log_path)

    if log_path:
        log_print(f"\n[EVAL] log saved to: {log_path}")

    # if log_f:
    #     log_f.close()


if __name__ == "__main__":
    main()
