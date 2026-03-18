# eval_e5_large_instruct.py
# Zero-shot eval for intfloat/multilingual-e5-large-instruct on ChunQiu benchmark (same eval combos)
#
# Install:
#   pip install -U transformers torch
#
# Run:
#   CUDA_VISIBLE_DEVICES=0 python method_eval_e5_large_instruct.py ./dataset --use_fp16
#   CUDA_VISIBLE_DEVICES=0 python ./method_eval_e5_large_instruct.py dataset
# or:
#   CUDA_VISIBLE_DEVICES=0 python method_eval_e5_large_instruct.py --data_dir ./dataset --use_fp16
#
# Notes (official tutorial style):
#   - Query must be wrapped with "Instruct: {task}\nQuery: {query}"
#   - Documents are raw text (no instruction needed)
#   - Pooling: average_pool (masked mean) on last_hidden_state then L2 normalize.

import os
import argparse
from datetime import datetime
from contextlib import nullcontext
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from src.ChunQiuDataset import ChunqiuEvalDataset, load_splits, build_corpus_index
from src.retrieval_utils import compute_retrieval_metrics
from tqdm import tqdm
from src.method_eval_utils import pretty_print_summary


# ====== defaults ======
DEFAULT_META     = "chunqiu_meta_sid_fixed.json"
DEFAULT_QUERIES  = "queries_all_labeledv3.jsonl"
DEFAULT_SPLITS   = "time_splits_by_month_v1.json"

DEFAULT_MODEL_NAME = "/amax/wangyh/pretrained/multilingual-e5-large-instruct"  # or local path

# Query needs one-sentence instruction; set to your task
DEFAULT_TASK_DESCRIPTION = (
    "Given a time-aware historical query about the Chunqiu corpus, retrieve relevant passages from the corpus."
    # "Given a classical Chinese query about the Spring and Autumn Annals, retrieve relevant passages that describe the corresponding historical events."
)

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


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    # tutorial: masked mean pooling
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    denom = attention_mask.sum(dim=1)[..., None].clamp(min=1)
    return last_hidden.sum(dim=1) / denom


def wrap_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


@torch.inference_mode()
def encode_texts(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
    use_fp16: bool,
) -> torch.Tensor:
    """Return (N, D) CPU float32, L2-normalized"""
    model.eval()

    # In tutorial they use max_length=512; keep safe cap
    max_length = min(int(max_length), 512)

    amp_ctx = torch.cuda.amp.autocast if (device.type == "cuda" and use_fp16) else nullcontext

    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        batch_dict = tokenizer(
            batch_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

        with amp_ctx():
            outputs = model(**batch_dict)
            embs = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])

        embs = embs.float()
        embs = F.normalize(embs, p=2, dim=1)
        all_embs.append(embs.detach().cpu())

    return torch.cat(all_embs, dim=0)


def encode_queries_instruct(queries, tokenizer, model, device, batch_size, max_length, use_fp16, task_description) -> torch.Tensor:
    texts = [wrap_instruct(task_description, q) for q in queries]
    return encode_texts(texts, tokenizer, model, device, batch_size, max_length, use_fp16)


def encode_corpus_raw(texts, tokenizer, model, device, batch_size, max_length, use_fp16) -> torch.Tensor:
    # documents are raw (no instruction)
    return encode_texts(texts, tokenizer, model, device, batch_size, max_length, use_fp16)


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(1)

    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default="dataset", help="dataset directory")
    ap.add_argument("--data_dir", dest="data_dir_opt", default=None)

    ap.add_argument("--model_name_or_path", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--task", default=DEFAULT_TASK_DESCRIPTION, help="one-sentence instruction for wrapping queries")
    ap.add_argument("--max_q_len", type=int, default=DEFAULT_MAX_Q_LEN)
    ap.add_argument("--max_d_len", type=int, default=DEFAULT_MAX_D_LEN)
    ap.add_argument("--q_bs", type=int, default=DEFAULT_Q_BS)
    ap.add_argument("--d_bs", type=int, default=DEFAULT_D_BS)
    ap.add_argument("--use_fp16", action="store_true")
    args = ap.parse_args()

    data_dir = args.data_dir_opt or args.data_dir
    meta_path    = os.path.join(data_dir, DEFAULT_META)
    queries_path = os.path.join(data_dir, DEFAULT_QUERIES)
    splits_path  = os.path.join(data_dir, DEFAULT_SPLITS)

    assert os.path.isfile(meta_path), f"meta not found: {meta_path}"
    assert os.path.isfile(queries_path), f"queries not found: {queries_path}"
    assert os.path.isfile(splits_path), f"splits not found: {splits_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = args.use_fp16 and (device.type == "cuda")

    now_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_tag = sanitize_model_tag(args.model_name_or_path)

    out_dir = os.path.join(OUT_ROOT, f"outputs-{model_tag}")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"eval_{model_tag}_{now_tag}.log") if WRITE_LOG else None

    print("[EVAL] device =", device)
    print("[EVAL] model  =", args.model_name_or_path)
    print("[EVAL] task   =", args.task)
    print("[EVAL] pooling: average_pool (masked mean)")
    print("[EVAL] max_q_len =", args.max_q_len, "max_d_len =", args.max_d_len)
    print("[EVAL] q_bs =", args.q_bs, "d_bs =", args.d_bs, "use_fp16 =", use_fp16)

    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"[RUN] {now_tag}\n")
            f.write(f"device={device}\n")
            f.write(f"model={args.model_name_or_path}\n")
            f.write(f"task={args.task}\n")
            f.write("pooling=average_pool (masked mean)\n")
            f.write("query_format=Instruct: {task}\\nQuery: {query}\n")
            f.write("doc_format=raw\n")
            f.write(f"use_fp16={use_fp16}\n")
            f.write(f"max_q_len={args.max_q_len}, max_d_len={args.max_d_len}\n")
            f.write(f"q_bs={args.q_bs}, d_bs={args.d_bs}\n")
            f.write("=" * 80 + "\n")

    # 1) corpus
    splits_by_sort_key = load_splits(splits_path)
    corpus = build_corpus_index(meta_path, splits_by_sort_key)

    full_gallery_sids = sorted(corpus.sid2text.keys())
    full_gallery_texts = [corpus.sid2text[sid] for sid in full_gallery_sids]
    sid2fullidx = {sid: i for i, sid in enumerate(full_gallery_sids)}
    print(f"[EVAL] FULL gallery size = {len(full_gallery_sids)}")

    # 2) load model
    print("[EVAL] Loading tokenizer/model ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dtype = torch.float16 if use_fp16 else None
    model = AutoModel.from_pretrained(args.model_name_or_path, torch_dtype=dtype).to(device)
    model.eval()
    print("[EVAL] Loaded.")

    # 3) encode FULL gallery once (documents raw)
    print("[EVAL] Encoding FULL gallery (docs raw) ...")
    full_gallery_embs = encode_corpus_raw(
        full_gallery_texts, tokenizer, model, device,
        batch_size=args.d_bs, max_length=args.max_d_len, use_fp16=use_fp16
    )
    print(f"[EVAL] Encoded FULL gallery: {tuple(full_gallery_embs.shape)}")

    # 4) encode FULL queries for val/test (queries wrapped with instruct)
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
        print(f"[EVAL] Encoding FULL {split} queries (with instruction) ...")

        query_embs_full = encode_queries_instruct(
            raw_queries, tokenizer, model, device,
            batch_size=args.q_bs, max_length=args.max_q_len, use_fp16=use_fp16,
            task_description=args.task
        )
        print(f"[EVAL] Encoded FULL {split} queries: {tuple(query_embs_full.shape)}")

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
            print(f"[EVAL] {split} | N={len(base_all)} | R1={m['Recall@1']:.4f} R5={m['Recall@5']:.4f} "
                  f"R10={m['Recall@10']:.4f} MRR10={m['MRR@10']:.4f} nDCG10={m['nDCG@10']:.4f}")

    pretty_print_summary(summary_metrics, log_path=log_path)
    if log_path:
        print(f"\n[EVAL] log saved to: {log_path}")


if __name__ == "__main__":
    main()
