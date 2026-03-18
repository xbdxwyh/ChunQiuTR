# eval_e5_mistral_7b_instruct.py
# Zero-shot eval for intfloat/e5-mistral-7b-instruct on ChunQiu benchmark (same eval combos)
#
# Install:
#   pip install -U transformers accelerate torch
#
# Run (use 2x GPUs):
#   CUDA_VISIBLE_DEVICES=0,1 python method_eval_e5_mistral.py ./dataset
#
# Run (use more GPUs if you want):
#   CUDA_VISIBLE_DEVICES=0,1,2,3 python method_eval_e5_mistral.py ./dataset
#
# Official notes:
#   - Queries must include one-sentence instruction: "Instruct: {task}\nQuery: {query}"
#   - No need to add instructions to documents
#   - Pooling: last_token_pool
#   - Using inputs longer than 4096 tokens is not recommended

import os
import json
import argparse
from datetime import datetime
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from src.ChunQiuDataset import ChunqiuEvalDataset, load_splits, build_corpus_index
from src.retrieval_utils import compute_retrieval_metrics
from src.method_eval_utils import pretty_print_summary
from tqdm import tqdm

# ====== defaults ======
DEFAULT_META     = "chunqiu_meta_sid_fixed.json"
DEFAULT_QUERIES  = "queries_all_labeledv3.jsonl"
DEFAULT_SPLITS   = "time_splits_by_month_v1.json"

DEFAULT_MODEL_NAME = "/amax/wangyh/pretrained/e5-mistral-7b-instruct"  # or local path
DEFAULT_TASK_DESCRIPTION = (
    # "Given a time-aware historical query about the Chunqiu corpus, retrieve relevant passages from the corpus."
    "Given a classical Chinese query about the Spring and Autumn Annals, retrieve relevant passages that describe the corresponding historical events."
)

# keep same as your other evals by default; can increase if you want
DEFAULT_MAX_Q_LEN = 128
DEFAULT_MAX_D_LEN = 256

# safer defaults for 7B
DEFAULT_Q_BS = 16
DEFAULT_D_BS = 8

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


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    # official snippet
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]


def wrap_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


def _first_param_device(model) -> torch.device:
    # device_map sharded models: pick first non-meta param
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def encode_texts_lasttok(
    texts: List[str],
    tokenizer,
    model,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    """
    Return: (N, D) CPU float32, L2-normalized
    """
    model.eval()

    # official: do not use longer than 4096
    max_length = min(int(max_length), 4096)

    input_device = _first_param_device(model)

    outs: List[torch.Tensor] = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        batch_dict = tokenizer(
            batch_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(input_device) for k, v in batch_dict.items()}

        outputs = model(**batch_dict)
        embs = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        embs = embs.float()
        embs = F.normalize(embs, p=2, dim=1)
        outs.append(embs.detach().cpu())

    return torch.cat(outs, dim=0)


def encode_queries(queries, tokenizer, model, batch_size, max_length, task) -> torch.Tensor:
    texts = [wrap_instruct(task, q) for q in queries]
    return encode_texts_lasttok(texts, tokenizer, model, batch_size, max_length)


def encode_docs(texts, tokenizer, model, batch_size, max_length) -> torch.Tensor:
    # docs are raw (no instruction)
    return encode_texts_lasttok(texts, tokenizer, model, batch_size, max_length)


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(1)

    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default="dataset", help="dataset directory")
    ap.add_argument("--data_dir", dest="data_dir_opt", default=None)

    ap.add_argument("--model_name_or_path", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--task", default=DEFAULT_TASK_DESCRIPTION)

    ap.add_argument("--max_q_len", type=int, default=DEFAULT_MAX_Q_LEN)
    ap.add_argument("--max_d_len", type=int, default=DEFAULT_MAX_D_LEN)
    ap.add_argument("--q_bs", type=int, default=DEFAULT_Q_BS)
    ap.add_argument("--d_bs", type=int, default=DEFAULT_D_BS)

    # multi-gpu (accelerate) loading
    ap.add_argument("--device_map", default="auto", help="auto | balanced | sequential | none")
    ap.add_argument("--max_memory_json", default=None, help='Optional JSON, e.g. {"cuda:0":"46GiB","cuda:1":"46GiB"}')
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])

    args = ap.parse_args()

    data_dir = args.data_dir_opt or args.data_dir
    meta_path    = os.path.join(data_dir, DEFAULT_META)
    queries_path = os.path.join(data_dir, DEFAULT_QUERIES)
    splits_path  = os.path.join(data_dir, DEFAULT_SPLITS)

    assert os.path.isfile(meta_path), f"meta not found: {meta_path}"
    assert os.path.isfile(queries_path), f"queries not found: {queries_path}"
    assert os.path.isfile(splits_path), f"splits not found: {splits_path}"

    now_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_tag = sanitize_model_tag(args.model_name_or_path)
    out_dir = os.path.join(OUT_ROOT, f"outputs-{model_tag}")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"eval_{model_tag}_{now_tag}.log") if WRITE_LOG else None

    print("[EVAL] model  =", args.model_name_or_path)
    print("[EVAL] task   =", args.task)
    print("[EVAL] pooling: last_token_pool")
    print("[EVAL] max_q_len =", args.max_q_len, "max_d_len =", args.max_d_len)
    print("[EVAL] q_bs =", args.q_bs, "d_bs =", args.d_bs)
    print("[EVAL] device_map =", args.device_map)

    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"[RUN] {now_tag}\n")
            f.write(f"model={args.model_name_or_path}\n")
            f.write(f"task={args.task}\n")
            f.write("pooling=last_token_pool\n")
            f.write("query_format=Instruct: {task}\\nQuery: {query}\n")
            f.write("doc_format=raw\n")
            f.write(f"device_map={args.device_map}\n")
            f.write(f"dtype={args.dtype}\n")
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

    # 2) load model (multi-gpu)
    print("[EVAL] Loading tokenizer/model ...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    max_memory = None
    if args.max_memory_json:
        max_memory = json.loads(args.max_memory_json)

    device_map = None if args.device_map.lower() == "none" else args.device_map

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        device_map=device_map,
        max_memory=max_memory,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()

    print("[EVAL] Loaded. first_param_device =", _first_param_device(model))

    # 3) encode FULL gallery once
    print("[EVAL] Encoding FULL gallery ...")
    full_gallery_embs = encode_docs(
        full_gallery_texts, tokenizer, model,
        batch_size=args.d_bs, max_length=args.max_d_len
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
        print(f"[EVAL] Encoding FULL {split} queries ...")

        query_embs_full = encode_queries(
            raw_queries, tokenizer, model,
            batch_size=args.q_bs, max_length=args.max_q_len,
            task=args.task
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
            print(f"[EVAL] {split} | N={len(base_all)} | R1={m['Recall@1']:.4f} R5={m['Recall@5']:.4f} "
                  f"R10={m['Recall@10']:.4f} MRR10={m['MRR@10']:.4f} nDCG10={m['nDCG@10']:.4f}")

    pretty_print_summary(summary_metrics, log_path=log_path)
    if log_path:
        print(f"\n[EVAL] log saved to: {log_path}")


if __name__ == "__main__":
    main()
