#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Eval month-level retrieval on ZZTJ-derived subsets (e.g., 齐纪 / 晋纪).

This script builds a month-level gallery (all event lines) and auto-generates
"point" queries per month.

IMPORTANT: Default query/doc time format is *traditional reign-year* (年号纪年),
not AD year-month.

Usage example:
  CUDA_VISIBLE_DEVICES=0 python eval_zztj_month_retrieval.py \
    --jsonl_files /mnt/data/zztj-qiji-part-demo.jsonl /mnt/data/zztj-jinji-demo.jsonl \
    --subset_names 齐纪 晋纪 \
    --ckpt_dir ../pretrained/Qwen3-Embedding-0.6B/ \
    --max_query_len 128 --max_doc_len 256 --batch_size 32

If you want to fall back to AD year-month queries (easy baseline):
  ... --query_time_format ad
"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig


# ---------------------------
# Helpers
# ---------------------------

MONTH_CN = {
    1: "正月", 2: "二月", 3: "三月", 4: "四月", 5: "五月", 6: "六月",
    7: "七月", 8: "八月", 9: "九月", 10: "十月", 11: "十一月", 12: "十二月",
}
SEASON_SET = {"春", "夏", "秋", "冬"}


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def is_meta_line(obj: Dict[str, Any]) -> bool:
    # Heuristic: meta lines often have id like *_meta_*, and time like "0479-0483".
    _id = str(obj.get("id", ""))
    if "meta" in _id:
        return True
    tm = obj.get("time_meta", {})
    # If time_meta has a range, it's likely a meta line.
    if any(k in tm for k in ("start_ad_year", "end_ad_year")) and "reign" not in tm:
        return True
    return False


def int_to_cn(n: int) -> str:
    """Convert 1..99 to Chinese numerals (一,二,...,十,十一,...)."""
    digits = {0: "零", 1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九"}
    if n <= 0:
        return str(n)
    if n < 10:
        return digits[n]
    if n == 10:
        return "十"
    if n < 20:
        return "十" + digits[n % 10]
    tens, ones = divmod(n, 10)
    if ones == 0:
        return digits[tens] + "十"
    return digits[tens] + "十" + digits[ones]


def era_year_to_text(era_year: Optional[int]) -> str:
    if era_year is None:
        return ""
    try:
        y = int(era_year)
    except Exception:
        return str(era_year)
    if y == 1:
        return "元"
    return int_to_cn(y)


def format_traditional_time(tm: Dict[str, Any], use_emperor: bool = True) -> str:
    """Return something like: '晋世祖武皇帝泰始元年三月' / '建元元年正月' / '泰始元年冬'."""
    emperor = (tm.get("emperor") or "").strip()
    reign = (tm.get("reign") or "").strip()
    era_year = tm.get("era_year")
    era_year_txt = era_year_to_text(era_year)

    # month/season label
    month_label = (tm.get("lunar_month_label") or "").strip()
    if not month_label:
        # fallback from lunar_month
        m = tm.get("lunar_month")
        if isinstance(m, int) and m in MONTH_CN:
            month_label = MONTH_CN[m]
        else:
            # last fallback: season
            month_label = (tm.get("season") or "").strip()

    pieces = []
    if use_emperor and emperor:
        pieces.append(emperor)
    if reign:
        pieces.append(reign)
    if era_year_txt:
        pieces.append(era_year_txt + "年")
    else:
        # sometimes era_year missing; avoid dangling '年'
        pass
    if month_label:
        pieces.append(month_label)

    return "".join(pieces).strip()


def parse_time_string_ad(time_str: str) -> Optional[Tuple[int, int]]:
    # Expected formats: "0479-01-甲辰" or "0479-01" or "0479-...".
    parts = time_str.split("-")
    if len(parts) < 2:
        return None
    y_str, m_str = parts[0], parts[1]
    if not (y_str.isdigit() and m_str.isdigit()):
        return None
    y = int(y_str)
    m = int(m_str)
    if m < 1 or m > 12:
        return None
    return y, m


def extract_month_key(
    obj: Dict[str, Any],
    time_format: str = "traditional",
    use_emperor_in_key: bool = True,
) -> Optional[Tuple[Any, ...]]:
    """Return a hashable month key.

    - traditional: (emperor?, reign, era_year, lunar_month_label)
    - ad:          (ad_year, month)
    """
    if time_format == "traditional":
        tm = obj.get("time_meta") or {}
        reign = (tm.get("reign") or "").strip()
        era_year = tm.get("era_year")
        month_label = (tm.get("lunar_month_label") or "").strip()
        if not month_label:
            m = tm.get("lunar_month")
            if isinstance(m, int) and m in MONTH_CN:
                month_label = MONTH_CN[m]
            else:
                month_label = (tm.get("season") or "").strip()

        if not reign or era_year is None or not month_label:
            # fallback to AD parsing if time_meta insufficient
            t = parse_time_string_ad(str(obj.get("time", "")))
            if t is None:
                return None
            return (t[0], t[1])

        if use_emperor_in_key:
            emperor = (tm.get("emperor") or "").strip()
            return (emperor, reign, int(era_year), month_label)
        return (reign, int(era_year), month_label)

    # ad baseline
    t = parse_time_string_ad(str(obj.get("time", "")))
    if t is None:
        return None
    return (t[0], t[1])


def month_key_to_display(
    month_key: Tuple[Any, ...],
    time_format: str = "traditional",
    use_emperor: bool = True,
) -> str:
    if time_format == "traditional":
        # month_key could be (emperor, reign, era_year, month_label) OR fallback (ad_year, month)
        if len(month_key) >= 3 and isinstance(month_key[-2], int) and isinstance(month_key[-1], str):
            if len(month_key) == 4:
                emperor, reign, era_year, month_label = month_key
            else:
                emperor = ""
                reign, era_year, month_label = month_key
            era_txt = era_year_to_text(era_year) + "年"
            return (emperor if (use_emperor and emperor) else "") + f"{reign}{era_txt}{month_label}"
        # fallback ad
        year, month = month_key
        return f"公元{year}年{MONTH_CN.get(month, str(month) + '月')}"

    # ad
    year, month = month_key
    return f"公元{year}年{MONTH_CN.get(month, str(month) + '月')}"


# ---------------------------
# Embedding
# ---------------------------

def build_tokenizer_and_model(ckpt_dir: str, device: torch.device):
    try:
        config = AutoConfig.from_pretrained(ckpt_dir, trust_remote_code=True)
        model_type = getattr(config, "model_type", "")
    except Exception:
        model_type = os.path.basename(os.path.abspath(ckpt_dir))

    is_qwen = "qwen" in str(model_type).lower()

    if is_qwen:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
        tokenizer.padding_side = "left"
    else:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)

    model = AutoModel.from_pretrained(ckpt_dir, trust_remote_code=is_qwen).to(device)
    model.eval()
    return tokenizer, model, is_qwen, model_type


@torch.no_grad()
def encode_texts(
    tokenizer,
    model,
    texts: List[str],
    device: torch.device,
    max_len: int,
    batch_size: int,
    is_qwen: bool,
    pooling: str = "last_token",
    task_description: Optional[str] = None,
) -> torch.Tensor:
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encode", ncols=100):
        batch = texts[i: i + batch_size]

        if is_qwen and task_description:
            # Keep it lightweight; use Qwen's standard instruction wrapping
            batch = [f"{task_description}\n\nQuery/Passage: {t}" for t in batch]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc)
        last_hidden = out.last_hidden_state  # [B, L, H]

        if pooling == "cls":
            embs = last_hidden[:, 0]
        elif pooling == "mean":
            mask = enc.get("attention_mask", torch.ones(last_hidden.shape[:2], device=device))
            mask = mask.unsqueeze(-1)
            embs = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            # last_token
            mask = enc.get("attention_mask", None)
            if mask is None:
                embs = last_hidden[:, -1]
            else:
                lengths = mask.sum(dim=1) - 1
                embs = last_hidden[torch.arange(last_hidden.size(0), device=device), lengths]

        embs = torch.nn.functional.normalize(embs, p=2, dim=-1)
        all_embs.append(embs.detach().cpu())

    return torch.cat(all_embs, dim=0)


# ---------------------------
# Data building: docs & month groups
# ---------------------------

def build_docs_and_month_groups(
    records: List[Dict[str, Any]],
    subset_name: str,
    time_format: str = "traditional",
    use_emperor_in_time: bool = True,
    use_emperor_in_key: bool = True,
) -> Tuple[List[str], Dict[Tuple[Any, ...], List[int]], Dict[Tuple[Any, ...], str]]:
    """Return:
      - docs: list[str]
      - month2doc_indices: month_key -> list of doc indices
      - month2time_text: month_key -> formatted time text (traditional or AD)
    """
    docs: List[str] = []
    month2doc_indices: Dict[Tuple[Any, ...], List[int]] = {}
    month2time_text: Dict[Tuple[Any, ...], str] = {}

    for obj in records:
        if is_meta_line(obj):
            continue

        month_key = extract_month_key(obj, time_format=time_format, use_emperor_in_key=use_emperor_in_key)
        if month_key is None:
            continue

        tm = obj.get("time_meta") or {}
        if time_format == "traditional" and tm.get("reign"):
            time_text = format_traditional_time(tm, use_emperor=use_emperor_in_time)
        else:
            time_text = month_key_to_display(month_key, time_format="ad")

        day_gz = (tm.get("day_ganzhi") or "").strip()
        day_text = f"（{day_gz}）" if day_gz else ""

        # Keep passage body as-is, but prepend a consistent tag.
        title = (obj.get("title") or "").strip()
        desc = (obj.get("description") or "").strip()
        if title and (title not in desc):
            body = title + "\n" + desc
        else:
            body = desc or title

        doc = f"【{subset_name} {time_text}{day_text}】\n{body}".strip()

        idx = len(docs)
        docs.append(doc)
        month2doc_indices.setdefault(month_key, []).append(idx)
        # store representative time_text
        if month_key not in month2time_text:
            month2time_text[month_key] = time_text

    return docs, month2doc_indices, month2time_text


def build_month_queries(
    subset_name: str,
    month_keys: List[Tuple[Any, ...]],
    month2time_text: Dict[Tuple[Any, ...], str],
    time_format: str = "traditional",
    query_template: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Auto-generate point queries for each month.

    Output query schema is intentionally close to your ChunQiu "point" queries.
    """
    queries: List[Dict[str, Any]] = []
    if query_template is None:
        # default: traditional time text
        query_template = "请问{time_text}发生了什么事？"

    for i, mk in enumerate(month_keys):
        time_text = month2time_text.get(mk)
        if not time_text:
            time_text = month_key_to_display(mk, time_format=time_format)

        q = {
            "qid": f"{subset_name}-point-{i:05d}",
            "query": query_template.format(subset=subset_name, time_text=time_text),
            "time_text": time_text,
            "type": "point",
            "target_month_key": list(mk),  # for debug
        }
        queries.append(q)

    return queries


# ---------------------------
# Evaluation
# ---------------------------

def compute_metrics(sim: torch.Tensor, month2doc_indices: Dict[Tuple[Any, ...], List[int]], month_keys: List[Tuple[Any, ...]]):
    """Compute Recall@K and MRR over month-level positives.

    sim: [num_queries, num_docs]
    """
    ranks = []
    recall_at = {1: 0, 5: 0, 10: 0}

    for qi, mk in enumerate(month_keys):
        pos = set(month2doc_indices.get(mk, []))
        if not pos:
            continue

        scores = sim[qi]
        sorted_idx = torch.argsort(scores, descending=True)

        # find best rank among positives
        best_rank = None
        for r, di in enumerate(sorted_idx.tolist(), start=1):
            if di in pos:
                best_rank = r
                break

        if best_rank is None:
            continue
        ranks.append(best_rank)
        for k in recall_at:
            if best_rank <= k:
                recall_at[k] += 1

    n = len(ranks)
    if n == 0:
        return {"n": 0, "mrr": 0.0, "recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0}

    mrr = sum(1.0 / r for r in ranks) / n
    metrics = {"n": n, "mrr": mrr}
    for k, v in recall_at.items():
        metrics[f"recall@{k}"] = v / n

    return metrics


def run_eval_for_subset(
    subset_name: str,
    jsonl_path: str,
    tokenizer,
    model,
    device: torch.device,
    args,
    log_f,
):
    records = load_jsonl(jsonl_path)
    docs, month2doc_indices, month2time_text = build_docs_and_month_groups(
        records,
        subset_name=subset_name,
        time_format=args.query_time_format,
        use_emperor_in_time=args.use_emperor,
        use_emperor_in_key=args.use_emperor,
    )

    month_keys = sorted(month2doc_indices.keys(), key=lambda x: str(x))
    queries = build_month_queries(
        subset_name=subset_name,
        month_keys=month_keys,
        month2time_text=month2time_text,
        time_format=args.query_time_format,
        query_template=args.query_template,
    )

    query_texts = [q["query"] for q in queries]

    # encode
    q_emb = encode_texts(
        tokenizer=tokenizer,
        model=model,
        texts=query_texts,
        device=device,
        max_len=args.max_query_len,
        batch_size=args.batch_size,
        is_qwen=args.is_qwen,
        pooling=args.pooling,
        task_description=args.task_description,
    )
    d_emb = encode_texts(
        tokenizer=tokenizer,
        model=model,
        texts=docs,
        device=device,
        max_len=args.max_doc_len,
        batch_size=args.batch_size,
        is_qwen=args.is_qwen,
        pooling=args.pooling,
        task_description=None,
    )

    sim = torch.matmul(q_emb, d_emb.T)
    metrics = compute_metrics(sim, month2doc_indices, month_keys)

    # log
    log_f.write("\n" + "=" * 80 + "\n")
    log_f.write(f"[SUBSET] {subset_name}\n")
    log_f.write(f"jsonl_path: {jsonl_path}\n")
    log_f.write(f"#docs: {len(docs)}\n")
    log_f.write(f"#months(queries): {len(month_keys)}\n")
    log_f.write(f"time_format: {args.query_time_format}\n")
    log_f.write(f"use_emperor: {args.use_emperor}\n")
    log_f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    print(f"[SUBSET {subset_name}] metrics: {metrics}")

    # optionally dump debug top1 errors
    if args.dump_errors > 0 and metrics["n"] > 0:
        log_f.write("\n[Top errors dump]\n")
        for qi, mk in enumerate(month_keys[: args.dump_errors]):
            pos = set(month2doc_indices.get(mk, []))
            scores = sim[qi]
            top_idx = torch.argsort(scores, descending=True)[: args.error_topk].tolist()
            mk_disp = month_key_to_display(mk, time_format=args.query_time_format, use_emperor=args.use_emperor)
            log_f.write("\n" + "-" * 60 + "\n")
            log_f.write(f"Month: {mk_disp}\n")
            log_f.write(f"Query: {queries[qi]['query']}\n")
            log_f.write(f"Pos doc idx: {sorted(pos)[:10]} ...\n")
            for rank, di in enumerate(top_idx, start=1):
                hit = "HIT" if di in pos else ""
                snippet = docs[di].replace("\n", " ")[:180]
                log_f.write(f"  @{rank:<2d} {hit:<3s} di={di:<6d}  {snippet}\n")

    return metrics


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser("Eval ZZTJ month retrieval")

    parser.add_argument("--jsonl_files", nargs="+", required=True, help="List of subset jsonl files")
    parser.add_argument("--subset_names", nargs="+", required=True, help="Names for subsets, aligned with jsonl_files")

    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_query_len", type=int, default=128)
    parser.add_argument("--max_doc_len", type=int, default=256)

    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument(
        "--pooling",
        type=str,
        default=None,
        choices=["cls", "mean", "last_token"],
        help="Pooling strategy for embeddings",
    )

    parser.add_argument(
        "--task_description",
        type=str,
        default=(
            "Given a classical Chinese query about historical chronicles, "
            "retrieve relevant passages that describe the corresponding events."
        ),
        help="Instruction wrapper (mainly for Qwen)",
    )

    parser.add_argument(
        "--query_time_format",
        type=str,
        default="traditional",
        choices=["traditional", "ad"],
        help="Query/doc time format: traditional reign-year (default) or AD year-month baseline.",
    )
    parser.add_argument(
        "--no_emperor",
        dest="use_emperor",
        action="store_false",
        help="Disable emperor name in time_text (default: include if available).",
    )

    parser.set_defaults(use_emperor=True)

    parser.add_argument(
        "--query_template",
        type=str,
        default=None,
        help="Optional query template, supports {subset} and {time_text}.",
    )

    parser.add_argument("--dump_errors", type=int, default=0, help="Dump first N months' topk results into log")
    parser.add_argument("--error_topk", type=int, default=10, help="TopK shown for each dumped month")

    args = parser.parse_args()

    if len(args.jsonl_files) != len(args.subset_names):
        raise ValueError("--jsonl_files and --subset_names must have the same length")

    return args


def main():
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(args.ckpt_dir)), "zztj_eval_outputs")
    os.makedirs(args.output_dir, exist_ok=True)

    log_time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(args.output_dir, f"zztj_eval_{args.query_time_format}_{log_time_str}.txt")

    # device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # build tokenizer/model
    tokenizer, model, is_qwen, model_type = build_tokenizer_and_model(args.ckpt_dir, device)
    args.is_qwen = is_qwen

    if args.pooling is None:
        # default pooling: qwen -> last_token, bert -> cls
        args.pooling = "last_token" if is_qwen else "cls"

    print("Using device:", device)
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print(f"[CKPT] {args.ckpt_dir}")
    print(f"[MODEL] model_type={model_type} is_qwen={is_qwen} pooling={args.pooling}")
    print(f"[EVAL] query_time_format={args.query_time_format} use_emperor={args.use_emperor}")
    print(f"[LOG] {log_path}")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("ZZTJ Month Retrieval Eval\n")
        f.write(f"time: {datetime.now().isoformat()}\n")
        f.write(f"ckpt_dir: {args.ckpt_dir}\n")
        f.write(f"model_type: {model_type}\n")
        f.write(f"is_qwen: {is_qwen}\n")
        f.write(f"pooling: {args.pooling}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"max_query_len: {args.max_query_len}\n")
        f.write(f"max_doc_len: {args.max_doc_len}\n")
        f.write(f"query_time_format: {args.query_time_format}\n")
        f.write(f"use_emperor: {args.use_emperor}\n")
        if args.query_template:
            f.write(f"query_template: {args.query_template}\n")
        f.write("=" * 80 + "\n")

        all_metrics = {}
        for subset_name, jsonl_path in zip(args.subset_names, args.jsonl_files):
            metrics = run_eval_for_subset(
                subset_name=subset_name,
                jsonl_path=jsonl_path,
                tokenizer=tokenizer,
                model=model,
                device=device,
                args=args,
                log_f=f,
            )
            all_metrics[subset_name] = metrics

        f.write("\n" + "=" * 80 + "\n")
        f.write("[ALL SUMMARY]\n")
        f.write(json.dumps(all_metrics, ensure_ascii=False, indent=2) + "\n")

    print(f"\n[EVAL DONE] log written to: {log_path}")


if __name__ == "__main__":
    main()
