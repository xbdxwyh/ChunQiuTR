# method_eval_os_sparse_mlv1.py
# pip install transformers huggingface_hub tqdm numpy
#
# Run:
#   CUDA_VISIBLE_DEVICES=0 python method_eval_os_sparse_mlv1.py
# or:
#   CUDA_VISIBLE_DEVICES=0 python method_eval_os_sparse_mlv1.py \
#       --val_dir  sparse_method/exports_sparse_val \
#       --test_dir sparse_method/exports_sparse_test

import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from huggingface_hub import hf_hub_download

from src.method_eval_utils import pretty_print_summary

# ---------------------------
# Defaults (match your layout)
# ---------------------------
DEFAULT_VAL_DIR  = "sparse_method/exports_sparse_val"
DEFAULT_TEST_DIR = "sparse_method/exports_sparse_test"

DEFAULT_MODEL = "/amax/wangyh/pretrained/opensearch-neural-sparse-encoding-multilingual-v1"
MODEL_TAG = "os_sparse_mlv1"

OUT_ROOT = "model_outputs_results"
WRITE_LOG = True


# ---------------------------
# IO helpers
# ---------------------------
def read_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def log_print(msg: str, log_path: Optional[str] = None):
    print(msg)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


# ---------------------------
# Neural sparse: idf + doc CSC index
# ---------------------------
def load_idf_vector(model_name: str, tokenizer, idf_path: Optional[str] = None) -> np.ndarray:
    """
    Load idf.json and convert to idf_vector[vocab_size]
    """
    if idf_path is None:
        idf_path = hf_hub_download(repo_id=model_name, filename="idf.json")
    with open(idf_path, "r", encoding="utf-8") as f:
        idf = json.load(f)

    vocab_size = tokenizer.vocab_size
    idf_vec = np.zeros((vocab_size,), dtype=np.float32)

    # model card uses tokenizer._convert_token_to_id_with_added_voc(token)
    # here we use convert_tokens_to_ids; fallback to vocab lookup if needed
    for tok, w in idf.items():
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is None or tid < 0 or tid >= vocab_size:
            continue
        idf_vec[tid] = float(w)

    return idf_vec


def encode_docs_build_csc(
    model,
    tokenizer,
    doc_texts: List[str],
    device: torch.device,
    max_d_len: int,
    batch_size: int,
    prune_ratio: float,
    use_fp16: bool,
    activation: str = "v1",          # NEW: 激活模式 v1 / v3
    log_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build CSC-like inverted index:
      col_ptr[vocab_size+1], row_idx[nnz], val[nnz]
    such that postings for token t are slice row_idx[col_ptr[t]:col_ptr[t+1]].

    activation:
      - "v1": multilingual-v1 风格，values = log(1 + relu(max_logits))，再按 prune_ratio 二次剪枝
      - "v3": doc-v3 风格，values = log(1 + log(1 + relu(max_logits)))，不再做二次剪枝
    """
    vocab_size = tokenizer.vocab_size
    special_ids = set(getattr(tokenizer, "all_special_ids", []))

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    model.eval()
    log_print(f"[INDEX] Build doc sparse CSC: docs={len(doc_texts)}, vocab={vocab_size}", log_path)
    log_print(f"[INDEX] max_d_len={max_d_len}, bs={batch_size}, prune_ratio={prune_ratio}, "
              f"fp16={use_fp16}, activation={activation}", log_path)  # NEW: 打印激活模式

    # NOTE: logits shape (B, L, V). This is heavy but manageable for ~20k docs offline.
    # Keep bs moderate.
    with torch.inference_mode():
        for st in tqdm(range(0, len(doc_texts), batch_size), desc="Encoding docs"):
            batch = doc_texts[st: st + batch_size]
            feat = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_d_len,
                return_tensors="pt",
                return_token_type_ids=False,
            )
            feat = {k: v.to(device) for k, v in feat.items()}

            if use_fp16 and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(**feat).logits
            else:
                logits = model(**feat).logits

            # (B, L, V) -> (B, V)
            mask = feat["attention_mask"].unsqueeze(-1)  # (B, L, 1)
            values = torch.max(logits * mask, dim=1).values

            # === 激活函数：v1 vs v3 ===
            if activation == "v3":
                # doc-v3: log(1 + log(1 + relu(x)))
                values = torch.log1p(torch.log1p(torch.relu(values)))
            else:
                # multilingual-v1 以及其他旧模型: log(1 + relu(x))
                values = torch.log1p(torch.relu(values))

            # zero out special tokens
            if special_ids:
                sid_list = torch.tensor(sorted(list(special_ids)), device=values.device, dtype=torch.long)
                values.index_fill_(1, sid_list, 0.0)

            # === 二次剪枝：只对 v1 做，v3 不再按 prune_ratio 削一遍 ===
            if activation != "v3":
                maxv = values.max(dim=1).values.unsqueeze(1) * float(prune_ratio)
                values = values * (values > maxv)

            values = values.cpu()

            # extract non-zeros per sample
            for i in range(values.size(0)):
                doc_idx = st + i
                nz = torch.nonzero(values[i], as_tuple=False).squeeze(1)
                if nz.numel() == 0:
                    continue
                w = values[i, nz]

                cols.extend(nz.tolist())
                rows.extend([doc_idx] * nz.numel())
                vals.extend(w.tolist())

            # free
            del logits, values, feat

    nnz = len(vals)
    log_print(f"[INDEX] nnz = {nnz}", log_path)

    # Convert to numpy
    cols_np = np.asarray(cols, dtype=np.int32)
    rows_np = np.asarray(rows, dtype=np.int32)
    vals_np = np.asarray(vals, dtype=np.float32)

    # Sort by col (stable), then build col_ptr
    order = np.argsort(cols_np, kind="mergesort")
    cols_s = cols_np[order]
    rows_s = rows_np[order]
    vals_s = vals_np[order]

    counts = np.bincount(cols_s, minlength=vocab_size).astype(np.int64)
    col_ptr = np.zeros((vocab_size + 1,), dtype=np.int64)
    np.cumsum(counts, out=col_ptr[1:])

    return col_ptr, rows_s, vals_s



def load_or_build_doc_index(
    cache_path: str,
    model,
    tokenizer,
    doc_texts: List[str],
    device: torch.device,
    max_d_len: int,
    batch_size: int,
    prune_ratio: float,
    use_fp16: bool,
    activation: str = "v1",                 # NEW
    log_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if os.path.isfile(cache_path):
        log_print(f"[INDEX] Loading cached CSC index: {cache_path}", log_path)
        data = np.load(cache_path, allow_pickle=False)
        col_ptr = data["col_ptr"]
        rows_s = data["rows"]
        vals_s = data["vals"]
        return col_ptr, rows_s, vals_s

    log_print(f"[INDEX] Cache not found, building: {cache_path}", log_path)
    col_ptr, rows_s, vals_s = encode_docs_build_csc(
        model=model,
        tokenizer=tokenizer,
        doc_texts=doc_texts,
        device=device,
        max_d_len=max_d_len,
        batch_size=batch_size,
        prune_ratio=prune_ratio,
        use_fp16=use_fp16,
        activation=activation,              # NEW: 传给 encode_docs_build_csc
        log_path=log_path,
    )
    ensure_dir(os.path.dirname(cache_path))
    np.savez_compressed(cache_path, col_ptr=col_ptr, rows=rows_s, vals=vals_s)
    log_print(f"[INDEX] Saved CSC index: {cache_path}", log_path)
    return col_ptr, rows_s, vals_s



# ---------------------------
# Query encoding + retrieval
# ---------------------------
def encode_query_inference_free(
    tokenizer,
    idf_vec: np.ndarray,
    query: str,
    max_q_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (token_ids, token_weights) for query sparse vector.
    """
    feat = tokenizer(
        query,
        padding=False,
        truncation=True,
        max_length=max_q_len,
        return_token_type_ids=False,
        return_attention_mask=False,
    )
    ids = np.asarray(feat["input_ids"], dtype=np.int32)
    if ids.size == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)

    # one-hot style: unique ids only
    ids_u = np.unique(ids)
    # remove special ids
    special_ids = getattr(tokenizer, "all_special_ids", [])
    if special_ids:
        mask = ~np.isin(ids_u, np.asarray(special_ids, dtype=np.int32))
        ids_u = ids_u[mask]

    w = idf_vec[ids_u]
    keep = w > 0
    ids_u = ids_u[keep]
    w = w[keep].astype(np.float32)

    return ids_u, w


def retrieve_topkprime(
    token_ids: np.ndarray,
    token_w: np.ndarray,
    col_ptr: np.ndarray,
    rows_s: np.ndarray,
    vals_s: np.ndarray,
    num_docs: int,
    topk_prime: int,
) -> List[int]:
    """
    Sparse dot product:
      score[d] += sum_{t in q} q_w[t] * doc_w[d,t]
    Return doc indices sorted by score desc, only for score>0, limited to topk_prime.
    """
    if token_ids.size == 0:
        return []

    scores = np.zeros((num_docs,), dtype=np.float32)

    # accumulate
    for tid, qw in zip(token_ids.tolist(), token_w.tolist()):
        s = col_ptr[tid]
        e = col_ptr[tid + 1]
        if s >= e:
            continue
        drows = rows_s[s:e]
        dvals = vals_s[s:e]
        # drows are unique within this posting (one entry per doc for this token)
        scores[drows] += (qw * dvals)

    cand = np.flatnonzero(scores > 0)
    if cand.size == 0:
        return []

    if cand.size > topk_prime:
        top_idx = np.argpartition(scores[cand], -topk_prime)[-topk_prime:]
        docs = cand[top_idx]
    else:
        docs = cand

    docs = docs[np.argsort(scores[docs])[::-1]]
    return docs.tolist()


# ---------------------------
# Metrics from rankings (supports post-filter)
# ---------------------------
def eval_from_rankings(
    rankings_full: List[List[int]],
    gold_indices_full: List[List[int]],
    q_types: List[str],
    is_pure_no_event: List[bool],
    doc_type: List[str],
    include_neg: bool,
    include_no_event_sids: bool,
    drop_no_event_q: bool,
    ks=(1, 5, 10),
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Return metrics for families: all/point/window.
    """
    # doc mask
    ok = np.zeros((len(doc_type),), dtype=np.bool_)
    for i, t in enumerate(doc_type):
        if t == "event":
            ok[i] = True
        elif t == "no_event" and include_no_event_sids:
            ok[i] = True
        elif t == "neg_comment" and include_neg:
            ok[i] = True

    # collect per family
    families = {
        "all":   [],
        "point": [],
        "window": [],
    }

    # build valid query list + family routing
    for qi in range(len(rankings_full)):
        if drop_no_event_q and is_pure_no_event[qi]:
            continue

        # gold indices within effective gallery
        gold_eff = [g for g in gold_indices_full[qi] if 0 <= g < len(ok) and ok[g]]
        if not gold_eff:
            continue

        qt = q_types[qi]
        fam = "point" if qt == "point" else "window"

        families["all"].append((qi, gold_eff))
        families[fam].append((qi, gold_eff))

    def compute_for_list(pairs: List[Tuple[int, List[int]]]) -> Dict[str, float]:
        if not pairs:
            return {}
        hits = {k: 0 for k in ks}
        mrr = 0.0
        ndcg = 0.0          # NEW
        n = 0
        discount = 1.0 / np.log2(np.arange(2, 12))  # rank=1..10 -> 1/log2(rank+1)

        for qi, gold_eff in pairs:
            gold_set = set(gold_eff)

            # filtered top10 (we only need up to max(ks))
            maxk = max(ks)
            filtered = []
            for d in rankings_full[qi]:
                if ok[d]:
                    filtered.append(d)
                    if len(filtered) >= maxk:
                        break

            # Recall@k
            for k in ks:
                topk = filtered[:k]
                if any(d in gold_set for d in topk):
                    hits[k] += 1

            # MRR@10
            rr = 0.0
            for rank, d in enumerate(filtered[:10], start=1):
                if d in gold_set:
                    rr = 1.0 / rank
                    break
            mrr += rr

            # nDCG@10 (binary relevance, multiple gold are equivalent)  # NEW
            dcg = 0.0
            top10 = filtered[:10]
            for r, d in enumerate(top10, start=1):
                if d in gold_set:
                    dcg += float(discount[r - 1])
            ideal_m = min(len(gold_set), 10)
            idcg = float(discount[:ideal_m].sum()) if ideal_m > 0 else 0.0
            if idcg > 0:
                ndcg += (dcg / idcg)

            n += 1

        out = {}
        for k in ks:
            out[f"Recall@{k}"] = hits[k] / max(n, 1)
        out["MRR@10"] = mrr / max(n, 1)
        out["_N"] = n
        out["nDCG@10"] = ndcg / max(n, 1)   # NEW
        return out

    results = {}
    for fam, pairs in families.items():
        r = compute_for_list(pairs)
        if r:
            results[fam] = r
    return results


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_dir", default=DEFAULT_VAL_DIR)
    ap.add_argument("--test_dir", default=DEFAULT_TEST_DIR)
    ap.add_argument("--val_split_name", default="val")
    ap.add_argument("--test_split_name", default="test")

    ap.add_argument("--model_name", default=DEFAULT_MODEL)
    ap.add_argument("--idf_path", default=os.path.join(DEFAULT_MODEL, "idf.json"))

    ap.add_argument("--max_q_len", type=int, default=128)
    ap.add_argument("--max_d_len", type=int, default=256)
    ap.add_argument("--doc_bs", type=int, default=8)
    ap.add_argument("--prune_ratio", type=float, default=0.1)
    ap.add_argument("--topk_prime", type=int, default=2000)

    ap.add_argument("--use_fp16", action="store_true")
    ap.add_argument("--device", default="cuda")

    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    now_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(OUT_ROOT, f"outputs_{MODEL_TAG}")
    log_dir = os.path.join(out_dir, f"logs_{now_tag}")
    ensure_dir(log_dir)
    log_path = os.path.join(log_dir, f"eval_{MODEL_TAG}.log") if WRITE_LOG else None

    log_print("=" * 80, log_path)
    log_print(f"[RUN] {now_tag}", log_path)
    log_print(f"[EVAL] device={device}", log_path)
    log_print(f"[EVAL] model={args.model_name}", log_path)
    log_print(f"[EVAL] max_q_len={args.max_q_len}, max_d_len={args.max_d_len}", log_path)
    log_print(f"[EVAL] doc_bs={args.doc_bs}, prune_ratio={args.prune_ratio}, topk_prime={args.topk_prime}", log_path)
    log_print(f"[EVAL] val_dir={args.val_dir}", log_path)
    log_print(f"[EVAL] test_dir={args.test_dir}", log_path)
    log_print("=" * 80, log_path)

    # ---- Load exported docs (use test as canonical), verify val matches ----
    test_docs_path = os.path.join(args.test_dir, f"docs.{args.test_split_name}.jsonl")
    val_docs_path  = os.path.join(args.val_dir,  f"docs.{args.val_split_name}.jsonl")
    assert os.path.isfile(test_docs_path), f"missing: {test_docs_path}"
    assert os.path.isfile(val_docs_path),  f"missing: {val_docs_path}"

    docs_test = read_jsonl(test_docs_path)
    docs_val  = read_jsonl(val_docs_path)

    sids_test = [int(x["sid"]) for x in docs_test]
    sids_val  = [int(x["sid"]) for x in docs_val]

    if sids_test != sids_val:
        raise RuntimeError(
            "Val/Test docs order differs. Please re-export so val/test use IDENTICAL docs (same sids in same order)."
        )

    doc_texts = [x["text"] for x in docs_test]
    doc_type  = [x.get("type", "event") for x in docs_test]
    sid2idx   = {sid: i for i, sid in enumerate(sids_test)}
    num_docs  = len(doc_texts)

    log_print(f"[DATA] gallery docs = {num_docs}", log_path)

    # ---- Load model/tokenizer + idf ----
    log_print("[MODEL] Loading tokenizer/model...", log_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    model.to(device)
    if args.use_fp16 and device.type == "cuda":
        model.half()

    idf_vec = load_idf_vector(args.model_name, tokenizer, idf_path=args.idf_path)
    log_print(f"[MODEL] idf_vec loaded: shape={idf_vec.shape}", log_path)

    # ---- Build/load doc CSC index (cached under outputs dir) ----
    # cache_dir = os.path.join(out_dir, "cache")
    # ensure_dir(cache_dir)

    model_name_lower = os.path.basename(str(args.model_name)).lower()
    if "doc-v3" in model_name_lower or "encoding-doc-v3" in model_name_lower:
        activation = "v3"
    else:
        activation = "v1"

    log_print(f"[MODEL] inferred activation={activation} for model_name={args.model_name}", log_path)


    pr_tag = str(args.prune_ratio).replace(".", "p")
    cache_path = os.path.join(
        log_dir,
        f"doc_csc_{MODEL_TAG}_D{num_docs}_L{args.max_d_len}_pr{pr_tag}.npz"
    )
    col_ptr, rows_s, vals_s = load_or_build_doc_index(
        cache_path=cache_path,
        model=model,
        tokenizer=tokenizer,
        doc_texts=doc_texts,
        device=device,
        max_d_len=args.max_d_len,
        batch_size=args.doc_bs,
        prune_ratio=args.prune_ratio,
        use_fp16=args.use_fp16,
        activation=activation,          # NEW: 传递激活模式
        log_path=log_path,
    )

    # ---- Load queries (val/test), encode + retrieve once on FULL gallery ----
    def load_split_queries(export_dir: str, split_name: str):
        qpath = os.path.join(export_dir, f"queries.{split_name}.jsonl")
        assert os.path.isfile(qpath), f"missing: {qpath}"
        qs = read_jsonl(qpath)

        q_texts = [x["query"] for x in qs]
        q_types = [x.get("query_type") or "point" for x in qs]
        gold_sids_list = [x["gold_sids"] for x in qs]

        gold_indices_full = []
        is_pure_no_event = []

        for gold_sids in gold_sids_list:
            idxs = [sid2idx[sid] for sid in gold_sids if sid in sid2idx]
            gold_indices_full.append(idxs)

            # pure no_event if ALL gold sids are type == "no_event"
            types = set(doc_type[i] for i in idxs) if idxs else set()
            is_pure = (len(types) == 1 and ("no_event" in types))
            is_pure_no_event.append(is_pure)

        return q_texts, q_types, gold_indices_full, is_pure_no_event

    eval_sets = {}
    for split in ("val", "test"):
        if split == "val":
            q_texts, q_types, gold_idx, pure_flags = load_split_queries(args.val_dir, args.val_split_name)
        else:
            q_texts, q_types, gold_idx, pure_flags = load_split_queries(args.test_dir, args.test_split_name)

        log_print(f"[DATA] Loaded {len(q_texts)} {split} queries", log_path)

        rankings_full: List[List[int]] = []
        for q in tqdm(q_texts, desc=f"Retrieving {split}"):
            tids, tw = encode_query_inference_free(tokenizer, idf_vec, q, max_q_len=args.max_q_len)
            top_docs = retrieve_topkprime(
                token_ids=tids,
                token_w=tw,
                col_ptr=col_ptr,
                rows_s=rows_s,
                vals_s=vals_s,
                num_docs=num_docs,
                topk_prime=args.topk_prime,
            )
            rankings_full.append(top_docs)

        eval_sets[split] = {
            "q_types": q_types,
            "gold_indices_full": gold_idx,
            "is_pure_no_event": pure_flags,
            "rankings_full": rankings_full,
        }

    # ---- Evaluate combos (same as your dense scripts) ----
    combos = []
    for include_neg in (False, True):
        for include_no_event_sids in (False, True):
            for drop_no_event_q in (False, True):
                # same constraint you used before
                if (not include_no_event_sids) and (not drop_no_event_q):
                    continue
                combos.append((include_neg, include_no_event_sids, drop_no_event_q))

    summary_metrics = {}

    for include_neg, include_no_event_sids, drop_no_event_q in combos:
        mode_name = f"neg{int(include_neg)}_ne{int(include_no_event_sids)}_dq{int(drop_no_event_q)}"
        log_print("\n" + "=" * 80, log_path)
        log_print(f"[EVAL] MODE = {mode_name}", log_path)

        for split in ("val", "test"):
            pack = eval_sets[split]
            results = eval_from_rankings(
                rankings_full=pack["rankings_full"],
                gold_indices_full=pack["gold_indices_full"],
                q_types=pack["q_types"],
                is_pure_no_event=pack["is_pure_no_event"],
                doc_type=doc_type,
                include_neg=include_neg,
                include_no_event_sids=include_no_event_sids,
                drop_no_event_q=drop_no_event_q,
                ks=(1, 5, 10),
            )

            if not results or "all" not in results:
                log_print(f"[EVAL] {split}: 0 valid queries, skip.", log_path)
                continue

            # save
            for fam in ("all", "point", "window"):
                if fam in results:
                    summary_metrics[(mode_name, fam, split)] = results[fam]

            m = results["all"]
            log_print(
                f"[EVAL] {split} | N={m['_N']} | R1={m['Recall@1']:.4f} R5={m['Recall@5']:.4f} "
                f"R10={m['Recall@10']:.4f} MRR10={m['MRR@10']:.4f} nDCG10={m['nDCG@10']:.4f}",
                log_path,
            )

    pretty_print_summary(summary_metrics, log_path=log_path)
    log_print(f"\n[EVAL] log saved to: {log_path}", log_path)


if __name__ == "__main__":
    main()
