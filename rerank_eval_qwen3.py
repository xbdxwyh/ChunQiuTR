# rerank_eval_qwen3.py
# -*- coding: utf-8 -*-
# Standalone: dense retrieval + Qwen3-Reranker rerank evaluation
# Requires: transformers>=4.51.0, torch, tqdm, and your project src/ package.

import os
import sys
import json
import time
import argparse
import logging
import platform
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM
from tqdm import tqdm

# ===== import your project code =====
from src.ChunQiuDataset import (
    ChunqiuEvalDataset,
    load_splits,
    build_corpus_index,
)
from src.retrieval_utils import encode_texts_bert
from src.models_temporal_dual import wrap_query_with_instruction


# --------------------------
# Logging helpers (timestamped, tqdm-friendly)
# --------------------------
def _safe_filename(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    # keep only safe chars
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in s)


class TqdmLogHandler(logging.Handler):
    """A logging handler that uses tqdm.write so it won't break progress bars."""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            pass


def setup_logger(args) -> Tuple[logging.Logger, str, str]:
    """
    Create a timestamped log file and a logger that writes:
      - console via tqdm.write (no tqdm break)
      - file via FileHandler (timestamped lines)
    Returns: (logger, log_path, run_id)
    """
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_tag = _safe_filename(args.run_name)
    run_id = f"{run_tag + '_' if run_tag else ''}{ts}"

    log_path = args.log_path
    if log_path is None:
        log_path = os.path.join(args.output_dir, f"rerank_eval_{run_id}.log")

    logger = logging.getLogger("rerank_eval_qwen3")
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # file handler (always timestamped)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logger.level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # console handler via tqdm.write
    ch = TqdmLogHandler()
    ch.setLevel(logger.level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger, log_path, run_id


def log_args_and_env(logger: logging.Logger, args, device: torch.device):
    logger.info("=" * 90)
    logger.info("[RUN] rerank_eval_qwen3.py")
    logger.info("[RUN] time              = %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("[RUN] cwd               = %s", os.getcwd())
    logger.info("[RUN] CUDA_VISIBLE_DEVICES = %s", os.environ.get("CUDA_VISIBLE_DEVICES"))
    logger.info("[ENV] python            = %s", sys.version.replace("\n", " "))
    logger.info("[ENV] platform          = %s", platform.platform())
    logger.info("[ENV] torch             = %s", torch.__version__)
    logger.info("[ENV] transformers      = %s", transformers.__version__)
    logger.info("[ENV] device            = %s", str(device))
    logger.info("[ENV] cuda_available    = %s", str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        try:
            logger.info("[ENV] cuda_device_name  = %s", torch.cuda.get_device_name(0))
        except Exception:
            pass
    logger.info("[ARGS] %s", json.dumps(vars(args), ensure_ascii=False, indent=2))
    logger.info("=" * 90)


# --------------------------
# Qwen3 Reranker wrapper
# --------------------------
class Qwen3Reranker:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        device: str = "cuda",
        max_length: int = 4096,
        torch_dtype=None,
        attn_implementation: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

        kwargs = {}
        # transformers 新版建议 dtype；老版可能只支持 torch_dtype
        if torch_dtype is not None:
            kwargs["dtype"] = torch_dtype
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation

        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        except TypeError:
            # fallback for older transformers
            if "dtype" in kwargs:
                kwargs.pop("dtype", None)
            if torch_dtype is not None:
                kwargs["torch_dtype"] = torch_dtype
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        except ImportError as e:
            # if someone toggled flash_attention_2 and flash_attn isn't usable
            if kwargs.get("attn_implementation", None) == "flash_attention_2":
                if self.logger:
                    self.logger.warning("[RR] flash_attention_2 unavailable; fallback to default attention. reason: %s", str(e))
                kwargs.pop("attn_implementation", None)
                model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            else:
                raise

        self.model = model.to(device).eval()
        self.device = self.model.device
        self.max_length = max_length

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        if self.token_false_id is None or self.token_true_id is None:
            raise ValueError("Cannot find token ids for 'yes'/'no' in tokenizer.")

        self.prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
            "<|im_start|>user\n"
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def _format(self, instruction: Optional[str], query: str, doc: str) -> str:
        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _process_inputs(self, texts: List[str]):
        max_len = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        inputs = self.tokenizer(
            texts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_len,
        )
        for i, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ids + self.suffix_tokens

        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        return {k: v.to(self.device) for k, v in inputs.items()}

    @torch.no_grad()
    def score_batch(
        self,
        instruction: Optional[str],
        queries: List[str],
        docs: List[str],
    ) -> List[float]:
        assert len(queries) == len(docs)
        texts = [self._format(instruction, q, d) for q, d in zip(queries, docs)]
        inputs = self._process_inputs(texts)
        logits = self.model(**inputs).logits[:, -1, :]
        true_v = logits[:, self.token_true_id]
        false_v = logits[:, self.token_false_id]
        two = torch.stack([false_v, true_v], dim=1)
        probs_yes = torch.softmax(two, dim=1)[:, 1]
        return probs_yes.detach().float().cpu().tolist()


# --------------------------
# Utils
# --------------------------
def build_tokenizer_and_flags(ckpt_dir: str):
    try:
        config = AutoConfig.from_pretrained(ckpt_dir, trust_remote_code=True)
        model_type = getattr(config, "model_type", "")
    except Exception:
        model_type = os.path.basename(os.path.abspath(ckpt_dir))

    is_qwen = "qwen" in str(model_type).lower()

    if is_qwen:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
        tokenizer.padding_side = "left"
        pooling = "last_token"
    else:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        pooling = "cls"

    return tokenizer, is_qwen, pooling, model_type


def metrics_from_rankings(rankings: List[List[int]], gold_indices_list: List[List[int]], ks=(1, 5, 10)) -> Dict[str, float]:
    Q = len(rankings)
    recalls = {k: 0 for k in ks}
    mrr10 = 0.0

    for qi in range(Q):
        gold = gold_indices_list[qi]
        if not gold:
            continue
        top10 = rankings[qi][:10]

        for k in ks:
            cut = min(k, len(top10))
            if any(idx in top10[:cut] for idx in gold):
                recalls[k] += 1

        rank = None
        for r, idx in enumerate(top10, start=1):
            if idx in gold:
                rank = r
                break
        if rank is not None:
            mrr10 += 1.0 / rank

    out = {f"Recall@{k}": (recalls[k] / Q if Q else 0.0) for k in ks}
    out["MRR@10"] = (mrr10 / Q if Q else 0.0)
    return out


def dense_topk_streaming(
    query_embs: torch.Tensor,    # (Q,D) CPU
    gallery_embs: torch.Tensor,  # (G,D) CPU
    topk: int,
    show_tqdm: bool = True,
) -> Tuple[List[List[int]], List[List[float]]]:
    """Return per-query topk indices + scores (dot product), streaming to avoid (Q,G) matrix."""
    Q = query_embs.size(0)
    rankings = []
    scores = []
    gT = gallery_embs.t().contiguous()  # (D,G)

    it = range(Q)
    if show_tqdm:
        it = tqdm(it, desc=f"Dense topk (Q={Q}, k={topk})", leave=False)

    for i in it:
        q = query_embs[i]  # (D,)
        sim = torch.mv(gT.t(), q)  # (G,)  (gallery_embs @ q)
        k = min(topk, sim.numel())
        v, idx = torch.topk(sim, k=k)
        rankings.append(idx.tolist())
        scores.append(v.tolist())
    return rankings, scores


def rerank_from_dense_candidates(
    raw_queries: List[str],
    gold_indices_list: List[List[int]],
    query_embs: torch.Tensor,            # (Q,D) CPU
    gallery_embs: torch.Tensor,          # (G,D) CPU
    gallery_texts: List[str],
    reranker: Qwen3Reranker,
    rerank_instruction: Optional[str],
    topn: int = 50,
    reranker_bs: int = 8,
    alpha: float = 0.0,
    show_tqdm: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Compute baseline (dense) + reranked metrics.
    alpha: combine score = (1-alpha)*P_yes + alpha*map_dense_to_[0,1]
    """
    # 1) dense candidates
    dense_rankings, dense_scores = dense_topk_streaming(
        query_embs, gallery_embs, topk=max(10, topn), show_tqdm=show_tqdm
    )
    dense_metrics = metrics_from_rankings([r[:10] for r in dense_rankings], gold_indices_list)

    # 2) flatten pairs for reranking (only topn per query)
    Q = len(raw_queries)
    cand_offsets = [0]
    flat_qidx = []
    flat_didx = []
    flat_dense = []
    for qi in range(Q):
        idxs = dense_rankings[qi][:topn]
        vs = dense_scores[qi][:topn]
        flat_qidx.extend([qi] * len(idxs))
        flat_didx.extend(idxs)
        flat_dense.extend(vs)
        cand_offsets.append(len(flat_qidx))

    # 3) score in batches (streaming)
    flat_scores = [0.0] * len(flat_qidx)
    steps = list(range(0, len(flat_qidx), reranker_bs))
    it = steps
    if show_tqdm:
        it = tqdm(it, desc=f"Rerank scoring (pairs={len(flat_qidx)}, bs={reranker_bs})", leave=False)

    for s in it:
        e = min(s + reranker_bs, len(flat_qidx))
        batch_q = [raw_queries[flat_qidx[i]] for i in range(s, e)]
        batch_d = [gallery_texts[flat_didx[i]] for i in range(s, e)]
        batch_scores = reranker.score_batch(rerank_instruction, batch_q, batch_d)
        flat_scores[s:e] = batch_scores

    # 4) build reranked top10
    rerank_rankings = []
    for qi in range(Q):
        s, e = cand_offsets[qi], cand_offsets[qi + 1]
        didxs = flat_didx[s:e]
        rrs = flat_scores[s:e]
        dens = flat_dense[s:e]

        if alpha > 0:
            dens01 = [max(0.0, min(1.0, (x + 1.0) * 0.5)) for x in dens]
            final = [((1 - alpha) * rr + alpha * d, rr, d, didx) for rr, d, didx in zip(rrs, dens01, didxs)]
            final.sort(key=lambda x: (x[0], x[2]), reverse=True)
        else:
            final = [(rr, rr, 0.0, didx) for rr, didx in zip(rrs, didxs)]
            final.sort(key=lambda x: x[0], reverse=True)

        rerank_rankings.append([x[3] for x in final[:10]])

    rerank_metrics = metrics_from_rankings(rerank_rankings, gold_indices_list)

    return {
        "dense": dense_metrics,
        "rerank": rerank_metrics,
    }


def parse_args():
    p = argparse.ArgumentParser("Standalone rerank evaluation (dense + Qwen3-Reranker)")

    # dataset paths
    p.add_argument("--meta_path", type=str, default="dataset/chunqiu_meta_sid_fixed.json")
    p.add_argument("--queries_path", type=str, default="dataset/queries_all_labeledv3.jsonl")
    p.add_argument("--splits_path", type=str, default="dataset/time_splits_by_month_v1.json")

    # dual-encoder ckpt
    p.add_argument("--ckpt_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_query_len", type=int, default=128)
    p.add_argument("--max_doc_len", type=int, default=256)

    # qwen query instruction for encoder
    p.add_argument(
        "--task_description",
        type=str,
        default=(
            "Given a classical Chinese query about the Spring and Autumn Annals, "
            "retrieve relevant passages that describe the corresponding historical events."
        ),
    )

    # reranker settings
    p.add_argument("--reranker_name", type=str, default="Qwen/Qwen3-Reranker-0.6B")
    p.add_argument("--reranker_maxlen", type=int, default=4096)
    p.add_argument("--reranker_bs", type=int, default=8)
    p.add_argument("--rerank_topn", type=int, default=50)
    p.add_argument("--rerank_alpha", type=float, default=0.0)
    p.add_argument("--rerank_instruction", type=str, default=None)

    # eval combos (same meaning as your evaluate)
    p.add_argument("--eval_include_neg_samples", action="store_true", default=False)
    p.add_argument("--eval_include_no_event_sids", action="store_true", default=False)
    p.add_argument("--eval_drop_no_event_queries", action="store_true", default=False)

    # for quick debug
    p.add_argument("--max_eval_queries", type=int, default=0, help="0 means all; else truncate eval queries to N.")

    # logging / outputs
    p.add_argument("--output_dir", type=str, default="rerank_logs", help="Directory to save logs/results.")
    p.add_argument("--run_name", type=str, default="", help="Optional tag for file names (safe chars recommended).")
    p.add_argument("--log_path", type=str, default=None, help="Optional explicit log file path.")
    p.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--save_json", action="store_true", default=True, help="Save a machine-readable results JSON.")
    p.add_argument("--no_extra_tqdm", action="store_true", default=False, help="Disable extra tqdm (dense/rerank), keep others.")

    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    # device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # setup logger (timestamped)
    logger, log_path, run_id = setup_logger(args)
    log_args_and_env(logger, args, device)

    show_extra_tqdm = (not args.no_extra_tqdm)

    # tokenizer + model type detect
    tokenizer, is_qwen, pooling, model_type = build_tokenizer_and_flags(args.ckpt_dir)
    logger.info("[ENC] ckpt_dir   = %s", args.ckpt_dir)
    logger.info("[ENC] model_type = %s", model_type)
    logger.info("[ENC] is_qwen    = %s", is_qwen)
    logger.info("[ENC] pooling    = %s", pooling)

    # load encoder (HF AutoModel)
    encoder = AutoModel.from_pretrained(args.ckpt_dir, trust_remote_code=True).to(device).eval()

    # build corpus
    splits_by_sort_key = load_splits(args.splits_path)
    corpus = build_corpus_index(args.meta_path, splits_by_sort_key)

    full_gallery_sids = sorted(corpus.sid2text.keys())
    full_gallery_texts = [corpus.sid2text[sid] for sid in full_gallery_sids]
    sid2fullidx = {sid: i for i, sid in enumerate(full_gallery_sids)}
    logger.info("[DATA] FULL gallery size = %d", len(full_gallery_sids))

    # encode full gallery embs
    full_gallery_embs = encode_texts_bert(
        full_gallery_texts,
        tokenizer,
        encoder,
        device=device,
        max_length=args.max_doc_len,
        batch_size=64,
        pool_mode=pooling,
    )
    logger.info("[ENC] FULL gallery embs: %s (on CPU)", str(tuple(full_gallery_embs.shape)))

    # init reranker
    rr_dtype = torch.float16 if str(device).startswith("cuda") else None
    rr_attn = None  # 不强开 flash_attention_2
    reranker = Qwen3Reranker(
        model_name=args.reranker_name,
        device=str(device),
        max_length=args.reranker_maxlen,
        torch_dtype=rr_dtype,
        attn_implementation=rr_attn,
        logger=logger,
    )
    logger.info("[RR] reranker=%s | maxlen=%d | bs=%d | topN=%d | alpha=%.4f",
                args.reranker_name, args.reranker_maxlen, args.reranker_bs, args.rerank_topn, args.rerank_alpha)

    # determine rerank instruction
    rerank_instruction = args.rerank_instruction or args.task_description
    logger.info("[RR] rerank_instruction = %s", rerank_instruction)

    results = {
        "run_id": run_id,
        "log_path": log_path,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "env": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "splits": {},
    }

    # evaluate splits
    for split in ("val", "test"):
        split_t0 = time.time()

        eval_ds = ChunqiuEvalDataset(
            meta_path=args.meta_path,
            queries_path=args.queries_path,
            splits_path=args.splits_path,
            split=split,
            include_no_event_queries=True,
        )

        raw_queries = []
        gold_sids_list = []
        is_pure_no_event_flags = []
        q_types = []

        for q in eval_ds.queries:
            raw_queries.append(q["query"])
            gold_sids_list.append(q["pos_sids"])
            is_pure_no_event_flags.append(q.get("is_pure_no_event", False))
            q_types.append(q.get("type", "point"))

        if args.max_eval_queries and args.max_eval_queries > 0:
            raw_queries = raw_queries[: args.max_eval_queries]
            gold_sids_list = gold_sids_list[: args.max_eval_queries]
            is_pure_no_event_flags = is_pure_no_event_flags[: args.max_eval_queries]
            q_types = q_types[: args.max_eval_queries]

        logger.info("")
        logger.info("[SPLIT] %s: loaded queries = %d", split, len(raw_queries))

        # wrap for qwen encoder
        if is_qwen:
            encode_texts = [wrap_query_with_instruction(args.task_description, q) for q in raw_queries]
        else:
            encode_texts = raw_queries

        query_embs_full = encode_texts_bert(
            encode_texts,
            tokenizer,
            encoder,
            device=device,
            max_length=args.max_query_len,
            batch_size=args.batch_size,
            pool_mode=pooling,
        )
        logger.info("[ENC] %s query embs: %s (on CPU)", split, str(tuple(query_embs_full.shape)))

        # build effective gallery
        include_neg = bool(args.eval_include_neg_samples)
        include_no_event_sids = bool(args.eval_include_no_event_sids)
        drop_no_event_q = bool(args.eval_drop_no_event_queries)

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
        eff_gallery_texts = [corpus.sid2text[sid] for sid in eff_gallery_sids]
        sid2effidx = {sid: i for i, sid in enumerate(eff_gallery_sids)}

        logger.info("[DATA] effective gallery size = %d (include_neg=%s, include_no_event_sids=%s)",
                    len(eff_gallery_sids), include_neg, include_no_event_sids)

        # build valid eval queries
        valid_entries = []
        for i, (sids, is_pure, _q_type) in enumerate(zip(gold_sids_list, is_pure_no_event_flags, q_types)):
            if drop_no_event_q and is_pure:
                continue
            indices = [sid2effidx[sid] for sid in sids if sid in sid2effidx]
            if not indices:
                continue
            valid_entries.append((i, indices))

        if not valid_entries:
            logger.warning("[WARN] no valid queries under current flags, skip split=%s.", split)
            continue

        base_indices = [x[0] for x in valid_entries]
        gold_indices = [x[1] for x in valid_entries]
        raw_queries_eff = [raw_queries[i] for i in base_indices]
        query_embs_eff = query_embs_full[base_indices]

        logger.info("[EVAL] #valid queries = %d (drop_no_event_q=%s)", len(raw_queries_eff), drop_no_event_q)

        # run dense + rerank
        out = rerank_from_dense_candidates(
            raw_queries=raw_queries_eff,
            gold_indices_list=gold_indices,
            query_embs=query_embs_eff,
            gallery_embs=eff_gallery_embs,
            gallery_texts=eff_gallery_texts,
            reranker=reranker,
            rerank_instruction=rerank_instruction,
            topn=args.rerank_topn,
            reranker_bs=args.reranker_bs,
            alpha=args.rerank_alpha,
            show_tqdm=show_extra_tqdm,
        )

        d = out["dense"]
        r = out["rerank"]

        logger.info("-" * 90)
        logger.info("[RESULT] SPLIT=%s", split)
        logger.info("Baseline Dense:  R@1=%.6f  R@5=%.6f  R@10=%.6f  MRR@10=%.6f",
                    d["Recall@1"], d["Recall@5"], d["Recall@10"], d["MRR@10"])
        logger.info("After Rerank:    R@1=%.6f  R@5=%.6f  R@10=%.6f  MRR@10=%.6f",
                    r["Recall@1"], r["Recall@5"], r["Recall@10"], r["MRR@10"])
        logger.info("[CONFIG] include_neg_samples=%s | include_no_event_sids=%s | drop_no_event_queries=%s",
                    include_neg, include_no_event_sids, drop_no_event_q)
        logger.info("[TIME] split_elapsed_sec=%.2f", time.time() - split_t0)
        logger.info("-" * 90)

        results["splits"][split] = {
            "n_loaded_queries": len(raw_queries),
            "n_valid_queries": len(raw_queries_eff),
            "effective_gallery_size": len(eff_gallery_sids),
            "flags": {
                "include_neg_samples": include_neg,
                "include_no_event_sids": include_no_event_sids,
                "drop_no_event_queries": drop_no_event_q,
            },
            "metrics": out,
            "elapsed_sec": time.time() - split_t0,
        }

    elapsed = time.time() - t0
    logger.info("[DONE] total_elapsed_sec=%.2f", elapsed)
    logger.info("[DONE] log_path=%s", log_path)

    if args.save_json:
        os.makedirs(args.output_dir, exist_ok=True)
        json_path = os.path.join(args.output_dir, f"results_{run_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("[DONE] results_json=%s", json_path)


if __name__ == "__main__":
    main()
