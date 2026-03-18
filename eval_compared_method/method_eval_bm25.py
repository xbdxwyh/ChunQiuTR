# bm25_baseline.py
# pip install rank-bm25 numpy tqdm
# Run:
#   python method_eval_bm25.py

import os
import argparse
import re
import numpy as np
from tqdm import tqdm
from datetime import datetime

from rank_bm25 import BM25Okapi

from src.ChunQiuDataset import (
    ChunqiuEvalDataset,
    load_splits,
    build_corpus_index,
    build_eval_gallery,
)

from src.method_eval_utils import pretty_print_summary 


def zh_char_ngrams(text: str, ngram: int = 2, keep_unigram: bool = True):
    text = re.sub(r"\s+", "", text)
    chars = [c for c in text if "\u4e00" <= c <= "\u9fff"]  # 只保留汉字
    toks = []
    if keep_unigram:
        toks.extend(chars)
    if ngram >= 2 and len(chars) >= ngram:
        toks.extend(["".join(chars[i : i + ngram]) for i in range(len(chars) - ngram + 1)])
    return toks if toks else ["<EMPTY>"]


def build_bm25(gallery_texts, ngram, keep_unigram, k1, b, epsilon):
    doc_tokens = [zh_char_ngrams(t, ngram=ngram, keep_unigram=keep_unigram)
                  for t in tqdm(gallery_texts, desc="Tokenizing docs")]

    kwargs = {}
    if k1 is not None: kwargs["k1"] = k1
    if b is not None: kwargs["b"] = b
    if epsilon is not None: kwargs["epsilon"] = epsilon

    try:
        bm25 = BM25Okapi(doc_tokens, **kwargs)
    except TypeError:
        # 兼容某些版本不接受 k1/b/epsilon 的情况
        bm25 = BM25Okapi(doc_tokens)

    return bm25


def eval_one_split_with_bm25(eval_ds, bm25, sid2idx, drop_no_event_q: bool, ks=(1, 5, 10)):
    # 抽取 query & gold
    raw_queries, gold_sids_list, is_pure_flags, q_types = [], [], [], []
    for q in eval_ds.queries:
        raw_queries.append(q["query"])
        gold_sids_list.append(q["pos_sids"])
        is_pure_flags.append(q.get("is_pure_no_event", False))
        q_types.append(q.get("type", "point"))

    kept = []
    for i, (q_text, gold_sids, is_pure, q_type) in enumerate(
        zip(raw_queries, gold_sids_list, is_pure_flags, q_types)
    ):
        if drop_no_event_q and is_pure:
            continue
        gold_indices = [sid2idx[sid] for sid in gold_sids if sid in sid2idx]
        if not gold_indices:
            continue
        kept.append((q_text, gold_indices, q_type))

    if not kept:
        return {}

    max_k = max(ks)

    hit_family = {("all", k): 0 for k in ks}
    hit_family.update({("point", k): 0 for k in ks})
    hit_family.update({("window", k): 0 for k in ks})

    mrr_family = {"all": 0.0, "point": 0.0, "window": 0.0}
    cnt_family = {"all": 0, "point": 0, "window": 0}
    ndcg_family = {"all": 0.0, "point": 0.0, "window": 0.0}

    for q_text, gold_indices, q_type in tqdm(kept, desc="Scoring queries"):
        qtok = zh_char_ngrams(q_text)  # query 用同样 tokenization
        scores = np.asarray(bm25.get_scores(qtok), dtype=np.float32)

        top_idx = np.argpartition(-scores, max_k - 1)[:max_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        gold = set(gold_indices)
        fam = "point" if q_type == "point" else "window"

        cnt_family["all"] += 1
        cnt_family[fam] += 1

        for k in ks:
            if any(i in gold for i in top_idx[:k]):
                hit_family[("all", k)] += 1
                hit_family[(fam, k)] += 1

        rr = 0.0
        for rank, i in enumerate(top_idx[:10], start=1):
            if i in gold:
                rr = 1.0 / rank
                break
        mrr_family["all"] += rr
        mrr_family[fam] += rr

        # nDCG@10 (binary relevance; multiple golds are equivalent)
        dcg = 0.0
        for rank, i in enumerate(top_idx[:10], start=1):
            if i in gold:
                dcg += 1.0 / np.log2(rank + 1)

        ideal = min(len(gold), 10)
        idcg = 0.0
        for r in range(1, ideal + 1):
            idcg += 1.0 / np.log2(r + 1)

        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        ndcg_family["all"] += ndcg
        ndcg_family[fam] += ndcg


    def pack(fam):
        n = cnt_family[fam]
        if n == 0:
            return {"Recall@1": 0.0, "Recall@5": 0.0, "Recall@10": 0.0, "MRR@10": 0.0, "nDCG@10": 0.0, "N": 0}
        return {
            "Recall@1": hit_family[(fam, 1)] / n,
            "Recall@5": hit_family[(fam, 5)] / n,
            "Recall@10": hit_family[(fam, 10)] / n,
            "MRR@10": mrr_family[fam] / n,
            "nDCG@10": ndcg_family[fam] / n,
            "N": n,
        }

    return {"all": pack("all"), "point": pack("point"), "window": pack("window")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_path", default="dataset/chunqiu_meta_sid_fixed.json")
    ap.add_argument("--queries_path", default="dataset/queries_all_labeledv3.jsonl")
    ap.add_argument("--splits_path", default="dataset/time_splits_by_month_v1.json")

    # 输出/日志
    ap.add_argument("--output_dir", default="model_outputs_results/outputs_bm25")
    ap.add_argument("--log_name", default="bm25_eval.log")
    ap.add_argument("--run_label", default="")

    # BM25 参数（可不改）
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--epsilon", type=float, default=0.25)

    # tokenization 参数（对古汉语更关键）
    ap.add_argument("--ngram", type=int, default=2)
    ap.add_argument("--keep_unigram", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, args.log_name)

    splits_by_sort_key = load_splits(args.splits_path)
    corpus = build_corpus_index(args.meta_path, splits_by_sort_key)

    # 预加载 eval datasets（避免反复读文件）
    eval_ds_map = {}
    for split in ("val", "test"):
        eval_ds_map[split] = ChunqiuEvalDataset(
            meta_path=args.meta_path,
            queries_path=args.queries_path,
            splits_path=args.splits_path,
            split=split,
            include_no_event_queries=True,
        )

    # combos（保持你那套逻辑）
    combos = []
    for include_neg in (False, True):
        for include_no_event_sids in (False, True):
            for drop_no_event_q in (False, True):
                if (not include_no_event_sids) and (not drop_no_event_q):
                    continue
                combos.append((include_neg, include_no_event_sids, drop_no_event_q))

    # 为了避免重复建 BM25：按 (include_neg, include_no_event_sids) 分组
    group_keys = sorted({(a, b) for a, b, _ in combos})

    summary_metrics = {}

    # 写入 run header
    if args.run_label:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"[RUN] {args.run_label}\n")
            f.write("=" * 80 + "\n")

    for include_neg, include_no_event_sids in group_keys:
        # 这个 gallery 对 dq 无关，所以只建一次
        gallery_sids, gallery_texts, sid2idx = build_eval_gallery(
            corpus,
            include_neg_samples=include_neg,
            include_no_event_sids=include_no_event_sids,
        )

        print("\n" + "=" * 80)
        print(f"[BM25] Build index: include_neg={include_neg}, include_no_event_sids={include_no_event_sids}")
        print(f"[BM25] gallery_size={len(gallery_sids)}")

        bm25 = build_bm25(
            gallery_texts,
            ngram=args.ngram,
            keep_unigram=args.keep_unigram,
            k1=args.k1,
            b=args.b,
            epsilon=args.epsilon,
        )

        # 在这个 group 下，跑所有 dq（但只跑 combos 里存在的）
        dq_list = [dq for a, b, dq in combos if a == include_neg and b == include_no_event_sids]
        dq_list = sorted(set(dq_list))

        for drop_no_event_q in dq_list:
            mode_name = (
                f"neg{int(include_neg)}_"
                f"ne{int(include_no_event_sids)}_"
                f"dq{int(drop_no_event_q)}"
            )

            print("\n" + "-" * 80)
            print(f"[BM25] === Combo: {mode_name} ===")

            for split in ("val", "test"):
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                out = eval_one_split_with_bm25(
                    eval_ds_map[split],
                    bm25=bm25,
                    sid2idx=sid2idx,
                    drop_no_event_q=drop_no_event_q,
                    ks=(1, 5, 10),
                )

                if not out:
                    print(f"[BM25] WARNING: split={split}, combo={mode_name} has 0 valid queries, skip.")
                    continue

                # all-family 详细日志（对齐你现有风格）
                m_all = out["all"]
                r1, r5, r10, mrr10, ndcg10 = m_all["Recall@1"], m_all["Recall@5"], m_all["Recall@10"], m_all["MRR@10"], m_all["nDCG@10"]
                run_name = f"bm25_{split}_{mode_name}"

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
                print(f"bm25_k1                = {args.k1}")
                print(f"bm25_b                 = {args.b}")
                print(f"bm25_epsilon           = {args.epsilon}")
                print(f"token_ngram            = {args.ngram}")
                print(f"token_keep_unigram     = {args.keep_unigram}")
                print(f"include_neg_samples    = {include_neg}")
                print(f"include_no_event_sids  = {include_no_event_sids}")
                print(f"drop_no_event_queries  = {drop_no_event_q}")
                print("=" * 80)

                with open(log_path, "a", encoding="utf-8") as f_log:
                    f_log.write(
                        f"[{now}] [EVAL] run={run_name}, "
                        f"R@1={r1:.6f}, R@5={r5:.6f}, "
                        f"R@10={r10:.6f}, MRR@10={mrr10:.6f}, nDCG@10={ndcg10:.6f}, "
                        f"gallery_size={len(gallery_sids)}, "
                        f"#queries={m_all['N']}, "
                        f"include_neg={include_neg}, "
                        f"include_no_event_sids={include_no_event_sids}, "
                        f"drop_no_event_q={drop_no_event_q}, "
                        f"k1={args.k1}, b={args.b}, eps={args.epsilon}, "
                        f"ngram={args.ngram}, unigram={int(args.keep_unigram)}\n"
                    )

                # 收集 summary（all/point/window）
                for fam in ("all", "point", "window"):
                    summary_metrics[(mode_name, fam, split)] = {
                        "Recall@1": out[fam]["Recall@1"],
                        "Recall@5": out[fam]["Recall@5"],
                        "Recall@10": out[fam]["Recall@10"],
                        "MRR@10": out[fam]["MRR@10"],
                        "nDCG@10": out[fam]["nDCG@10"],
                    }

    # ===== 输出 summary 表（对齐你 evaluate 的格式）=====
    if not summary_metrics:
        print("[BM25] No metrics collected, skip summary table.")
        return
    pretty_print_summary(summary_metrics, log_path=log_path)
    if log_path:
        print(f"\n[EVAL] log saved to: {log_path}")


if __name__ == "__main__":
    main()
