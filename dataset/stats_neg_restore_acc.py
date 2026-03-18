# stats_neg_restore_acc.py
# -*- coding: utf-8 -*-

import os
import sys
import json
import re
import argparse
from collections import defaultdict
from datetime import datetime

from tqdm import tqdm


# ========= 0) path: import ChunQiuDataset (可选，只是保持你项目风格) =========
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "../src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# 不强依赖 ChunQiuDataset，但保留接口风格（未来你要按 split 统计时会方便）
try:
    from ChunQiuDataset import load_splits  # noqa: F401
except Exception:
    load_splits = None


# ========= 1) 作者名规范化（简繁/别名统一成一个 key） =========
CANONICAL_AUTHORS = ["顾栋高", "魏了翁", "孔颖达", "杜预", "吕祖谦"]

AUTHOR_VARIANTS = {
    "顾栋高": ["顾栋高", "顧棟高", "顧棟髙", "顧東高", "顧栋高"],
    "魏了翁": ["魏了翁"],
    "孔颖达": ["孔颖达", "孔穎達", "孔頴達"],
    "杜预": ["杜预", "杜預"],
    "吕祖谦": ["吕祖谦", "呂祖謙", "呂祖譣"],
}

def normalize_author(raw: str) -> str:
    if raw is None:
        return "UNKNOWN"
    s = str(raw)
    s = re.sub(r"\s+", "", s)

    # 先用“包含关系”兜底：raw 里只要出现某个变体，就归到 canonical
    for canon, vars_ in AUTHOR_VARIANTS.items():
        for v in vars_:
            if v in s:
                return canon

    # 再做一点轻微清洗（可按需加）
    s = re.sub(r"[《》〈〉“”\"'（）()【】\[\]]", "", s)
    return s if s else "UNKNOWN"


# ========= 2) 判定规则 =========
def is_hit(neg_block: dict, criterion: str, score_thr: float, score_key: str, strong_key: str) -> bool:
    strong = bool(neg_block.get(strong_key, False))
    if criterion == "strong":
        return strong

    # score 可能是 None / str
    score = neg_block.get(score_key, None)
    try:
        score = float(score) if score is not None else None
    except Exception:
        score = None

    score_ok = (score is not None) and (score >= score_thr)

    if criterion == "score":
        return score_ok
    if criterion == "strong_or_score":
        return strong or score_ok

    raise ValueError(f"Unknown criterion={criterion}")


# ========= 3) 主流程 =========
def main():
    ap = argparse.ArgumentParser("Compute neg_sample restore accuracy by compiler/author")
    ap.add_argument("--meta_path", default="dataset/chunqiu_meta_sid_fixed.json")
    ap.add_argument("--output_dir", default="outputs_neg_restore_acc")
    ap.add_argument("--log_name", default="auto")  # auto -> time stamp

    # 你关心的：强匹配 or score 阈值
    ap.add_argument(
        "--criterion",
        type=str,
        default="strong",
        choices=["strong", "score", "strong_or_score"],
        help="strong: only match_strong==True; score: match_score>=thr; strong_or_score: either",
    )
    ap.add_argument("--score_thr", type=float, default=0.9, help="Used when criterion involves score")
    ap.add_argument("--score_key", type=str, default="match_score")
    ap.add_argument("--strong_key", type=str, default="match_strong")

    # 作者字段名：你 demo 里是 compiler
    ap.add_argument("--author_key", type=str, default="compiler")

    # 可选：把失败样本 dump 出来方便人工看
    ap.add_argument("--dump_failed_jsonl", type=str, default="")

    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = args.log_name
    if log_name == "auto" or not log_name:
        log_name = f"neg_restore_acc_{now}.log"
    log_path = os.path.join(args.output_dir, log_name)

    def log(msg: str):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("=" * 80)
    log(f"[{now}] stats_neg_restore_acc")
    log(f"meta_path     = {args.meta_path}")
    log(f"criterion     = {args.criterion}")
    log(f"score_thr     = {args.score_thr}")
    log(f"score_key     = {args.score_key}")
    log(f"strong_key    = {args.strong_key}")
    log(f"author_key    = {args.author_key}")
    log(f"dump_failed   = {args.dump_failed_jsonl}")
    log("=" * 80)

    with open(args.meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    time_mapping = meta.get("time_mapping", {})
    if not isinstance(time_mapping, dict) or not time_mapping:
        log("[ERR] meta['time_mapping'] is empty or not a dict.")
        return

    stats = defaultdict(lambda: {"N": 0, "A": 0})
    other_stats = {"N": 0, "A": 0}
    total_N, total_A = 0, 0

    failed_out = None
    if args.dump_failed_jsonl:
        failed_out = open(args.dump_failed_jsonl, "w", encoding="utf-8")

    # 遍历每个月份节点
    for time_text, info in tqdm(time_mapping.items(), desc="Scanning time_mapping"):
        if not isinstance(info, dict):
            continue

        # A) version 级别：info["versions"][i]["neg_sample"]
        versions = info.get("versions", [])
        if isinstance(versions, list):
            for ver in versions:
                if not isinstance(ver, dict):
                    continue
                for neg_block in ver.get("neg_sample", []) or []:
                    if not isinstance(neg_block, dict):
                        continue
                    raw_author = neg_block.get(args.author_key)
                    author = normalize_author(raw_author)
                    hit = is_hit(neg_block, args.criterion, args.score_thr, args.score_key, args.strong_key)

                    total_N += 1
                    total_A += int(hit)

                    if author in CANONICAL_AUTHORS:
                        stats[author]["N"] += 1
                        stats[author]["A"] += int(hit)
                    else:
                        other_stats["N"] += 1
                        other_stats["A"] += int(hit)

                    if (not hit) and failed_out is not None:
                        failed_out.write(json.dumps({
                            "time_text": time_text,
                            "author_raw": raw_author,
                            "author_norm": author,
                            "match_strong": neg_block.get(args.strong_key, False),
                            "match_score": neg_block.get(args.score_key, None),
                            "origin_event": neg_block.get("origin_event", ""),
                            "work": neg_block.get("work", ""),
                            "source": neg_block.get("source", ""),
                            "comment": neg_block.get("comment", ""),
                        }, ensure_ascii=False) + "\n")

        # B) time 级别：info["neg_samples"]（注意有的 meta 用复数）
        for neg_block in info.get("neg_samples", []) or []:
            if not isinstance(neg_block, dict):
                continue
            raw_author = neg_block.get(args.author_key)
            author = normalize_author(raw_author)
            hit = is_hit(neg_block, args.criterion, args.score_thr, args.score_key, args.strong_key)

            total_N += 1
            total_A += int(hit)

            if author in CANONICAL_AUTHORS:
                stats[author]["N"] += 1
                stats[author]["A"] += int(hit)
            else:
                other_stats["N"] += 1
                other_stats["A"] += int(hit)

            if (not hit) and failed_out is not None:
                failed_out.write(json.dumps({
                    "time_text": time_text,
                    "author_raw": raw_author,
                    "author_norm": author,
                    "match_strong": neg_block.get(args.strong_key, False),
                    "match_score": neg_block.get(args.score_key, None),
                    "origin_event": neg_block.get("origin_event", ""),
                    "work": neg_block.get("work", ""),
                    "source": neg_block.get("source", ""),
                    "comment": neg_block.get("comment", ""),
                }, ensure_ascii=False) + "\n")

    if failed_out is not None:
        failed_out.close()

    # 输出结果
    def fmt_acc(a, n):
        return 0.0 if n == 0 else (a / n)

    log("\n=== Neg restore accuracy by author ===")
    log(f"{'author':10s} | {'N':>8s} | {'A':>8s} | {'acc':>8s}")
    log("-" * 46)

    for author in CANONICAL_AUTHORS:
        n = stats[author]["N"]
        a = stats[author]["A"]
        log(f"{author:10s} | {n:8d} | {a:8d} | {fmt_acc(a, n):8.4f}")

    # 其他作者
    log("-" * 46)
    log(f"{'OTHER':10s} | {other_stats['N']:8d} | {other_stats['A']:8d} | {fmt_acc(other_stats['A'], other_stats['N']):8.4f}")
    log("=" * 46)
    log(f"{'TOTAL':10s} | {total_N:8d} | {total_A:8d} | {fmt_acc(total_A, total_N):8.4f}")
    log(f"[DONE] log saved to: {log_path}")


if __name__ == "__main__":
    main()
