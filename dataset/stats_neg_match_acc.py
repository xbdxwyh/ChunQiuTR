# stats_neg_match_acc.py
# -*- coding: utf-8 -*-

import os
import json
import argparse
import logging
from datetime import datetime
from collections import defaultdict, Counter

from tqdm import tqdm


# =========================
# 1) 作者名归一化：兼容简繁/别写
# =========================

AUTHOR_CANON = {
    "GDG": "顾栋高",
    "WLW": "魏了翁",
    "KYD": "孔颖达",
    "DY":  "杜预",
    "LZQ": "吕祖谦",
    "OTHER": "其他/未知",
}

# 你关心的 5 个作者：给简繁/常见写法做包含匹配
AUTHOR_PATTERNS = {
    "GDG": ["顾栋高", "顧棟高"],
    "WLW": ["魏了翁"],  # 一般无繁体差异；如你遇到再补
    "KYD": ["孔颖达", "孔穎達"],
    "DY":  ["杜预", "杜預"],
    "LZQ": ["吕祖谦", "呂祖謙"],
}

def normalize_name(s: str) -> str:
    """尽量温和的归一化：去空白、统一一些分隔符。"""
    if not s:
        return ""
    s = str(s).strip()
    s = s.replace("\u3000", " ").replace("\t", " ").replace("\n", " ")
    s = " ".join(s.split())
    return s

def map_author(compiler_raw: str) -> str:
    """把 compiler/作者字符串映射到 GDG/WLW/...，否则 OTHER。"""
    s = normalize_name(compiler_raw)
    if not s:
        return "OTHER"
    for k, pats in AUTHOR_PATTERNS.items():
        for p in pats:
            if p in s:
                return k
    return "OTHER"


# =========================
# 2) 递归找 neg_sample（更鲁棒：适配你 meta 里不同嵌套层级）
# =========================

def iter_neg_samples(obj):
    """
    递归遍历 dict/list，遇到 key == "neg_sample" 且为 list，则 yield 每个 item。
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "neg_sample" and isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        yield item
            else:
                yield from iter_neg_samples(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from iter_neg_samples(x)


# =========================
# 3) logging：带时间戳 log 文件
# =========================

def setup_logger(log_dir: str, prefix: str = "neg_match_acc"):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{prefix}_{ts}.log")

    logger = logging.getLogger(prefix)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger, log_path


# =========================
# 4) 主逻辑：统计 N / A / acc
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_path", type=str, default="chunqiu_meta_sid_fixed.json",
                    help="包含 time_mapping 的 meta json 文件")
    ap.add_argument("--log_dir", type=str, default="logs",
                    help="log 输出目录")
    ap.add_argument("--out_json", type=str, default="",
                    help="可选：把统计结果保存成 json 的路径")
    ap.add_argument("--show_other", action="store_true",
                    help="可选：打印/记录 OTHER 的 compiler 原始名字频次，便于你补映射")
    args = ap.parse_args()

    logger, log_path = setup_logger(args.log_dir)
    logger.info("=== Neg_sample strong-match accuracy stats ===")
    logger.info(f"meta_path = {args.meta_path}")
    logger.info(f"log_path  = {log_path}")

    if not os.path.exists(args.meta_path):
        raise FileNotFoundError(f"meta_path not found: {args.meta_path}")

    with open(args.meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    time_mapping = meta.get("time_mapping", {})
    logger.info(f"time_mapping size = {len(time_mapping)}")

    # 统计结构：per-author totals
    total_by_author = defaultdict(int)
    pass_by_author = defaultdict(int)

    # debug: OTHER 的 compiler 原值
    other_compiler_counter = Counter()

    # 遍历：以月份条目为 tqdm 单位（不会卡死在某个月很长的 neg_sample）
    for _time_text, info in tqdm(time_mapping.items(), total=len(time_mapping), desc="scan time_mapping"):
        for neg in iter_neg_samples(info):
            compiler_raw = neg.get("compiler") or neg.get("author") or neg.get("compiler_name") or ""
            akey = map_author(compiler_raw)

            total_by_author[akey] += 1

            # 只认 match_strong == True 为“通过”
            ms = neg.get("match_strong", False)
            passed = bool(ms) is True

            if passed:
                pass_by_author[akey] += 1

            if akey == "OTHER":
                other_compiler_counter[normalize_name(compiler_raw)] += 1

    # 汇总打印
    def safe_acc(p, n):
        return (p / n) if n > 0 else 0.0

    logger.info("")
    logger.info("=== Per-author results (neg_sample) ===")

    keys_order = ["GDG", "WLW", "KYD", "DY", "LZQ", "OTHER"]
    for k in keys_order:
        n = total_by_author.get(k, 0)
        p = pass_by_author.get(k, 0)
        acc = safe_acc(p, n)
        logger.info(f"{AUTHOR_CANON.get(k, k):<6s} | total={n:<8d} pass={p:<8d} acc={acc:.4f}")

    # overall（只算这几类全部加一起，包括 OTHER）
    N_all = sum(total_by_author.values())
    P_all = sum(pass_by_author.values())
    logger.info("")
    logger.info(f"=== Overall === total={N_all} pass={P_all} acc={safe_acc(P_all, N_all):.4f}")

    # 按你给的那种输出格式也补一份（方便你直接贴论文/README）
    logger.info("")
    logger.info("=== Copy-friendly summary ===")
    logger.info(f"* 顾栋高：总 {total_by_author.get('GDG',0)}，通过 {pass_by_author.get('GDG',0)}；")
    logger.info(f"* 魏了翁：总 {total_by_author.get('WLW',0)}，通过 {pass_by_author.get('WLW',0)}；")
    logger.info(f"* 孔颖达：总 {total_by_author.get('KYD',0)}，通过 {pass_by_author.get('KYD',0)}；")
    logger.info(f"* 杜预：总 {total_by_author.get('DY',0)}，通过 {pass_by_author.get('DY',0)}；")
    logger.info(f"* 吕祖谦：总 {total_by_author.get('LZQ',0)}，通过 {pass_by_author.get('LZQ',0)}。")

    # OTHER debug
    if args.show_other:
        logger.info("")
        logger.info("=== OTHER compiler raw-name top (for mapping extension) ===")
        for name, cnt in other_compiler_counter.most_common(30):
            if not name:
                name = "<EMPTY>"
            logger.info(f"{cnt:6d}  {name}")

    # 保存 json
    if args.out_json:
        out = {
            "meta_path": args.meta_path,
            "total_by_author": {AUTHOR_CANON.get(k,k): int(v) for k, v in total_by_author.items()},
            "pass_by_author": {AUTHOR_CANON.get(k,k): int(v) for k, v in pass_by_author.items()},
            "acc_by_author":  {AUTHOR_CANON.get(k,k): safe_acc(pass_by_author.get(k,0), total_by_author.get(k,0))
                               for k in total_by_author.keys()},
            "overall": {
                "total": int(N_all),
                "pass": int(P_all),
                "acc": safe_acc(P_all, N_all),
            }
        }
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        logger.info(f"saved out_json => {args.out_json}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
