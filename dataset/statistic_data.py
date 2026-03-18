# stats_chunqiu_splits.py
# -*- coding: utf-8 -*-

import json
import os
import sys
from collections import defaultdict

# ==== 0. 配置 Python 路径，导入 Dataset 工具函数 ====
# 假设本脚本放在项目根目录下，ChunQiuDataset 在 ./src/ChunQiuDataset.py
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "../src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from ChunQiuDataset import (
    load_splits,
    build_corpus_index,
    load_all_queries,
    infer_query_split,
)

# ==== 1. 配置路径 ====

META_PATH = "chunqiu_meta_sid_fixed.json"
QUERIES_PATH = "queries_all_labeledv3.jsonl"
SPLITS_PATH = "time_splits_by_month_v1.json"

# ==== 2. 加载 splits & meta，统计 month 数 ====

# (gong, year, month) -> "train"/"val"/"test"
splits_by_sort_key = load_splits(SPLITS_PATH)

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

time_mapping = meta["time_mapping"]

# 只统计在 meta 里实际出现过的月份
months_by_split = {"train": set(), "val": set(), "test": set()}

for _time_text, info in time_mapping.items():
    g, y, m = info["sort_key"]
    sk = (int(g), int(y), int(m))
    split = splits_by_sort_key.get(sk, "train")
    if split in months_by_split:
        months_by_split[split].add(sk)

# ==== 3. 构建语料索引，统计 sentence-level 单元 ====

corpus = build_corpus_index(META_PATH, splits_by_sort_key)

sentences_by_split = {"train": 0, "val": 0, "test": 0}
sentences_total = 0

for sid, sk in corpus.sid2_sort_key.items():
    split = splits_by_sort_key.get(sk, "train")
    if split in sentences_by_split:
        sentences_by_split[split] += 1
        sentences_total += 1

# 进一步拆分：event / no_event / neg_comment
sentences_by_split_and_type = {
    "train": defaultdict(int),
    "val": defaultdict(int),
    "test": defaultdict(int),
}

for sid, sk in corpus.sid2_sort_key.items():
    split = splits_by_sort_key.get(sk, "train")
    if split not in sentences_by_split_and_type:
        continue
    t = corpus.sid2_type.get(sid, "unknown")
    sentences_by_split_and_type[split][t] += 1

# ==== 4. 统计每个 split 的 queries 和 (query, sentence) pairs ====

all_queries = load_all_queries(QUERIES_PATH)

queries_by_split = {"train": 0, "val": 0, "test": 0}
pairs_by_split = {"train": 0, "val": 0, "test": 0}  # sum of |pos_sids|

for q in all_queries:
    pos_sids = q.get("pos_sids") or []
    if not pos_sids:
        continue  # 没有 gold 的 query 直接跳过

    # 过滤掉 meta 里已经不存在的 sid（保险）
    pos_sids = [sid for sid in pos_sids if sid in corpus.sid2text]
    if not pos_sids:
        continue

    q_split = infer_query_split(q, splits_by_sort_key)
    # 丢掉 mixed / None，只保留干净落在 train/val/test 的查询
    if q_split not in ("train", "val", "test"):
        continue

    queries_by_split[q_split] += 1
    pairs_by_split[q_split] += len(pos_sids)

# ==== 5. 打印结果（直接对照 LaTeX 表） ====

print("=== Final benchmark statistics ===")
print(
    "Split      | #months | #sentences | #queries | Avg. gold sents/query "
    "| #event sents | #no-event sents | #neg. comments"
)
print(
    "-----------+---------+------------+----------+------------------------"
    "+--------------+-----------------+----------------"
)

total_months = 0
total_queries = 0
total_pairs = 0

total_event = 0
total_no_event = 0
total_neg = 0

for split in ["train", "val", "test"]:
    n_months = len(months_by_split[split])
    n_sents = sentences_by_split[split]
    n_queries = queries_by_split[split]
    n_pairs = pairs_by_split[split]

    avg_gold = (n_pairs / n_queries) if n_queries > 0 else 0.0

    dist = sentences_by_split_and_type[split]
    n_event = dist.get("event", 0)
    n_no_event = dist.get("no_event", 0)
    n_neg = dist.get("neg_comment", 0)

    total_months += n_months
    total_queries += n_queries
    total_pairs += n_pairs

    total_event += n_event
    total_no_event += n_no_event
    total_neg += n_neg

    print(
        f"{split:9s} | "
        f"{n_months:7d} | "
        f"{n_sents:10d} | "
        f"{n_queries:8d} | "
        f"{avg_gold:22.1f} | "
        f"{n_event:12d} | "
        f"{n_no_event:15d} | "
        f"{n_neg:14d}"
    )

print(
    "-----------+---------+------------+----------+------------------------"
    "+--------------+-----------------+----------------"
)

avg_gold_total = (total_pairs / total_queries) if total_queries > 0 else 0.0

print(
    f"{'Total':9s} | "
    f"{total_months:7d} | "
    f"{sentences_total:10d} | "
    f"{total_queries:8d} | "
    f"{avg_gold_total:22.1f} | "
    f"{total_event:12d} | "
    f"{total_no_event:15d} | "
    f"{total_neg:14d}"
)

# （可选）如果仍然想 debug 具体 type 分布，可以保留这一段简单输出：
print("\n=== Sentence type breakdown by split (event / no_event / neg_comment / other) ===")
for split in ["train", "val", "test"]:
    dist = sentences_by_split_and_type[split]
    # 带上 unknown 方便检查有没有漏标的
    for t in ["event", "no_event", "neg_comment", "unknown"]:
        if dist.get(t, 0) == 0:
            continue
        print(f"{split:9s} - {t:11s}: {dist[t]}")
