# chunqiu_dataset.py
# -*- coding: utf-8 -*-

import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from torch.utils.data import Dataset

POS_SOURCES = ["春秋", "春秋左氏傳", "春秋公羊傳", "春秋穀梁傳"]


# ---------- 工具函数 ----------

def load_splits(splits_path: str) -> Dict[Tuple[int, int, int], str]:
    """
    加载 time_splits_by_month_v1.json，返回:
        {(gong, year, month): "train"/"val"/"test"}
    """
    with open(splits_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data["split_by_sort_key"]
    out = {}
    for key_str, split in raw.items():
        g, y, m = map(int, key_str.split("-"))
        out[(g, y, m)] = split
    return out


def sort_key_to_str(sk: Tuple[int, int, int]) -> str:
    return f"{sk[0]}-{sk[1]}-{sk[2]}"


def normalize_sort_keys_and_time_ids(
    sort_keys: List[Tuple[int, int, int]],
    sort_key2_time_id: Dict[Tuple[int, int, int], int],
    fallback_sk: Optional[Tuple[int, int, int]] = None,
) -> Tuple[List[Tuple[int, int, int]], List[int]]:
    """
    将 sort_keys 过滤为 corpus 中存在的月份，并按 time_id 升序排序。
    返回 (sorted_sort_keys, sorted_time_ids)。

    - 如果过滤后为空且提供了 fallback_sk，则用 fallback_sk 补一个。
    - 保证返回的列表长度一致，且与排序后的 time_id 对齐。
    """
    valid_pairs: List[Tuple[int, Tuple[int, int, int]]] = []
    for sk in sort_keys:
        sk = tuple(sk)
        tid = sort_key2_time_id.get(sk, None)
        if tid is None:
            continue
        valid_pairs.append((tid, sk))

    if not valid_pairs and fallback_sk is not None:
        fb_tid = sort_key2_time_id.get(tuple(fallback_sk), None)
        if fb_tid is not None:
            valid_pairs = [(fb_tid, tuple(fallback_sk))]

    if not valid_pairs:
        return [], []

    valid_pairs.sort(key=lambda x: x[0])  # sort by time_id
    sorted_time_ids = [tid for tid, _ in valid_pairs]
    sorted_sort_keys = [sk for _, sk in valid_pairs]
    return sorted_sort_keys, sorted_time_ids


@dataclass
class CorpusIndex:
    sid2text: Dict[int, str]
    sid2_sort_key: Dict[int, Tuple[int, int, int]]
    sid2_type: Dict[int, str]     # "event", "no_event", "neg_comment"
    sid2_source: Dict[int, str]

    pos_sids_by_sort_key: Dict[Tuple[int, int, int], List[int]]
    neg_sids_by_sort_key: Dict[Tuple[int, int, int], List[int]]

    all_pos_sids_train: List[int]
    all_neg_sids_train: List[int]

    sort_key2_time_id: Dict[Tuple[int, int, int], int]
    time_id2_sort_key: Dict[int, Tuple[int, int, int]]
    num_time_bins: int


def build_corpus_index(
    meta_path: str,
    splits_by_sort_key: Dict[Tuple[int, int, int], str],
) -> CorpusIndex:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    time_mapping = meta["time_mapping"]

    sid2text: Dict[int, str] = {}
    sid2_sort_key: Dict[int, Tuple[int, int, int]] = {}
    sid2_type: Dict[int, str] = {}
    sid2_source: Dict[int, str] = {}

    pos_sids_by_sort_key: Dict[Tuple[int, int, int], List[int]] = {}
    neg_sids_by_sort_key: Dict[Tuple[int, int, int], List[int]] = {}

    all_pos_sids_train: List[int] = []
    all_neg_sids_train: List[int] = []

    seen_sort_keys: set[Tuple[int, int, int]] = set()

    for _, info in time_mapping.items():
        sk_list = info["sort_key"]
        sort_key = (int(sk_list[0]), int(sk_list[1]), int(sk_list[2]))
        seen_sort_keys.add(sort_key)

        split = splits_by_sort_key.get(sort_key, "train")
        no_event_flag = bool(info.get("no_event", False))

        pos_sids_here: List[int] = []
        neg_sids_here: List[int] = []

        # 1) 正样本：春秋 + 三传
        for ver in info.get("versions", []):
            for src in POS_SOURCES:
                if src not in ver:
                    continue
                entries = ver[src]
                if not isinstance(entries, list):
                    continue
                for e in entries:
                    sid = int(e["sid"])
                    text = e["text"]
                    sid2text[sid] = text
                    sid2_sort_key[sid] = sort_key
                    sid2_source[sid] = src
                    sid2_type[sid] = "no_event" if no_event_flag else "event"
                    pos_sids_here.append(sid)
                    if split == "train":
                        all_pos_sids_train.append(sid)

            # 2) version 级别注疏负样本
            for neg_block in ver.get("neg_sample", []):
                parsed = neg_block.get("parsed_comment")
                if not isinstance(parsed, list):
                    continue
                for e in parsed:
                    sid = int(e["sid"])
                    text = e["text"]
                    sid2text[sid] = text
                    sid2_sort_key[sid] = sort_key
                    sid2_source[sid] = neg_block.get("source", "comment")
                    sid2_type[sid] = "neg_comment"
                    neg_sids_here.append(sid)
                    if split == "train":
                        all_neg_sids_train.append(sid)

        # 3) time 级别 no_event 的 neg_samples
        for neg_block in info.get("neg_samples", []):
            parsed = neg_block.get("parsed_comment")
            if not isinstance(parsed, list):
                continue
            for e in parsed:
                sid = int(e["sid"])
                text = e["text"]
                sid2text[sid] = text
                sid2_sort_key[sid] = sort_key
                sid2_source[sid] = neg_block.get("source", "comment")
                sid2_type[sid] = "neg_comment"
                neg_sids_here.append(sid)
                if split == "train":
                    all_neg_sids_train.append(sid)

        if pos_sids_here:
            pos_sids_by_sort_key.setdefault(sort_key, []).extend(pos_sids_here)
        if neg_sids_here:
            neg_sids_by_sort_key.setdefault(sort_key, []).extend(neg_sids_here)

    # 全局时间轴（按 sort_key 排序）
    sort_key2_time_id: Dict[Tuple[int, int, int], int] = {}
    time_id2_sort_key: Dict[int, Tuple[int, int, int]] = {}
    for tid, sk in enumerate(sorted(seen_sort_keys)):
        sort_key2_time_id[sk] = tid
        time_id2_sort_key[tid] = sk
    num_time_bins = len(sort_key2_time_id)

    all_pos_sids_train = sorted(set(all_pos_sids_train))
    all_neg_sids_train = sorted(set(all_neg_sids_train))

    return CorpusIndex(
        sid2text=sid2text,
        sid2_sort_key=sid2_sort_key,
        sid2_type=sid2_type,
        sid2_source=sid2_source,
        pos_sids_by_sort_key=pos_sids_by_sort_key,
        neg_sids_by_sort_key=neg_sids_by_sort_key,
        all_pos_sids_train=all_pos_sids_train,
        all_neg_sids_train=all_neg_sids_train,
        sort_key2_time_id=sort_key2_time_id,
        time_id2_sort_key=time_id2_sort_key,
        num_time_bins=num_time_bins,
    )


def load_all_queries(queries_path: str) -> List[dict]:
    out = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def infer_query_split(
    q: dict,
    splits_by_sort_key: Dict[Tuple[int, int, int], str],
) -> Optional[str]:
    qtype = q.get("type") or q.get("source")
    if qtype == "point":
        sk = tuple(q["sort_key"])
        return splits_by_sort_key.get(sk, "train")

    tkeys = q.get("target_sort_keys") or []
    if not tkeys:
        return None
    splits = set()
    for sk_list in tkeys:
        sk = tuple(sk_list)
        s = splits_by_sort_key.get(sk)
        if s is None:
            continue
        splits.add(s)
    if not splits:
        return None
    if len(splits) == 1:
        return splits.pop()
    return "mixed"


def get_query_target_sort_keys(q: dict) -> List[Tuple[int, int, int]]:
    qtype = q.get("type")
    if qtype == "point":
        sk = q.get("sort_key")
        if sk is None:
            return []
        return [tuple(sk)]
    tkeys = q.get("target_sort_keys") or []
    return [tuple(sk) for sk in tkeys]


# ---------- Train Dataset ----------

class ChunqiuTrainDataset(Dataset):
    def __init__(
        self,
        meta_path: str,
        queries_path: str,
        splits_path: str,
        num_negatives: int = 16,
        seed: int = 42,
    ):
        super().__init__()
        self.num_negatives = num_negatives
        self.rng = random.Random(seed)

        self.splits_by_sort_key = load_splits(splits_path)
        self.corpus = build_corpus_index(meta_path, self.splits_by_sort_key)

        all_queries = load_all_queries(queries_path)
        train_queries: List[dict] = []
        for q in all_queries:
            qtype = q.get("type", "point")
            pos_sids = q.get("pos_sids") or []
            if not pos_sids:
                continue

            split = infer_query_split(q, self.splits_by_sort_key)
            if split != "train":
                continue

            pos_sids = [sid for sid in pos_sids if sid in self.corpus.sid2text]
            if not pos_sids:
                continue

            # ★ 利用 corpus.sid2_type 判断这个 query 是否是“纯 no_event”
            types = {self.corpus.sid2_type.get(sid, "event") for sid in pos_sids}
            is_pure_no_event = (len(types) == 1 and "no_event" in types)

            q = dict(q)
            q["pos_sids"] = pos_sids
            q["query_split"] = "train"
            q["type"] = qtype
            q["is_pure_no_event"] = is_pure_no_event   # ★ 新增
            train_queries.append(q)

        self.queries = train_queries
        print(f"[ChunqiuTrainDataset] loaded {len(self.queries)} train queries.")

    def __len__(self) -> int:
        return len(self.queries)

    def _sample_positive(self, q: dict) -> int:
        return self.rng.choice(q["pos_sids"])

    def _sample_negative_sids(self, q: dict, pos_sid: int) -> List[int]:
        target_sks = get_query_target_sort_keys(q)

        local_candidates: set[int] = set()
        for sk in target_sks:
            if self.splits_by_sort_key.get(sk, "train") != "train":
                continue
            local_candidates.update(self.corpus.neg_sids_by_sort_key.get(sk, []))

        if pos_sid in local_candidates:
            local_candidates.remove(pos_sid)

        local_candidates = list(local_candidates)
        self.rng.shuffle(local_candidates)

        neg_sids: List[int] = []
        for sid in local_candidates:
            if len(neg_sids) >= self.num_negatives:
                break
            neg_sids.append(sid)

        if len(neg_sids) < self.num_negatives:
            need = self.num_negatives - len(neg_sids)
            global_pool = [
                sid for sid in self.corpus.all_neg_sids_train
                if sid != pos_sid and sid not in neg_sids
            ]
            if len(global_pool) <= need:
                neg_sids.extend(global_pool)
            else:
                neg_sids.extend(self.rng.sample(global_pool, need))

        return neg_sids

    def __getitem__(self, idx: int) -> dict:
        q = self.queries[idx]
        qid = q.get("qid", idx)
        qtype = q.get("type", "point")
        query_text = q["query"]

        is_pure_no_event = q.get("is_pure_no_event", False)  # ★ 新增

        # ---- 正样本 ----
        pos_sid = self._sample_positive(q)
        pos_text = self.corpus.sid2text[pos_sid]

        pos_sk = self.corpus.sid2_sort_key[pos_sid]
        pos_gong, pos_year, pos_month = pos_sk

        pos_time_id = self.corpus.sort_key2_time_id[pos_sk]
        pos_start_time_id = pos_time_id
        pos_end_time_id = pos_time_id
        pos_start_sort_key = pos_sk
        pos_end_sort_key = pos_sk

        # ---- 负样本 ----
        neg_sids = self._sample_negative_sids(q, pos_sid)
        neg_texts = [self.corpus.sid2text[sid] for sid in neg_sids]

        # ---- Query 的 sort_key 范围（核心：start/end）----
        if qtype == "point":
            anchor_sk = tuple(q.get("sort_key"))
            raw_target_sks = [anchor_sk]
        else:
            anchor_sk = tuple(q.get("anchor_sort_key") or [-1, -1, -1])
            raw_target_sks = get_query_target_sort_keys(q)

        # 统一：用 time_id 排序后的 target_sks / target_time_ids
        target_sks_sorted, target_time_ids_sorted = normalize_sort_keys_and_time_ids(
            raw_target_sks,
            self.corpus.sort_key2_time_id,
            fallback_sk=pos_sk,   # 极端兜底：用正样本月份
        )

        # 保证 start <= end
        query_start_time_id = int(target_time_ids_sorted[0])
        query_end_time_id = int(target_time_ids_sorted[-1])
        query_start_sort_key = target_sks_sorted[0]
        query_end_sort_key = target_sks_sorted[-1]

        # 兼容你原先的“代表点”标签（仍保留，但 start/end 才是你要用的）
        if qtype == "point":
            query_sk_rep = query_start_sort_key
        else:
            mid = len(target_sks_sorted) // 2
            query_sk_rep = target_sks_sorted[mid]

        query_gong, query_year, query_month = query_sk_rep
        query_time_id = int(self.corpus.sort_key2_time_id[tuple(query_sk_rep)])

        return {
            "qid": qid,
            "query": query_text,
            "query_type": qtype,

            "anchor_sort_key": anchor_sk,
            "target_sort_keys": target_sks_sorted,              # 已按 time_id 排序
            "target_time_ids": target_time_ids_sorted,          # NEW：同样已排序（可选，但很实用）

            # ---- Query 范围（你要的）----
            "query_start_sort_key": query_start_sort_key,       # NEW
            "query_end_sort_key": query_end_sort_key,           # NEW
            "query_start_time_id": query_start_time_id,         # NEW
            "query_end_time_id": query_end_time_id,             # NEW

            # ---- 正样本（句子范围：start=end）----
            "pos_sid": pos_sid,
            "pos_text": pos_text,
            "pos_start_sort_key": pos_start_sort_key,           # NEW
            "pos_end_sort_key": pos_end_sort_key,               # NEW
            "pos_start_time_id": int(pos_start_time_id),         # NEW
            "pos_end_time_id": int(pos_end_time_id),             # NEW

            # ---- neg ----
            "neg_sids": neg_sids,
            "neg_texts": neg_texts,

            # ---- 时间标签（原样保留：仍是 1-based sort_key 三元组）----
            "query_gong_label": int(query_gong),
            "query_year_label": int(query_year),
            "query_month_label": int(query_month),

            "pos_gong_label": int(pos_gong),
            "pos_year_label": int(pos_year),
            "pos_month_label": int(pos_month),

            # ---- 代表点的 time_id（原样保留）----
            "query_time_id": int(query_time_id),
            "pos_time_id": int(pos_time_id),

            # ★ 新增：query 级别的 no_event 标记
            "is_pure_no_event": is_pure_no_event,
        }


# ---------- Eval Dataset ----------

class ChunqiuEvalDataset(Dataset):
    def __init__(
        self,
        meta_path: str,
        queries_path: str,
        splits_path: str,
        split: str = "val",
        include_no_event_queries: bool = True,
    ):
        assert split in ("train", "val", "test")
        self.split = split
        self.include_no_event_queries = include_no_event_queries

        self.splits_by_sort_key = load_splits(splits_path)
        self.corpus = build_corpus_index(meta_path, self.splits_by_sort_key)
        all_queries = load_all_queries(queries_path)

        eval_queries: List[dict] = []
        for q in all_queries:
            pos_sids = q.get("pos_sids") or []
            if not pos_sids:
                continue
            q_split = infer_query_split(q, self.splits_by_sort_key)
            if q_split != split:
                continue

            pos_sids = [sid for sid in pos_sids if sid in self.corpus.sid2text]
            if not pos_sids:
                continue

            types = {self.corpus.sid2_type.get(sid, "event") for sid in pos_sids}
            is_pure_no_event = (len(types) == 1 and "no_event" in types)

            if (not include_no_event_queries) and is_pure_no_event:
                continue

            q = dict(q)
            q["pos_sids"] = pos_sids
            q["query_split"] = split
            q["is_pure_no_event"] = is_pure_no_event
            eval_queries.append(q)

        self.queries = eval_queries
        print(
            f"[ChunqiuEvalDataset] loaded {len(self.queries)} {split} "
            f"queries (include_no_event_queries={include_no_event_queries})"
        )

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> dict:
        q = self.queries[idx]
        qid = q.get("qid", idx)
        query_text = q["query"]
        qtype = q.get("type", "point")

        gold_sids = q["pos_sids"]

        if qtype == "point":
            anchor_sk = tuple(q.get("sort_key"))
            raw_target_sks = [anchor_sk]
        else:
            anchor_sk = tuple(q.get("anchor_sort_key") or [-1, -1, -1])
            raw_target_sks = get_query_target_sort_keys(q)

        target_sks_sorted, target_time_ids_sorted = normalize_sort_keys_and_time_ids(
            raw_target_sks,
            self.corpus.sort_key2_time_id,
            fallback_sk=self.corpus.sid2_sort_key[gold_sids[0]],  # 兜底：用任意 gold 的月份
        )

        query_start_time_id = int(target_time_ids_sorted[0])
        query_end_time_id = int(target_time_ids_sorted[-1])
        query_start_sort_key = target_sks_sorted[0]
        query_end_sort_key = target_sks_sorted[-1]

        return {
            "qid": qid,
            "query": query_text,
            "query_type": qtype,

            "anchor_sort_key": anchor_sk,
            "target_sort_keys": target_sks_sorted,
            "target_time_ids": target_time_ids_sorted,          # NEW

            "query_start_sort_key": query_start_sort_key,       # NEW
            "query_end_sort_key": query_end_sort_key,           # NEW
            "query_start_time_id": query_start_time_id,         # NEW
            "query_end_time_id": query_end_time_id,             # NEW

            "gold_sids": gold_sids,
        }


def build_eval_gallery(
    corpus: CorpusIndex,
    include_neg_samples: bool = True,
    include_no_event_sids: bool = True,
):
    gallery_sids: List[int] = []
    for sid, t in corpus.sid2_type.items():
        if t == "event":
            gallery_sids.append(sid)
        elif t == "no_event" and include_no_event_sids:
            gallery_sids.append(sid)
        elif t == "neg_comment" and include_neg_samples:
            gallery_sids.append(sid)

    gallery_sids = sorted(gallery_sids)
    gallery_texts = [corpus.sid2text[sid] for sid in gallery_sids]
    sid2idx = {sid: i for i, sid in enumerate(gallery_sids)}
    return gallery_sids, gallery_texts, sid2idx
