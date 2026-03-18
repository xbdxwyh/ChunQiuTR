import os
import json

def pretty_print_summary(summary_metrics, log_path=None):
    if not summary_metrics:
        print("[EVAL] No metrics collected, skip summary table.")
        return

    header = [
        "mode", "family",
        "Val_R1", "Val_R5", "Val_R10", "Val_MRR", "Val_nDCG",
        "Test_R1", "Test_R5", "Test_R10", "Test_MRR", "Test_nDCG",
    ]

    rows = []
    all_mode_names = sorted({k[0] for k in summary_metrics.keys()})
    family_order = ["all", "point", "window"]

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, (float, int)) else str(v)

    for mode_name in all_mode_names:
        for family in family_order:
            has_any = ((mode_name, family, "val") in summary_metrics) or ((mode_name, family, "test") in summary_metrics)
            if not has_any:
                continue

            if (mode_name, family, "val") in summary_metrics:
                mv = summary_metrics[(mode_name, family, "val")]
                v_r1, v_r5, v_r10, v_mrr, v_ndcg = fmt(mv["Recall@1"]), fmt(mv["Recall@5"]), fmt(mv["Recall@10"]), fmt(mv["MRR@10"]), fmt(mv["nDCG@10"])
            else:
                v_r1 = v_r5 = v_r10 = v_mrr = v_ndcg = "N/A"

            if (mode_name, family, "test") in summary_metrics:
                mt = summary_metrics[(mode_name, family, "test")]
                t_r1, t_r5, t_r10, t_mrr, t_ndcg = fmt(mt["Recall@1"]), fmt(mt["Recall@5"]), fmt(mt["Recall@10"]), fmt(mt["MRR@10"]), fmt(mt["nDCG@10"])
            else:
                t_r1 = t_r5 = t_r10 = t_mrr = t_ndcg = "N/A"

            rows.append([mode_name, family, v_r1, v_r5, v_r10, v_mrr, v_ndcg, t_r1, t_r5, t_r10, t_mrr, t_ndcg])
    col_widths = [len(h) for h in header]
    for row in rows:
        for j, cell in enumerate(row):
            col_widths[j] = max(col_widths[j], len(str(cell)))

    def fmt_row(cells):
        return " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(cells))

    sep_line = "-+-".join("-" * w for w in col_widths)

    print("\n[SUMMARY] Eval combinations (Val/Test, all/point/window):")
    print(fmt_row(header))
    print(sep_line)
    for row in rows:
        print(fmt_row(row))

    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n[SUMMARY] Eval combinations (Val/Test, all/point/window):\n")
            f.write(fmt_row(header) + "\n")
            f.write(sep_line + "\n")
            for row in rows:
                f.write(fmt_row(row) + "\n")


def export_ranked_results_jsonl(
    output_path,
    ranked_ids_list,
    ranked_scores_list,
    gold_ids_list,
    raw_queries,
    q_types=None,
    is_pure_no_event=None,
    qids=None,
    gallery_mode=None,
    top_k=20,
):
    """
    通用导出函数：把检索结果按 JSONL 存下来，方便后续可视化 / case 分析。

    参数约定（长度都相同 = #queries）：
      - ranked_ids_list:   List[List[str]]，每个 query 的候选 doc_id 排序列表
      - ranked_scores_list:List[List[float]]，同一位置的相似度分数
      - gold_ids_list:     List[List[str]]，每个 query 的 gold doc_id 列表
      - raw_queries:       List[str]，原始 query 文本
      - q_types:           可选，List[str]，比如 "point"/"window"
      - is_pure_no_event:  可选，List[bool]
      - qids:              可选，List[任意，可 JSON 化]，比如 "point-18"
      - gallery_mode:      可选，字符串，记录当前 eval 协议，比如 "neg0_ne1_dq1"
      - top_k:             导出前截断到前 top_k 个候选
    """
    n = len(raw_queries)
    assert len(ranked_ids_list) == n
    assert len(ranked_scores_list) == n
    assert len(gold_ids_list) == n

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(n):
            entry = {
                "qid": qids[i] if qids is not None else i,
                "query": raw_queries[i],
                "type": q_types[i] if q_types is not None else None,
                "is_pure_no_event": bool(is_pure_no_event[i]) if is_pure_no_event is not None else None,
                "gold_sids": gold_ids_list[i],
                "topk_sids": ranked_ids_list[i][:top_k],
                "topk_scores": ranked_scores_list[i][:top_k],
                "gallery_mode": gallery_mode,
            }
            # 把值是 None 的字段去掉，防止 JSON 里一堆 null
            entry = {k: v for k, v in entry.items() if v is not None}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
