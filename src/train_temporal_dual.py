from .ChunQiuDataset import (
    ChunqiuTrainDataset,
    ChunqiuEvalDataset,
    load_splits,
    build_corpus_index,
    load_all_queries,
    infer_query_split,
    build_eval_gallery,   # 如果 test eval 里要用 gallery 构建
)

from .models_temporal_dual import (
    BertDualEncoder,
    RetrievalCollator,
    wrap_query_with_instruction,  # ★ 必须加这个
)
from .retrieval_utils import (
    contrastive_loss_inbatch,
    triplet_loss,
    encode_texts_bert,
    simple_collate,
    compute_retrieval_metrics,
    compute_retrieval_per_query,
    contrastive_loss_global_inbatch,
    point_singlepos_loss_inbatch,
    point_singlepos_loss_global_inbatch
)
import random
import torch
import argparse
import os
import time
from datetime import datetime
from .time_losses import compute_time_losses
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.distributed as dist 
from torch.utils.data import DataLoader

# ============ 训练脚本 main ============


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a simple BERT dual-encoder on Chunqiu temporal retrieval (triplet / contrastive + time aux)"
    )

    # ===== 数据路径（★ 新增这三行）=====
    parser.add_argument(
        "--meta_path",
        type=str,
        default="dataset/chunqiu_meta_sid_fixed.json",
        help="包含 time_mapping 和句子级 sid 的 meta JSON",
    )
    parser.add_argument(
        "--queries_path",
        type=str,
        default="dataset/queries_all_labeledv3.jsonl",
        help="合并后的查询文件（point+window）的 jsonl",
    )
    parser.add_argument(
        "--splits_path",
        type=str,
        default="dataset/time_splits_by_month_v1.json",
        help="按 (gong, year, month) 的时间切分配置",
    )

    # 模型相关
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/amax/wangyh/pretrained/bert-base-chinese",
        help="预训练 BERT 路径或 HuggingFace 名字",
    )
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean"])
    parser.add_argument("--margin", type=float, default=0.2)

    # 训练超参
    parser.add_argument("--output_dir", type=str, default="ckpts_bert_dual")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_query_len", type=int, default=64)
    parser.add_argument("--max_doc_len", type=int, default=128)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_epochs", type=int, default=1)

    # 数据路径
    parser.add_argument("--eval_include_neg_samples", action="store_true", default=False)
    parser.add_argument("--eval_include_no_event_sids", action="store_true", default=False)
    parser.add_argument("--eval_drop_no_event_queries", action="store_true", default=False)

    parser.add_argument(
        "--val_gallery_mode",
        type=str,
        default="all",
        choices=["all", "events_only"],
        help="Validation 时用的 gallery 类型"
    )

    parser.add_argument(
        "--loss_type",
        type=str,
        default="contrastive",  # "triplet" or "contrastive"
        choices=["triplet", "contrastive"],
        help="Use classic triplet loss or in-batch contrastive (InfoNCE) loss.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help="Temperature for contrastive loss."
    )

    # === NEW: 时间辅助任务相关 ===

    # 总权重（乘在 time_loss_total 上）
    parser.add_argument(
        "--time_loss_weight",
        type=float,
        default=0,
        help="Weight for auxiliary time prediction loss (0 to disable).",
    )

    # A. 邻近 label smoothing（gong/year/month 分类）
    parser.add_argument(
        "--time_label_smoothing",
        action="store_true",
        help="Use neighbor-aware label smoothing for gong/year/month classification."
    )
    parser.add_argument(
        "--time_label_smoothing_eps",
        type=float,
        default=0.2,
        help="Epsilon for neighbor-aware label smoothing."
    )

    # B. query-doc 时间分布对齐 KL
    parser.add_argument(
        "--use_time_align",
        action="store_true",
        help="Align query/doc time distributions via symmetric KL."
    )
    parser.add_argument(
        "--time_align_weight",
        type=float,
        default=1.0,
        help="Weight of time alignment KL in time_loss_total."
    )

    # C. 连续时间回归
    parser.add_argument(
        "--use_time_regression",
        action="store_true",
        help="Enable continuous time regression head."
    )
    parser.add_argument(
        "--time_reg_weight",
        type=float,
        default=1.0,
        help="Weight of time regression loss in time_loss_total."
    )

    parser.add_argument(
        "--use_time_context",
        action="store_true",
        help="Whether to enable time-context fusion into embeddings.",
    )
    parser.add_argument(
        "--time_emb_dim",
        type=int,
        default=64,
        help="Dimensionality of time embedding when use_time_context is enabled.",
    )
    parser.add_argument(
        "--use_time_context_pred",
        action="store_true",
        help="Whether to enable time-context prediction fusion into embeddings.",
    )

    # D. 基于 Δt 的 relative-time Fourier bias（可选）
    parser.add_argument(
        "--use_time_rel_bias",
        action="store_true",
        help="Enable relative-time Fourier bias on retrieval logits.",
    )
    parser.add_argument(
        "--time_rel_dim",
        type=int,
        default=32,
        help="Dimensionality of relative-time Fourier features.",
    )

    parser.add_argument(
        "--time_kernel_type",
        type=str,
        default="fourier_mlp",
        choices=["fourier_mlp", "linear", "poly", "gaussian"],
        help="Type of relative time kernel for bias on logits.",
    )

    parser.add_argument(
        "--eval_steps",
        type=int,
        default=0,
        help="If > 0, run a validation on val split every N global steps during training. "
            "0 means only validate at the end of each epoch."
    )

    parser.add_argument(
        "--max_eval_queries_per_step",
        type=int,
        default=256,
        help="When doing step-level validation, only use the first K val queries for speed."
    )

    parser.add_argument(
        "--use_multipos_sup",
        action="store_true",
        help="Enable supervised multi-positive InfoNCE using time ranges.",
    )

    parser.add_argument(
        "--use_global_inbatch",
        action="store_true",
        help="Use DDP all_gather to build global in-batch negatives for contrastive loss."
    )

    parser.add_argument(
        "--use_neg_train",
        action="store_true"
    )

    # 控制是否启用Points的时间辅助任务
    parser.add_argument(
        "--point_loss_weight",
        type=float,
        default=0,
        help="Weight for point contrastive loss (0 to disable).",
    )

    # 硬开关：点损失
    parser.add_argument(
        "--use_point_loss",
        action="store_true",
        help="Use point contrastive loss during training.",
    )

    # 设备
    parser.add_argument("--device", type=str, default="cuda", help="'cuda' 或 'cpu'")

    # 随机种子
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return args

# 其他 import（ChunqiuTrainDataset / ChunqiuEvalDataset / load_splits / build_corpus_index / wrap_query_with_instruction）
# 保持你原来的就行
def build_dataloaders_and_corpus(args):
    """
    负责：
      - 加载 train / val dataset
      - 构建 train_loader（支持单机多卡 DDP）
      - 构建全局 corpus & val gallery
      - 统计时间类别数 (gong/year/month)

    返回一个 dict，里面打包好后面要用到的一切。
    """
    is_main = (getattr(args, "rank", 0) == 0)

    # 1) Train Dataset
    train_ds = ChunqiuTrainDataset(
        meta_path=args.meta_path,
        queries_path=args.queries_path,
        splits_path=args.splits_path,
        num_negatives=16,  # collator 里再随机采 1 个
        seed=args.seed,
    )
    if is_main:
        print(f"[INFO] Loaded train dataset, size = {len(train_ds)}")

    # 2) 统计时间类别数
    all_sort_keys = list(train_ds.corpus.sid2_sort_key.values())
    max_gong = max(sk[0] for sk in all_sort_keys)   # 1..G
    max_year = max(sk[1] for sk in all_sort_keys)   # 1..Y
    max_month = max(sk[2] for sk in all_sort_keys)  # 1..M

    num_gong = max_gong
    num_year = max_year
    num_month = max_month
    if is_main:
        print(f"[INFO] Time classes: gong={num_gong}, year={num_year}, month={num_month}")

    # === 2.5) 判断是否是 Qwen3-Embedding ===
    # 优先使用 main 里已经写入的 args.is_qwen；如果没有再自己判断一次
    if hasattr(args, "is_qwen"):
        is_qwen = args.is_qwen
    else:
        model_name = args.model_name_or_path
        is_qwen = (
            ("Qwen3-Embedding" in model_name)
            or ("Qwen3" in model_name)
            or ("Qwen" in model_name)
        )
        args.is_qwen = is_qwen

    # 3) Tokenizer & Collator & Train DataLoader
    if is_qwen:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
        )
        tokenizer.padding_side = "left"

        if not hasattr(args, "task_description") or args.task_description is None:
            args.task_description = (
                "Given a classical Chinese query about the Spring and Autumn Annals, "
                "retrieve relevant passages that describe the corresponding historical events."
            )
        task_description = args.task_description

        collator = RetrievalCollator(
            tokenizer=tokenizer,
            max_query_len=args.max_query_len,
            max_doc_len=args.max_doc_len,
            use_instruction=True,
            task_description=task_description,
        )
        if is_main:
            print("[INFO] Using Qwen-style tokenizer (left padding) and Instruct-wrapped queries.")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        collator = RetrievalCollator(
            tokenizer=tokenizer,
            max_query_len=args.max_query_len,
            max_doc_len=args.max_doc_len,
            use_instruction=False,   # 对 BERT 不包装
            task_description="",
        )

    # ★ DDP: Sampler
    if getattr(args, "distributed", False):
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
            seed=args.seed,
        )
        shuffle_flag = False  # 有 sampler 就不要再 shuffle 了
    else:
        train_sampler = None
        shuffle_flag = True

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        sampler=train_sampler,
        collate_fn=collator,
        drop_last=True,
    )

    if is_main:
        print("[INFO] Building validation dataset & corpus index ...")

    # 4) Val Dataset（只需要 query 文本 & gold sids）
    val_ds = ChunqiuEvalDataset(
        meta_path=args.meta_path,
        queries_path=args.queries_path,
        splits_path=args.splits_path,
        split="val",
    )

    raw_val_queries = []
    val_gold_sids_list = []
    for item in val_ds:
        raw_val_queries.append(item["query"])
        val_gold_sids_list.append(item["gold_sids"])

    # ★ 如果是 Qwen，验证集的 query 也要用同样的 Instruct 包装
    if is_qwen:
        task_description = getattr(
            args,
            "task_description",
            "Given a classical Chinese query about the Spring and Autumn Annals, "
            "retrieve relevant passages that describe the corresponding historical events.",
        )
        val_queries = [
            wrap_query_with_instruction(task_description, q)
            for q in raw_val_queries
        ]
    else:
        val_queries = raw_val_queries

    # 5) 全局 corpus + gallery（base gallery 只按 val_gallery_mode 筛一次）
    splits_by_sort_key = load_splits(args.splits_path)
    corpus = build_corpus_index(args.meta_path, splits_by_sort_key)

    if getattr(args, "val_gallery_mode", None) is None:
        args.val_gallery_mode = "all"

    if args.val_gallery_mode == "all":
        # 所有句子都进 gallery（包括 event / no_event / neg_comment）
        val_gallery_sids = sorted(corpus.sid2text.keys())
    elif args.val_gallery_mode == "events_only":
        # 只保留 event + no_event
        val_gallery_sids = sorted(
            sid for sid, t in corpus.sid2_type.items() if t in ("event", "no_event")
        )
    else:
        raise ValueError(f"Unknown val_gallery_mode: {args.val_gallery_mode}")

    val_gallery_texts = [corpus.sid2text[sid] for sid in val_gallery_sids]
    val_sid2idx = {sid: i for i, sid in enumerate(val_gallery_sids)}

    # 6) gold_sids -> gold_indices（仅当前 base gallery 下）
    val_gold_indices_list = []
    for sids in val_gold_sids_list:
        indices = [val_sid2idx[sid] for sid in sids if sid in val_sid2idx]
        val_gold_indices_list.append(indices)

    if is_main:
        print(f"[VAL] #queries={len(val_queries)}, gallery_size={len(val_gallery_sids)}")

    num_time_bins = train_ds.corpus.num_time_bins

    return {
        "train_ds": train_ds,
        "train_loader": train_loader,
        "tokenizer": tokenizer,
        "val_ds": val_ds,
        "val_queries": val_queries,              # Qwen 已包装
        "val_gold_sids_list": val_gold_sids_list,
        "val_gold_indices_list": val_gold_indices_list,
        "val_gallery_sids": val_gallery_sids,
        "val_gallery_texts": val_gallery_texts,
        "val_sid2idx": val_sid2idx,
        "corpus": corpus,
        "num_gong": num_gong,
        "num_year": num_year,
        "num_month": num_month,
        "num_time_bins": num_time_bins,
    }


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    device,
    epoch,
    global_step,
    args,
    log_path,
    num_update_steps_per_epoch,
    start_time,
    eval_steps=0,
    step_eval_callback=None,
):
    is_main = (getattr(args, "rank", 0) == 0)

    # 这几个 flag 的含义和老版本一致
    use_global_inbatch     = getattr(args, "use_global_inbatch", False)
    use_multipos_sup       = getattr(args, "use_multipos_sup", False)
    use_time_rel_bias_flag = getattr(args, "use_time_rel_bias", False)
    use_neg_train          = getattr(args, "use_neg_train", False)   # ★ 新增：是否在 loss 中使用 neg_emb

    model.train()
    epoch_loss = 0.0
    step_in_epoch = 0

    for step_in_epoch, batch in enumerate(train_loader, start=1):
        global_step += 1

        batch_t = {
            k: v.to(device)
            for k, v in batch.items()
            if isinstance(v, torch.Tensor)
        }

        # ---------- 1) 前向 ----------
        # 现在的模型 forward 已经把 time_rel_bias 一并算好并返回
        outputs = model(batch_t)
        (
            q_emb, pos_emb, neg_emb,
            q_base, pos_base,
            gong_logits_q, year_logits_q, month_logits_q,
            gong_logits_p, year_logits_p, month_logits_p,
            time_rel_bias,
        ) = outputs

        # if len(outputs) == 5:
        #     # 没开 use_time_rel_bias：不返回 bias
        #     q_emb, pos_emb, neg_emb, q_base, pos_base = outputs
        #     time_rel_bias = None
        # elif len(outputs) == 6:
        #     # 开了 use_time_rel_bias：多一个 bias
        #     q_emb, pos_emb, neg_emb, q_base, pos_base, time_rel_bias = outputs
        # else:
        #     raise ValueError(
        #         f"Unexpected number of outputs from model.forward: {len(outputs)}"
        #     )

        # 实际给 loss 用的 bias / neg
        time_bias_for_loss = time_rel_bias if use_time_rel_bias_flag else None
        neg_for_loss = neg_emb if use_neg_train else None

        # ---------- 2) 检索主损失 ----------
        if getattr(args, "loss_type", "triplet") == "triplet":
            # Triplet 模式本身就依赖 neg_emb，不受 use_neg_train 控制
            retrieval_loss = triplet_loss(
                q_emb, pos_emb, neg_emb, margin=args.margin
            )
        else:
            # 对比学习模式：InfoNCE / supervised contrastive

            if use_global_inbatch:
                # ★ 保持老版本语义：
                #   只要 use_global_inbatch=True，就统一走全局 in-batch 版本
                #   multipos/time_bias/neg_emb 全部通过参数控制
                retrieval_loss = contrastive_loss_global_inbatch(
                    q_emb=q_emb,
                    p_emb=pos_emb,
                    neg_emb=neg_for_loss,                      # 可选额外负样本
                    temperature=args.temperature,
                    time_bias=time_bias_for_loss,              # 可选时间 bias
                    # multipos 只在有监督时才传时间标签
                    query_start_time_id=batch_t.get("query_start_time_id", None)
                        if use_multipos_sup else None,
                    query_end_time_id=batch_t.get("query_end_time_id", None)
                        if use_multipos_sup else None,
                    pos_start_time_id=batch_t.get("pos_start_time_id", None)
                        if use_multipos_sup else None,
                    pos_end_time_id=batch_t.get("pos_end_time_id", None)
                        if use_multipos_sup else None,
                    symmetric=True,
                )
            else:
                # ★ use_global_inbatch=False：退回本地 in-batch 版本
                if use_multipos_sup:
                    # 有监督 multipos
                    retrieval_loss = contrastive_loss_inbatch(
                        q_emb=q_emb,
                        p_emb=pos_emb,
                        neg_emb=neg_for_loss,
                        temperature=args.temperature,
                        time_bias=time_bias_for_loss,
                        query_start_time_id=batch_t.get("query_start_time_id", None),
                        query_end_time_id=batch_t.get("query_end_time_id", None),
                        pos_start_time_id=batch_t.get("pos_start_time_id", None),
                        pos_end_time_id=batch_t.get("pos_end_time_id", None),
                        symmetric=True,
                    )
                else:
                    # 无监督（单对角 InfoNCE）
                    retrieval_loss = contrastive_loss_inbatch(
                        q_emb=q_emb,
                        p_emb=pos_emb,
                        neg_emb=neg_for_loss,
                        temperature=args.temperature,
                        time_bias=time_bias_for_loss,
                        # 不传时间标签 => has_supervision=False => 退回无监督 InfoNCE
                    )

        # ---------- 3) 时间 loss（保持和原来一致） ----------
        time_loss_total = torch.tensor(0.0, device=device)
        time_loss_doc_ce = torch.tensor(0.0, device=device)
        time_loss_align = torch.tensor(0.0, device=device)
        time_loss_reg = torch.tensor(0.0, device=device)
        point_loss = torch.tensor(0.0, device=device)

        # 只有同时满足这三点才启用：
        #   1) 显式打开 --use_point_loss
        #   2) point_loss_weight > 0
        #   3) 有监督时间标签（use_multipos_sup=True）
        if getattr(args, "use_point_loss", False) and args.point_loss_weight > 0.0 and use_multipos_sup:
            q_st = batch_t.get("query_start_time_id", None)
            q_ed = batch_t.get("query_end_time_id", None)
            p_st = batch_t.get("pos_start_time_id", None)
            p_ed = batch_t.get("pos_end_time_id", None)

            if q_st is not None and q_ed is not None and p_st is not None and p_ed is not None:
                if use_global_inbatch:
                    # DDP 版：利用全局 gallery
                    point_loss = point_singlepos_loss_global_inbatch(
                        q_emb=q_emb,
                        p_emb=pos_emb,
                        query_start_time_id=q_st,
                        query_end_time_id=q_ed,
                        pos_start_time_id=p_st,
                        pos_end_time_id=p_ed,
                        temperature=args.temperature,
                        symmetric=False,  # 先用单向更稳定
                        query_is_pure_no_event=batch_t.get("query_is_pure_no_event", None),  # ← 这里
                    )
                else:
                    # 单卡版
                    point_loss = point_singlepos_loss_inbatch(
                        q_emb=q_emb,
                        p_emb=pos_emb,
                        query_start_time_id=q_st,
                        query_end_time_id=q_ed,
                        pos_start_time_id=p_st,
                        pos_end_time_id=p_ed,
                        temperature=args.temperature,
                        symmetric=False,
                        query_is_pure_no_event=batch_t.get("query_is_pure_no_event", None),  # ← 这里
                    )
        
        base_model = getattr(model, "module", model)
        if args.time_loss_weight > 0.0 and getattr(base_model, "use_time_heads", False):
            (
                time_loss_total,
                time_loss_doc_ce,
                time_loss_align,
                time_loss_reg,
            ) = compute_time_losses(
                model=base_model,
                q_emb=q_base,
                pos_emb=pos_base,
                gong_logits_q=gong_logits_q,
                year_logits_q=year_logits_q,
                month_logits_q=month_logits_q,
                gong_logits_p=gong_logits_p,
                year_logits_p=year_logits_p,
                month_logits_p=month_logits_p,
                batch_t=batch_t,
                use_neighbor_smoothing=args.time_label_smoothing,
                smoothing_eps=args.time_label_smoothing_eps,
                use_time_align=args.use_time_align,
                use_time_regression=args.use_time_regression,
                time_align_weight=args.time_align_weight,
                time_reg_weight=args.time_reg_weight,
            )

        loss = retrieval_loss + args.time_loss_weight * time_loss_total + args.point_loss_weight * point_loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()

        # ---------- 4) 日志（只在主进程写） ----------
        if global_step % args.logging_steps == 0 and is_main and log_path is not None:
            avg_loss = epoch_loss / step_in_epoch
            elapsed = time.time() - start_time
            msg = (
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"[Epoch {epoch}/{args.num_epochs}] "
                f"step {step_in_epoch}/{num_update_steps_per_epoch}, "
                f"global_step={global_step}, "
                f"loss={loss.item():.4f}, avg_loss={avg_loss:.4f}, "
                f"retrieval={retrieval_loss.item():.4f}, "
                f"point_loss={point_loss.item():.4f}, "   # ★ 新增
                f"time_total={time_loss_total.item():.4f}, "
                f"time_doc_ce={time_loss_doc_ce.item():.4f}, "
                f"time_align={time_loss_align.item():.4f}, "
                f"time_reg={time_loss_reg.item():.4f}, "
                f"elapsed={elapsed/60:.2f} min"
            )
            print(msg)
            with open(log_path, "a", encoding="utf-8") as f_log:
                f_log.write(msg + "\n")

        # ---------- 5) step-level eval ----------
        if eval_steps > 0 and (global_step % eval_steps == 0):
            if step_eval_callback is not None:
                step_eval_callback(model, global_step, epoch)

    avg_loss_epoch = epoch_loss / max(1, step_in_epoch)
    return global_step, avg_loss_epoch


def unwrap_model(m):
    # 统一包装：无论是 DDP 还是 DataParallel，都能取到真身
    return m.module if isinstance(m, (DDP, torch.nn.DataParallel)) else m

def evaluate_on_val(
    model,
    tokenizer,
    val_queries,
    val_gold_sids_list,
    val_gallery_sids,
    corpus,
    device,
    args,
):
    """
    在 val split 上评估：

      - 根据 args.eval_include_neg_samples / eval_include_no_event_sids
        从 val_gallery_sids 中筛选有效 gallery；
      - 根据 args.eval_drop_no_event_queries 丢弃纯 no_event query；
      - 同时要保证每个保留的 query 在 gallery 中至少有 1 个 gold，
        否则也丢弃防止干扰指标。
    """
    is_main = (getattr(args, "rank", 0) == 0)
    distributed = getattr(args, "distributed", False)

    # 分布式时防御性处理：如果误在非主进程调用，就直接返回全 0
    if distributed and (not is_main):
        # 不打印、不算，直接交回一个 dummy 指标，避免多卡重复算 gallery
        return {
            "Recall@1": 0.0,
            "Recall@5": 0.0,
            "Recall@10": 0.0,
            "MRR@10": 0.0,
            "nDCG@10": 0.0,
        }

    if is_main:
        print("[VAL] Evaluating on val split with eval_* flags ...")

    # 先拿到 base_model 再取 encoder（兼容 DDP / 单卡）
    base_model = unwrap_model(model)
    encoder = base_model.encoder

    # 1) 构建有效 gallery（在 base val_gallery_sids 上再按类型过滤）
    eff_gallery_sids = []
    for sid in val_gallery_sids:
        t = corpus.sid2_type.get(sid, "event")
        if (t == "neg_comment") and (not args.eval_include_neg_samples):
            continue
        if (t == "no_event") and (not args.eval_include_no_event_sids):
            continue
        eff_gallery_sids.append(sid)

    eff_gallery_texts = [corpus.sid2text[sid] for sid in eff_gallery_sids]
    sid2effidx = {sid: i for i, sid in enumerate(eff_gallery_sids)}

    if is_main:
        print(
            f"[VAL] Base gallery size={len(val_gallery_sids)}, "
            f"effective gallery size={len(eff_gallery_sids)}, "
            f"include_neg={args.eval_include_neg_samples}, "
            f"include_no_event_sids={args.eval_include_no_event_sids}"
        )

    # 2) 按需要过滤 query & gold（纯 no_event / 在 gallery 里没有 gold 的都丢掉）
    eff_queries = []
    eff_gold_indices_list = []

    n_total_q = len(val_queries)
    n_drop_pure_no_event = 0
    n_drop_empty_gold = 0

    for q_text, sids in zip(val_queries, val_gold_sids_list):
        types = {corpus.sid2_type.get(sid, "event") for sid in sids}
        is_pure_no_event = (len(types) == 1 and "no_event" in types)

        if args.eval_drop_no_event_queries and is_pure_no_event:
            n_drop_pure_no_event += 1
            continue

        indices = [sid2effidx[sid] for sid in sids if sid in sid2effidx]
        if not indices:
            n_drop_empty_gold += 1
            continue

        eff_queries.append(q_text)
        eff_gold_indices_list.append(indices)

    if is_main:
        print(
            f"[VAL] #queries={n_total_q}, "
            f"kept={len(eff_queries)}, "
            f"dropped_pure_no_event={n_drop_pure_no_event}, "
            f"dropped_empty_gold={n_drop_empty_gold}"
        )

    if len(eff_queries) == 0:
        if is_main:
            print("[VAL] WARNING: no valid queries left after filtering, return zeros.")
        return {
            "Recall@1": 0.0,
            "Recall@5": 0.0,
            "Recall@10": 0.0,
            "MRR@10": 0.0,
            "nDCG@10": 0.0,
        }

    # 3) 编码 gallery
    val_gallery_embs = encode_texts_bert(
        eff_gallery_texts,
        tokenizer,
        encoder,
        device=device,
        max_length=getattr(args, "max_doc_len", 128),
        batch_size=64,
        pool_mode=args.pooling,
    )

    # 4) 编码 queries
    val_query_embs = encode_texts_bert(
        eff_queries,
        tokenizer,
        encoder,
        device=device,
        max_length=getattr(args, "max_query_len", 64),
        batch_size=32,
        pool_mode=args.pooling,
    )

    # 5) 计算指标
    val_metrics = compute_retrieval_metrics(
        query_embs=val_query_embs,
        gallery_embs=val_gallery_embs,
        gold_indices_list=eff_gold_indices_list,
        ks=(1, 5, 10),
    )

    r1 = val_metrics["Recall@1"]
    r5 = val_metrics["Recall@5"]
    r10 = val_metrics["Recall@10"]
    mrr10 = val_metrics["MRR@10"]
    ndcg10 = val_metrics["nDCG@10"]

    if is_main:
        val_msg = (
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"[VAL] R@1={r1:.4f}, R@5={r5:.4f}, R@10={r10:.4f}, MRR@10={mrr10:.4f}, nDCG@10={ndcg10:.4f}"
        )
        print(val_msg)

    return val_metrics

def evaluate_on_test_best_ckpt(
    best_dir,
    tokenizer,
    args,
    log_path,
    device,
    encoder=None,      # 可选：外面传好的 encoder（比如还想用 time 头玩的版本）
    run_label="",      # 可选：区分不同 run
):
    """
    使用 best checkpoint，在以下组合上评估：

      split ∈ {val, test}
      eval_include_neg_samples ∈ {False, True}
      eval_include_no_event_sids ∈ {False, True}
      eval_drop_no_event_queries ∈ {False, True}
    """
    is_main = (getattr(args, "rank", 0) == 0)
    distributed = getattr(args, "distributed", False)

    # 分布式时，只在主进程跑 eval，其他进程直接返回
    if distributed and (not is_main):
        return None

    # --- eval filters (optional) ---
    eval_splits = ("val", "test")
    only_split = getattr(args, "eval_only_split", None)
    if only_split in ("val", "test"):
        eval_splits = (only_split,)
    only_mode = getattr(args, "eval_only_mode", None)

    if not os.path.isdir(best_dir):
        if is_main:
            print(f"[TEST] WARNING: best_dir={best_dir} not found, skip test eval.")
        return None

    if is_main:
        print("\n[TEST] Evaluating BEST checkpoint on all eval_* combinations ...")
        print(f"[TEST] Loading best encoder from {best_dir} ...")

    # 0) 准备一个 encoder：
    #    - 如果外部没传 encoder，就从 best_dir 里加载 AutoModel
    #    - 如果外部传了 encoder（比如 DualEncoder.encoder），就直接用它
    if encoder is None:
        encoder = AutoModel.from_pretrained(
            best_dir,
            trust_remote_code=True,
        ).to(device)
    else:
        if is_main:
            print("[TEST] Using provided encoder object for evaluation (encoder argument).")
        encoder = encoder.to(device)

    encoder.eval()

    # 1) 构建语料索引
    splits_by_sort_key = load_splits(args.splits_path)
    corpus = build_corpus_index(args.meta_path, splits_by_sort_key)

    # 2) FULL gallery
    full_gallery_sids = sorted(corpus.sid2text.keys())
    full_gallery_texts = [corpus.sid2text[sid] for sid in full_gallery_sids]
    sid2fullidx = {sid: i for i, sid in enumerate(full_gallery_sids)}

    if is_main:
        print(f"[TEST] FULL gallery size = {len(full_gallery_sids)}")

    full_gallery_embs = encode_texts_bert(
        full_gallery_texts,
        tokenizer,
        encoder,
        device=device,
        max_length=args.max_doc_len,
        batch_size=64,
        pool_mode=args.pooling,
    )
    if is_main:
        print(
            f"[TEST] Encoded FULL gallery: shape={tuple(full_gallery_embs.shape)} "
            f"(doc_bs=64, max_len={args.max_doc_len})"
        )

    # 3) 预先编码 val / test FULL queries（下面保持你原来的逻辑）
    eval_sets = {}
    for split in eval_splits:
    # for split in ("val", "test"):
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

        if is_main:
            print(f"[TEST] Loaded {len(raw_queries)} FULL {split} queries.")

        if getattr(args, "is_qwen", False):
            task_description = getattr(
                args,
                "task_description",
                "Given a classical Chinese query about the Spring and Autumn Annals, "
                "retrieve relevant passages that describe the corresponding historical events.",
            )
            encode_texts = [
                wrap_query_with_instruction(task_description, q)
                for q in raw_queries
            ]
        else:
            encode_texts = raw_queries

        query_embs_full = encode_texts_bert(
            encode_texts,
            tokenizer,
            encoder,
            device=device,
            max_length=args.max_query_len,
            batch_size=args.batch_size,
            pool_mode=args.pooling,
        )

        eval_sets[split] = {
            "raw_queries": raw_queries,
            "gold_sids_list": gold_sids_list,
            "is_pure_no_event": is_pure_no_event_flags,
            "q_types": q_types,
            "query_embs_full": query_embs_full,
        }

    # 4) eval_* 组合 & family summary
    # （这部分我基本保持你原来的，只把所有 print / 写 log 包一层 is_main）

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

        mode_name = (
            f"neg{int(include_neg)}_"
            f"ne{int(include_no_event_sids)}_"
            f"dq{int(drop_no_event_q)}"
        )

        if only_mode and (mode_name != only_mode):
            continue

        if is_main:
            print(
                f"\n[TEST] === Combo: {mode_name} "
                f"(include_neg={include_neg}, "
                f"include_no_event_sids={include_no_event_sids}, "
                f"drop_no_event_q={drop_no_event_q}) ==="
            )
            print(f"[TEST] Effective gallery size = {len(eff_gallery_sids)}")

        # for split in ("val", "test"):
        for split in eval_splits:
            raw_queries = eval_sets[split]["raw_queries"]
            gold_sids_list = eval_sets[split]["gold_sids_list"]
            is_pure_flags = eval_sets[split]["is_pure_no_event"]
            q_types = eval_sets[split]["q_types"]
            query_embs_full = eval_sets[split]["query_embs_full"]

            valid_entries = []
            for i, (sids, is_pure, q_type) in enumerate(
                zip(gold_sids_list, is_pure_flags, q_types)
            ):
                if drop_no_event_q and is_pure:
                    continue

                indices = [sid2effidx[sid] for sid in sids if sid in sid2effidx]
                if not indices:
                    continue

                valid_entries.append(
                    {
                        "orig_idx": i,
                        "gold_indices": indices,
                        "q_type": q_type,
                    }
                )

            if not valid_entries:
                if is_main:
                    print(
                        f"[TEST] WARNING: split={split}, combo={mode_name} has 0 valid queries, "
                        f"skip metrics."
                    )
                continue

            # all family
            base_indices_all = [e["orig_idx"] for e in valid_entries]
            gold_indices_all = [e["gold_indices"] for e in valid_entries]
            query_embs_all = query_embs_full[base_indices_all]

            metrics_all = compute_retrieval_metrics(
                query_embs=query_embs_all,
                gallery_embs=eff_gallery_embs,
                gold_indices_list=gold_indices_all,
                ks=(1, 5, 10),
            )
            r1 = metrics_all["Recall@1"]
            r5 = metrics_all["Recall@5"]
            r10 = metrics_all["Recall@10"]
            mrr10 = metrics_all["MRR@10"]
            ndcg10 = metrics_all["nDCG@10"]
            
            ###### --- per-query dump (for significance tests / analysis) ---
            if getattr(args, "dump_per_query", False):
                perq = compute_retrieval_per_query(
                    query_embs=query_embs_all,
                    gallery_embs=eff_gallery_embs,
                    gold_indices_list=gold_indices_all,
                    ks=(1, 5, 10),
                    topk=10,
                )
                save_obj = {
                    "split": split,
                    # "mode_name": mode_name,
                    # "run_name": run_name,
                    "orig_query_indices": base_indices_all,   # 对齐用：原始 query 下标
                    "hit1": perq["hit"][1].cpu(),
                    "hit5": perq["hit"][5].cpu(),
                    "hit10": perq["hit"][10].cpu(),
                    "rr10": perq["rr"].cpu(),
                    "first_rank": perq["first_rank"].cpu(),
                }
                out_dir = os.path.join(args.output_dir, "per_query")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{run_label}_{split}_{mode_name}.pt")
                torch.save(save_obj, out_path)

                if is_main:
                    with open(log_path, "a", encoding="utf-8") as f_log:
                        f_log.write(f"[DUMP] per-query saved to: {out_path}\n")

            run_name = f"bert_best_{split}_{mode_name}"
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if is_main:
                print("=" * 80)
                print(f"Run name       : {run_name}")
                print(f"Time           : {now}")
                print(f"Model          : {best_dir}")
                print(f"Device         : {device}")
                print(f"Split          : {split}")
                print(f"Gallery size   : {len(eff_gallery_sids)}")
                print(f"#Eval queries  : {len(base_indices_all)}")
                print("- Metrics ------------------------------")
                print(f"Recall@1    : {r1:.6f}")
                print(f"Recall@5    : {r5:.6f}")
                print(f"Recall@10   : {r10:.6f}")
                print(f"MRR@10      : {mrr10:.6f}")
                print(f"nDCG@10     : {ndcg10:.6f}")
                print("- Config -------------------------------")
                print(f"max_query_len           = {args.max_query_len}")
                print(f"max_doc_len             = {args.max_doc_len}")
                print(f"query_bs                = {args.batch_size}")
                print(f"doc_bs                  = 64")
                print(f"include_neg_samples     = {include_neg}")
                print(f"include_no_event_sids   = {include_no_event_sids}")
                print(f"drop_no_event_queries   = {drop_no_event_q}")
                print("=" * 80)

                with open(log_path, "a", encoding="utf-8") as f_log:
                    f_log.write(
                        f"[{now}] [EVAL] run={run_name}, "
                        f"R@1={r1:.6f}, R@5={r5:.6f}, "
                        f"R@10={r10:.6f}, MRR@10={mrr10:.6f}, nDCG@10={ndcg10:.6f}, "
                        f"gallery_size={len(eff_gallery_sids)}, "
                        f"#queries={len(base_indices_all)}, "
                        f"include_neg={include_neg}, "
                        f"include_no_event_sids={include_no_event_sids}, "
                        f"drop_no_event_q={drop_no_event_q}\n"
                    )

            summary_metrics[(mode_name, "all", split)] = metrics_all

            # point / window family
            for family in ("point", "window"):
                if family == "point":
                    sub_entries = [e for e in valid_entries if e["q_type"] == "point"]
                else:
                    sub_entries = [e for e in valid_entries if e["q_type"] != "point"]

                if not sub_entries:
                    continue

                base_indices_sub = [e["orig_idx"] for e in sub_entries]
                gold_indices_sub = [e["gold_indices"] for e in sub_entries]
                query_embs_sub = query_embs_full[base_indices_sub]

                metrics_sub = compute_retrieval_metrics(
                    query_embs=query_embs_sub,
                    gallery_embs=eff_gallery_embs,
                    gold_indices_list=gold_indices_sub,
                    ks=(1, 5, 10),
                )

                summary_metrics[(mode_name, family, split)] = metrics_sub

    # 5) 打印 summary 表（只在主进程）
    if not summary_metrics:
        if is_main:
            print("[TEST] No metrics collected, skip summary table.")
        return summary_metrics

    header = [
        "mode",
        "family",
        "Val_R1", "Val_R5", "Val_R10", "Val_MRR", "Val_nDCG",
        "Test_R1", "Test_R5", "Test_R10", "Test_MRR", "Test_nDCG",
    ]

    rows = []
    all_mode_names = sorted({key[0] for key in summary_metrics.keys()})
    family_order = ["all", "point", "window"]

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, (float, int)) else str(v)

    for mode_name in all_mode_names:
        for family in family_order:
            has_any = (
                (mode_name, family, "val") in summary_metrics
                or (mode_name, family, "test") in summary_metrics
            )
            if not has_any:
                continue

            if (mode_name, family, "val") in summary_metrics:
                m_val = summary_metrics[(mode_name, family, "val")]
                v_r1 = fmt(m_val["Recall@1"])
                v_r5 = fmt(m_val["Recall@5"])
                v_r10 = fmt(m_val["Recall@10"])
                v_mrr = fmt(m_val["MRR@10"])
                v_ndcg = fmt(m_val["nDCG@10"])
            else:
                v_r1 = v_r5 = v_r10 = v_mrr = v_ndcg = "N/A"

            if (mode_name, family, "test") in summary_metrics:
                m_test = summary_metrics[(mode_name, family, "test")]
                t_r1 = fmt(m_test["Recall@1"])
                t_r5 = fmt(m_test["Recall@5"])
                t_r10 = fmt(m_test["Recall@10"])
                t_mrr = fmt(m_test["MRR@10"])
                t_ndcg = fmt(m_test["nDCG@10"])
            else:
                t_r1 = t_r5 = t_r10 = t_mrr = t_ndcg = "N/A"

            rows.append([
                mode_name,
                family,
                v_r1, v_r5, v_r10, v_mrr, v_ndcg,
                t_r1, t_r5, t_r10, t_mrr, t_ndcg,
            ])

    if not rows:
        if is_main:
            print("[TEST] No rows to summarize, skip summary table.")
        return summary_metrics

    col_widths = [len(h) for h in header]
    for row in rows:
        for j, cell in enumerate(row):
            col_widths[j] = max(col_widths[j], len(str(cell)))

    def fmt_row(cells):
        return " | ".join(
            str(c).ljust(col_widths[i]) for i, c in enumerate(cells)
        )

    sep_line = "-+-".join("-" * w for w in col_widths)

    if is_main:
        print("\n[TEST] ==== Summary over all eval_* combinations (Val vs Test, all/point/window) ====")
        print(fmt_row(header))
        print(sep_line)
        for row in rows:
            print(fmt_row(row))

        with open(log_path, "a", encoding="utf-8") as f_log:
            if run_label:
                f_log.write("\n" + "=" * 80 + "\n")
                f_log.write(f"[RUN] {run_label}\n")
                f_log.write("=" * 80 + "\n")
            f_log.write("\n[SUMMARY] Eval combinations (Val/Test, all/point/window):\n")
            f_log.write(fmt_row(header) + "\n")
            f_log.write(sep_line + "\n")
            for row in rows:
                f_log.write(fmt_row(row) + "\n")

    return summary_metrics
