# main.py
# -*- coding: utf-8 -*-

import os
import time
import json
from datetime import datetime
import sys
import shlex

# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29799 main.py --model_name_or_path /amax/wangyh/pretrained/Qwen3-Embedding-0.6B/  --batch_size 8   --num_epochs 3 --learning_rate 3e-6  --max_doc_len 256  --eval_include_neg_samples  --eval_include_no_event_sids   --max_query_len 128  --use_global_inbatch --output_dir ckpts_qwen3_0.6B_baseline_global_sup_ctx  --use_multipos_sup  --use_time_context_pred --use_neg_train
# Eval mode flags for mode_name = f"neg{int(include_neg_samples)}_ne{int(include_no_event_sids)}_dq{int(drop_no_event_q)}"
#
# neg0 / neg1 : whether to include exegetical negative-comment sentences in the gallery
#   neg0 -> include_neg_samples = False  : gallery 只包含 event（以及可选的 no_event，占位句），不含注疏里的 neg_comment
#   neg1 -> include_neg_samples = True   : gallery 额外加入注疏 neg_comment 句子，形成更难、更接近真实语料库的设置
#
# ne0 / ne1 : whether to include no_event sentences in the gallery
#   ne0 -> include_no_event_sids = False : gallery 中不包含 no_event（空月占位句），只检索真实事件 / 注疏
#   ne1 -> include_no_event_sids = True  : gallery 中包含 no_event 句子，用来模拟时间轴上“空月”的干扰
#
# dq0 / dq1 : whether to drop queries whose gold sentences are all no_event
#   dq0 -> drop_no_event_q = False : 保留所有 query，包括那些“金标准都是 no_event”的纯空月 query
#   dq1 -> drop_no_event_q = True  : 丢弃纯 no_event query，只在评测中保留至少含一个 event 的 query


import torch
import torch.distributed as dist                           # ★ NEW(DDP)
from torch.nn.parallel import DistributedDataParallel as DDP  # ★ NEW(DDP)
from transformers import get_linear_schedule_with_warmup

from src.train_temporal_dual import (
    parse_args,
    set_seed,
    build_dataloaders_and_corpus,
    train_one_epoch,
    evaluate_on_val,
    evaluate_on_test_best_ckpt,
)

from src.models_temporal_dual import BertDualEncoder, QwenTimeDualEncoder


def is_dist_avail_and_initialized():                       # ★ NEW(DDP helper)
    return dist.is_available() and dist.is_initialized()


def main():
    args = parse_args()
    set_seed(args.seed)

    # ====== 1. DDP 初始化部分 ======
    # torchrun --nproc_per_node=2 main.py ... 时，这些 env 会自动注入
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        distributed = True
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )
    else:
        distributed = False
        local_rank = 0

    args.distributed = distributed                         # ★ NEW
    args.local_rank = local_rank                           # ★ NEW
    args.world_size = dist.get_world_size() if distributed else 1  # ★ NEW
    args.rank = dist.get_rank() if distributed else 0      # ★ NEW

    is_main_process = (args.rank == 0)                     # ★ NEW

    # ====== 2. 设备选择 ======
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if is_main_process:
        print("Using device:", device)
        print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print(f"[INFO] distributed = {distributed}, rank = {args.rank}, world_size = {args.world_size}")

    # ====== 3. 数据 & corpus ======
    # 注意：build_dataloaders_and_corpus 里稍后要加 DistributedSampler，
    # 但 main 这边不需要改接口
    data_bundle = build_dataloaders_and_corpus(args)
    train_loader = data_bundle["train_loader"]
    tokenizer = data_bundle["tokenizer"]
    val_queries = data_bundle["val_queries"]
    val_gold_sids_list = data_bundle["val_gold_sids_list"]
    val_gallery_sids = data_bundle["val_gallery_sids"]
    corpus = data_bundle["corpus"]

    num_gong = data_bundle["num_gong"]
    num_year = data_bundle["num_year"]
    num_month = data_bundle["num_month"]
    num_time_bins = data_bundle["num_time_bins"]

    # ====== 4. 构建模型：BERT vs Qwen ======
    # 先识别是否 Qwen（这段逻辑你原来就有，只是我搬到 DDP 初始化之后）
    is_qwen = (
        "Qwen3-Embedding" in args.model_name_or_path
        or ("Qwen3" in args.model_name_or_path and "Embedding" in args.model_name_or_path)
        or ("Qwen" in args.model_name_or_path and "Embedding" in args.model_name_or_path)
    )
    args.is_qwen = is_qwen

    if args.is_qwen:
        args.pooling = "last_token"
        if not hasattr(args, "task_description") or args.task_description is None:
            args.task_description = (
                "Given a classical Chinese query about the Spring and Autumn Annals, "
                "retrieve relevant passages that describe the corresponding historical events."
            )
    else:
        if not hasattr(args, "pooling") or args.pooling is None:
            args.pooling = "cls"

    if args.is_qwen:
        base_model = QwenTimeDualEncoder(
            model_name_or_path=args.model_name_or_path,
            normalize=True,
            num_gong=num_gong,
            num_year=num_year,
            num_month=num_month,
            use_time_context_pred=args.use_time_context_pred,
            time_emb_dim=args.time_emb_dim,
            use_time_rel_bias=args.use_time_rel_bias,
            time_rel_dim=args.time_rel_dim,
        ).to(device)
    else:
        base_model = BertDualEncoder(
            model_name_or_path=args.model_name_or_path,
            pooling=args.pooling,
            normalize=True,
            num_gong=num_gong,
            num_year=num_year,
            num_month=num_month,
            use_time_context_pred=args.use_time_context_pred,
            time_emb_dim=args.time_emb_dim,
            use_time_rel_bias=args.use_time_rel_bias,
            time_rel_dim=args.time_rel_dim,
        ).to(device)

    # ★ NEW：DDP 包装
    if distributed:
        model = DDP(
            base_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,   # 如果后面你确定有些 head 可能没被用到，可以改 True
        )
    else:
        model = base_model

    # 小工具：在需要“真身”时用（比如保存 ckpt）
    def unwrap_model(m):
        return m.module if isinstance(m, DDP) else m

    # ====== 5. Optimizer & Scheduler ======
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
    )

    num_update_steps_per_epoch = len(train_loader)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(args.warmup_ratio * max_train_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    if is_main_process:
        print(f"[INFO] #steps/epoch = {num_update_steps_per_epoch}, total steps = {max_train_steps}")
        print(f"[INFO] warmup_steps = {num_warmup_steps}")

    # ====== 6. 日志文件（只在 rank 0 建） ======
    start_time = time.time()
    start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(args.output_dir, f"train_log_{log_time_str}.txt")

    raw_argv = " ".join(shlex.quote(x) for x in sys.argv)
    # 2) 一些关键的 env（基本等价于你 shell 里的前缀）
    launcher_env = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "WORLD_SIZE":           os.environ.get("WORLD_SIZE", ""),
        "RANK":                 os.environ.get("RANK", ""),
        "LOCAL_RANK":           os.environ.get("LOCAL_RANK", ""),
        "MASTER_ADDR":          os.environ.get("MASTER_ADDR", ""),
        "MASTER_PORT":          os.environ.get("MASTER_PORT", ""),
    }

    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f_log:
            f_log.write("=" * 80 + "\n")
            f_log.write("Training Log\n")
            # ★ 新增：记录“命令行 + 启动环境”
            f_log.write("- CMD ------------------------------\n")
            f_log.write(f"argv (as seen by Python):\n{raw_argv}\n")
            f_log.write("launcher env (subset):\n")
            for k, v in launcher_env.items():
                f_log.write(f"  {k}={v}\n")

            f_log.write("=" * 80 + "\n")
            f_log.write(f"Start time : {start_dt}\n")
            f_log.write(f"Model      : {args.model_name_or_path}\n")
            f_log.write(f"is_qwen    : {args.is_qwen}\n")
            f_log.write(f"Pooling    : {args.pooling}\n")
            f_log.write(f"Output dir : {args.output_dir}\n")
            f_log.write(f"Loss type  : {args.loss_type}\n")
            f_log.write(f"Margin     : {args.margin}\n")
            f_log.write(f"Temperature: {args.temperature}\n")
            f_log.write(f"Time loss w: {args.time_loss_weight}\n")
            f_log.write(
                f"Time label smoothing: {args.time_label_smoothing}, "
                f"eps={args.time_label_smoothing_eps}\n"
            )
            f_log.write(
                f"Time align KL: {args.use_time_align}, "
                f"w={args.time_align_weight}\n"
            )
            f_log.write(
                f"Time regression: {args.use_time_regression}, "
                f"w={args.time_reg_weight}\n"
            )
            f_log.write(f"Time context: {args.use_time_context}, dim={args.time_emb_dim}\n")
            f_log.write(f"Batch size : {args.batch_size}\n")
            f_log.write(f"Num epochs : {args.num_epochs}\n")
            f_log.write(f"LR         : {args.learning_rate}\n")
            f_log.write(f"Max q len  : {args.max_query_len}\n")
            f_log.write(f"Max d len  : {args.max_doc_len}\n")
            f_log.write(f"Logging    : every {args.logging_steps} steps\n")
            f_log.write("- ARGS raw ------------------------------\n")
            f_log.write(json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n")
            f_log.write("=" * 80 + "\n\n")

    # ====== 7. 训练 + 验证 ======
    global_step = 0
    best_metric = -1.0
    best_epoch = -1
    best_step = -1

    # ----- step-level eval 回调：只在 rank 0 真正跑 eval -----
    def step_eval_callback(model_for_train, global_step, epoch):
        if args.eval_steps <= 0:
            return

        if not is_main_process:
            return  # 非主进程直接跳过

        model_for_eval = unwrap_model(model_for_train)
        model_for_eval.eval()
        with torch.no_grad():
            metrics = evaluate_on_val(
                model=model_for_eval,
                tokenizer=tokenizer,
                val_queries=val_queries,
                val_gold_sids_list=val_gold_sids_list,
                val_gallery_sids=val_gallery_sids,
                corpus=corpus,
                device=device,
                args=args,
            )
        model_for_eval.train()

        nonlocal best_metric, best_epoch, best_step

        r1 = metrics["Recall@1"]
        r5 = metrics["Recall@5"]
        r10 = metrics["Recall@10"]
        mrr10 = metrics["MRR@10"]

        msg = (
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"[STEP-EVAL] global_step={global_step}, "
            f"R@1={r1:.4f}, R@5={r5:.4f}, R@10={r10:.4f}, MRR@10={mrr10:.4f} "
            f"(#queries={len(val_queries)})"
        )
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f_log:
            f_log.write(msg + "\n")

        if r1 > best_metric:
            best_metric = r1
            best_epoch = epoch
            best_step = global_step

            best_dir = os.path.join(args.output_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            real_model = unwrap_model(model_for_train)

            print(
                f"[CKPT-STEP] New best at epoch {epoch}, step {global_step}, "
                f"saving to {best_dir} (R@1={best_metric:.4f})"
            )
            real_model.encoder.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            torch.save(
                {
                    "state_dict": real_model.state_dict(),
                    "config": {
                        "dual_type": "qwen" if args.is_qwen else "bert",
                        "model_name_or_path": args.model_name_or_path,
                        "pooling": args.pooling,
                        "normalize": True,
                        "num_gong": num_gong,
                        "num_year": num_year,
                        "num_month": num_month,
                        "use_time_context_pred": args.use_time_context_pred,
                        "time_emb_dim": args.time_emb_dim,
                        "use_time_rel_bias": args.use_time_rel_bias,
                        "time_rel_dim": args.time_rel_dim,
                    },
                },
                os.path.join(best_dir, "dual_model.pt"),
            )

            with open(log_path, "a", encoding="utf-8") as f_log:
                f_log.write(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"[CKPT-STEP] New best epoch={epoch}, step={global_step}, "
                    f"R@1={best_metric:.4f}\n"
                )

    # --------------------------------------------
    for epoch in range(1, args.num_epochs + 1):
        # DDP: 每个 epoch 重置 sampler 随机种子
        if args.distributed:
            # build_dataloaders_and_corpus 里记得用 DistributedSampler 存在 train_loader.sampler
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

        global_step, avg_loss_epoch = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            global_step=global_step,
            args=args,
            log_path=log_path if is_main_process else None,  # 非主进程可在内部少写文件
            num_update_steps_per_epoch=num_update_steps_per_epoch,
            start_time=start_time,
            eval_steps=args.eval_steps,
            step_eval_callback=step_eval_callback,
        )

        if is_main_process:
            print(f"[TRAIN] Epoch {epoch} finished, avg_loss={avg_loss_epoch:.4f}")

        # ===== 每个 epoch 末尾做一次 full eval（只在主进程） =====
        if is_main_process:
            model_for_eval = unwrap_model(model)
            val_metrics = evaluate_on_val(
                model=model_for_eval,
                tokenizer=tokenizer,
                val_queries=val_queries,
                val_gold_sids_list=val_gold_sids_list,
                val_gallery_sids=val_gallery_sids,
                corpus=corpus,
                device=device,
                args=args,
            )
            r1 = val_metrics["Recall@1"]
            r5 = val_metrics["Recall@5"]
            r10 = val_metrics["Recall@10"]
            mrr10 = val_metrics["MRR@10"]

            main_metric = r1
            if main_metric > best_metric:
                best_metric = main_metric
                best_epoch = epoch
                best_step = global_step

                best_dir = os.path.join(args.output_dir, "best")
                os.makedirs(best_dir, exist_ok=True)
                real_model = unwrap_model(model)

                print(
                    f"[CKPT] New best at epoch {epoch}, saving to {best_dir} "
                    f"(R@1={best_metric:.4f})"
                )
                real_model.encoder.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                torch.save(
                    {
                        "state_dict": real_model.state_dict(),
                        "config": {
                            "dual_type": "qwen" if args.is_qwen else "bert",
                            "model_name_or_path": args.model_name_or_path,
                            "pooling": args.pooling,
                            "normalize": True,
                            "num_gong": num_gong,
                            "num_year": num_year,
                            "num_month": num_month,
                            "use_time_context_pred": args.use_time_context_pred,
                            "time_emb_dim": args.time_emb_dim,
                            "use_time_rel_bias": args.use_time_rel_bias,
                            "time_rel_dim": args.time_rel_dim,
                        },
                    },
                    os.path.join(best_dir, "dual_model.pt"),
                )
                with open(log_path, "a", encoding="utf-8") as f_log:
                    f_log.write(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"[CKPT] New best epoch={epoch}, step={global_step}, "
                        f"R@1={best_metric:.4f}\n"
                    )

            with open(log_path, "a", encoding="utf-8") as f_log:
                f_log.write(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"Epoch {epoch}, avg_loss={avg_loss_epoch:.4f}, "
                    f"R@1={r1:.4f}, R@5={r5:.4f}, "
                    f"R@10={r10:.4f}, MRR@10={mrr10:.4f}\n"
                )

        # 让所有进程在 epoch 末尾对齐一下（不是必须，但比较整洁）
        if args.distributed:
            dist.barrier()

    # ===== 8. 训练结束 =====
    if is_main_process:
        end_time = time.time()
        end_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed_min = (end_time - start_time) / 60.0
        print(
            f"[DONE] Training finished. start={start_dt}, end={end_dt}, "
            f"elapsed={elapsed_min:.2f} min"
        )
        print(f"[DONE] Best epoch = {best_epoch}, best R@1 = {best_metric:.4f}")

        if best_epoch > 0:
            best_dir = os.path.join(args.output_dir, "best")
            evaluate_on_test_best_ckpt(
                best_dir=best_dir,
                tokenizer=tokenizer,
                args=args,
                log_path=log_path,
                device=device,
            )
        else:
            print("[TEST] No best_epoch recorded (best_epoch <= 0), skip test eval.")

        print("[ALL DONE] Training + best-checkpoint TEST eval finished.")

    # ===== 9. 关闭 DDP =====
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
