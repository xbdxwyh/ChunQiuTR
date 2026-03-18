# evaluate.py
# -*- coding: utf-8 -*-

import os
import argparse
from datetime import datetime

import torch
from transformers import AutoConfig, AutoTokenizer

from src.train_temporal_dual import (
    set_seed,
    evaluate_on_test_best_ckpt,
)
from src.models_temporal_dual import BertDualEncoder, QwenTimeDualEncoder

# python evaluate.py   --ckpt_dir ../pretrained/Qwen3-Embedding-0.6B/ --output_dir model_outputs_results/outputs_qwen3 --max_query_len 128 --max_doc_len 256
# python evaluate.py   --ckpt_dir ../pretrained/Qwen3-Embedding-4B/ --output_dir model_outputs_results/outputs_qwen3 --max_query_len 128 --max_doc_len 256


def parse_eval_args():
    """
    专门用于评估的参数解析：
      - 数据路径：和 train_temporal_dual 里保持一致默认值
      - 评估相关：ckpt_dir, max_len, batch_size, device, seed 等
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained dual-encoder checkpoint on Chunqiu retrieval benchmark."
    )

    # ===== 数据路径（和 train_temporal_dual 默认保持一致）=====
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

    # ===== 评估相关超参 =====
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="要评估的 checkpoint 目录（训练时保存的 best/，里面有 encoder、tokenizer 以及 dual_model.pt）。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="评估日志输出目录；若不指定，则默认使用 ckpt_dir 的上一级目录。",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="编码 query 时的 batch size。",
    )
    parser.add_argument(
        "--max_query_len",
        type=int,
        default=64,
        help="编码 query 的最大长度。",
    )
    parser.add_argument(
        "--max_doc_len",
        type=int,
        default=196,
        help="编码 passage 的最大长度。",
    )

    # 设备 & 随机种子
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="'cuda' 或 'cpu'",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子。",
    )

    # 可选：给 Qwen 用的 instruction 文本（和训练保持一致）
    parser.add_argument(
        "--task_description",
        type=str,
        default=(
            "Given a classical Chinese query about the Spring and Autumn Annals, "
            "retrieve relevant passages that describe the corresponding historical events."
        ),
        help="对 Qwen 类模型的指令包装文本。",
    )

    # pooling 在 encoder-only 跑法中使用；DualEncoder 跑法会忽略该参数
    parser.add_argument(
        "--pooling",
        type=str,
        default=None,
        choices=["cls", "mean", "last_token"],
        help="评估时对 HF AutoModel 使用的池化方式；DualEncoder 跑法会忽略该参数。",
    )

    args = parser.parse_args()
    return args


def build_tokenizer_and_flags(args):
    """
    从 ckpt_dir 中读取 config，自动判断是不是 Qwen，
    并据此构造 tokenizer / pooling / is_qwen 标志。
    """
    try:
        config = AutoConfig.from_pretrained(args.ckpt_dir, trust_remote_code=True)
        model_type = getattr(config, "model_type", "")
    except Exception:
        # 兜底：config 读不到就从路径名猜一下
        model_type = os.path.basename(os.path.abspath(args.ckpt_dir))

    is_qwen = "qwen" in model_type.lower()
    args.is_qwen = is_qwen

    if is_qwen:
        if args.pooling is None:
            args.pooling = "last_token"
        tokenizer = AutoTokenizer.from_pretrained(
            args.ckpt_dir,
            trust_remote_code=True,
        )
        tokenizer.padding_side = "left"
        if not getattr(args, "task_description", None):
            args.task_description = (
                "Given a classical Chinese query about the Spring and Autumn Annals, "
                "retrieve relevant passages that describe the corresponding historical events."
            )
    else:
        if args.pooling is None:
            args.pooling = "cls"
        tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir)

    return tokenizer, model_type


def load_dual_encoder(ckpt_dir, device):
    """
    从 ckpt_dir/dual_model.pt 里恢复完整的 DualEncoder（包含时间头 + 时间上下文）。
    如果文件不存在，返回 None。
    """
    dual_path = os.path.join(ckpt_dir, "dual_model.pt")
    if not os.path.isfile(dual_path):
        print(f"[EVAL] dual_model.pt not found under {ckpt_dir}, skip time-context run.")
        return None

    ckpt = torch.load(dual_path, map_location=device)
    state_dict = ckpt["state_dict"]
    config = ckpt.get("config", {})

    dual_type = config.get("dual_type", "bert")
    normalize = config.get("normalize", True)
    num_gong = config.get("num_gong")
    num_year = config.get("num_year")
    num_month = config.get("num_month")
    use_time_context_pred = config.get("use_time_context_pred", False)
    time_emb_dim = config.get("time_emb_dim", 64)
    use_time_rel_bias = config.get("use_time_rel_bias", False)
    time_rel_dim = config.get("time_rel_dim", 32)

    if dual_type == "qwen":
        dual_encoder = QwenTimeDualEncoder(
            model_name_or_path=ckpt_dir,  # 直接从 fine-tuned encoder 权重目录加载
            normalize=normalize,
            num_gong=num_gong,
            num_year=num_year,
            num_month=num_month,
            use_time_context_pred=use_time_context_pred,
            time_emb_dim=time_emb_dim,
            use_time_rel_bias=use_time_rel_bias,
            time_rel_dim=time_rel_dim,
        ).to(device)
    else:
        dual_encoder = BertDualEncoder(
            model_name_or_path=ckpt_dir,
            pooling=config.get("pooling", "cls"),
            normalize=normalize,
            num_gong=num_gong,
            num_year=num_year,
            num_month=num_month,
            use_time_context_pred=use_time_context_pred,
            time_emb_dim=time_emb_dim,
            use_time_rel_bias=use_time_rel_bias,
            time_rel_dim=time_rel_dim,
        ).to(device)

    # 用 non-strict 防止以后你在模型里多加了一些新东西
    missing, unexpected = dual_encoder.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("[EVAL] Warning: loading dual_model.pt with non-strict state_dict.")
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")

    return dual_encoder


def main():
    args = parse_eval_args()
    set_seed(args.seed)

    # 1) 决定 output_dir + log_path（一个 eval 脚本只写一个 log 文件）
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.ckpt_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    log_time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(args.output_dir, f"eval_log_{log_time_str}.txt")

    # 2) tokenizer + 模型类型判定
    tokenizer, model_type = build_tokenizer_and_flags(args)

    # 3) 设备
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print(f"[EVAL] ckpt_dir   = {args.ckpt_dir}")
    print(f"[EVAL] model_type = {model_type}")
    print(f"[EVAL] is_qwen    = {args.is_qwen}")
    print(f"[EVAL] pooling    = {args.pooling}")
    print(f"[EVAL] output_dir = {args.output_dir}")

    # 4) 写一段公共的 eval 配置到 log
    with open(log_path, "a", encoding="utf-8") as f_log:
        f_log.write("=" * 80 + "\n")
        f_log.write(f"[EVAL] ckpt_dir   : {args.ckpt_dir}\n")
        f_log.write(f"[EVAL] meta_path  : {args.meta_path}\n")
        f_log.write(f"[EVAL] queries    : {args.queries_path}\n")
        f_log.write(f"[EVAL] splits     : {args.splits_path}\n")
        f_log.write(f"[EVAL] is_qwen    : {args.is_qwen}\n")
        f_log.write(f"[EVAL] pooling    : {args.pooling}\n")
        f_log.write(f"[EVAL] max_q_len  : {args.max_query_len}\n")
        f_log.write(f"[EVAL] max_d_len  : {args.max_doc_len}\n")
        f_log.write(f"[EVAL] batch_size : {args.batch_size}\n")
        f_log.write("=" * 80 + "\n\n")

    # 5) Run 1: 只用 encoder + pooling（不显式注入时间上下文）
    run_label_1 = "enc_only"
    print("\n" + "=" * 80)
    print(f"[EVAL] RUN 1: {run_label_1} (HF encoder + pooling, no time context)")
    with open(log_path, "a", encoding="utf-8") as f_log:
        f_log.write("\n" + "=" * 80 + "\n")
        f_log.write(f"[RUN] {run_label_1}  (encoder-only, no time context)\n")
        f_log.write("=" * 80 + "\n")

    evaluate_on_test_best_ckpt(
        best_dir=args.ckpt_dir,
        tokenizer=tokenizer,
        args=args,
        log_path=log_path,
        device=device,
        encoder=None,          # 让函数内部自己 AutoModel.from_pretrained
        run_label=run_label_1,
    )

    # # 6) Run 2: 使用完整 DualEncoder（显式调用时间头 + 时间上下文）
    # dual_encoder = load_dual_encoder(args.ckpt_dir, device)
    # if dual_encoder is not None:
    #     run_label_2 = "time_ctx"
    #     print("\n" + "=" * 80)
    #     print(f"[EVAL] RUN 2: {run_label_2} (DualEncoder with time heads / context)")
    #     with open(log_path, "a", encoding="utf-8") as f_log:
    #         f_log.write("\n" + "=" * 80 + "\n")
    #         f_log.write(f"[RUN] {run_label_2}  (DualEncoder, time heads + context)\n")
    #         f_log.write("=" * 80 + "\n")

    #     evaluate_on_test_best_ckpt(
    #         best_dir=args.ckpt_dir,
    #         tokenizer=tokenizer,
    #         args=args,
    #         log_path=log_path,
    #         device=device,
    #         encoder=dual_encoder,
    #         run_label=run_label_2,
    #     )
    # else:
    #     with open(log_path, "a", encoding="utf-8") as f_log:
    #         f_log.write("\n[WARN] dual_model.pt not found, skip time-context run.\n")

    print(f"\n[EVAL DONE] Summary written to: {log_path}")


if __name__ == "__main__":
    main()
