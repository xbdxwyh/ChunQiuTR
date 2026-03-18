import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from tqdm import tqdm
from torch import Tensor
import torch.distributed as dist  # ★ 新增
import numpy as np

# ============ DDP: 全局 gather ============

def gather_embeddings(x: torch.Tensor) -> torch.Tensor:
    """
    对任意形状的 Tensor 做 all_gather 并 cat 到第 0 维：
      - 未初始化 DDP: 直接返回 x
      - 已初始化 DDP: 返回 [world_size * B, ...]
    """
    if not dist.is_initialized():
        return x

    world_size = dist.get_world_size()
    x_list = [torch.zeros_like(x) for _ in range(world_size)]
    dist.all_gather(x_list, x)
    x_all = torch.cat(x_list, dim=0)
    return x_all


def contrastive_loss_global_inbatch(
    q_emb: torch.Tensor,
    p_emb: torch.Tensor,
    neg_emb: torch.Tensor = None,          # 可选：额外负样本 (B, D)
    temperature: float = 0.05,
    time_bias: torch.Tensor = None,        # (B,B) 本 rank 的时间偏置（只对“本 rank 的 q<->p 块”生效）
    # multipos 相关（接口和 contrastive_loss_inbatch 对齐）
    pos_mask: torch.Tensor = None,
    query_start_time_id: torch.Tensor = None,
    query_end_time_id:   torch.Tensor = None,
    pos_start_time_id:   torch.Tensor = None,
    pos_end_time_id:     torch.Tensor = None,
    symmetric: bool = True,
) -> torch.Tensor:
    """
    全局 in-batch 对比损失（支持 multipos + 额外 neg_emb）:

    - 若未初始化 DDP：直接退回 contrastive_loss_inbatch（**会**使用 neg_emb）
    - 若已初始化 DDP：
        * 无监督：global InfoNCE，正例 index = rank*B + i
        * multipos：用时间区间在【全局】上构造正例集合；
                    如果某行没有正例，就强制把自己 (rank*B + i) 作为正例。
        * 若提供 neg_emb：在全局 gallery 里 append 一块纯负样本列，不参与正例 mask
    """

    # ======================================================
    # 0) 单卡 / 非 DDP：直接走本地实现（这里也透传 neg_emb）
    # ======================================================
    if not dist.is_initialized():
        return contrastive_loss_inbatch(
            q_emb=q_emb,
            p_emb=p_emb,
            neg_emb=neg_emb,                      # ★ 新增：透传 neg_emb
            temperature=temperature,
            time_bias=time_bias,
            pos_mask=pos_mask,
            query_start_time_id=query_start_time_id,
            query_end_time_id=query_end_time_id,
            pos_start_time_id=pos_start_time_id,
            pos_end_time_id=pos_end_time_id,
            symmetric=symmetric,
        )

    rank = dist.get_rank()
    B = q_emb.size(0)
    device = q_emb.device

    # ======================================================
    # 1) 归一化 & 全局 gather
    # ======================================================
    q = F.normalize(q_emb, p=2, dim=-1)
    p = F.normalize(p_emb, p=2, dim=-1)

    use_extra_negs = (neg_emb is not None) and (neg_emb.numel() > 0)
    if use_extra_negs:
        n = F.normalize(neg_emb, p=2, dim=-1)  # (B, D)

    with torch.no_grad():
        q_all = gather_embeddings(q.detach())  # [B_global, D]
        p_all = gather_embeddings(p.detach())  # [B_global, D]
        if use_extra_negs:
            n_all = gather_embeddings(n.detach())  # [N_global, D]，一般 N_global == B_global

    B_global = p_all.size(0)

    # gallery_all: [所有 rank 的正样本, 所有 rank 的负样本]
    if use_extra_negs:
        gallery_all = torch.cat([p_all, n_all], dim=0)  # (B_global + N_global, D)
        G = gallery_all.size(0)
    else:
        gallery_all = p_all
        G = B_global

    # 当前 rank 的样本，在“正样本块”中的全局对角线位置
    local_idx = torch.arange(B, device=device)
    global_diag_idx = rank * B + local_idx
    assert global_diag_idx.max().item() < B_global  # 仍然只在 pos 块内

    # ======================================================
    # 2) 计算 logits（本 rank query vs 全局 gallery）
    # ======================================================
    logits_i2t = (q @ gallery_all.t()) / temperature     # [B, G]
    logits_t2i = (p @ q_all.t()) / temperature           # [B, B_global]（t2i 只看 pos↔query）

    # ---------- 修正后的 time_bias 逻辑 ----------
    # time_bias: 形状 (B,B)，行 = 本 rank 的 query，列 = 本 rank 的 pos
    # i2t: 当前 rank 的 query（行 0..B-1） vs 全局 gallery（列 0..G-1）
    #      只对“本 rank 的 pos 子块” [rank*B : rank*B + B] 加 bias
    # t2i: 当前 rank 的 pos（行 0..B-1） vs 全局 query（列 0..B_global-1）
    #      只对“本 rank 的 query 子块” [rank*B : rank*B + B] 加 bias.T
    if time_bias is not None:
        assert time_bias.shape == (B, B)
        col_start = rank * B
        col_end = col_start + B
        assert col_end <= B_global

        # i2t: (query_i, pos_j) 对应 (row=i, col=col_start + j_local)
        logits_i2t[:, col_start:col_end] = logits_i2t[:, col_start:col_end] + time_bias

        # t2i: (pos_i, query_j_local) 对应 (row=i, col=col_start + j_local)
        #      这里用 time_bias.T: [pos_i, query_j_local] 对应 [query_j_local, pos_i]
        logits_t2i[:, col_start:col_end] = logits_t2i[:, col_start:col_end] + time_bias.t()

    # ======================================================
    # 3) 是否有监督信号（multipos）
    # ======================================================
    has_supervision = (
        (pos_mask is not None) or
        (query_start_time_id is not None and query_end_time_id is not None and
         pos_start_time_id is not None and pos_end_time_id is not None)
    )

    # ---------- 3.1 无监督：global InfoNCE ----------
    if not has_supervision:
        targets = global_diag_idx          # 正例仍然只在正样本块 0..B_global-1
        loss_i2t = F.cross_entropy(logits_i2t, targets)
        if not symmetric:
            return loss_i2t
        loss_t2i = F.cross_entropy(logits_t2i, targets)
        return 0.5 * (loss_i2t + loss_t2i)

    # ---------- 3.2 如果外面直接给了本地 pos_mask ----------
    if pos_mask is not None:
        # 这里同样把 neg_emb 也透传回去
        return contrastive_loss_inbatch(
            q_emb=q_emb,
            p_emb=p_emb,
            neg_emb=neg_emb,                  # ★ 新增：透传 neg_emb
            temperature=temperature,
            time_bias=time_bias,
            pos_mask=pos_mask,
            query_start_time_id=query_start_time_id,
            query_end_time_id=query_end_time_id,
            pos_start_time_id=pos_start_time_id,
            pos_end_time_id=pos_end_time_id,
            symmetric=symmetric,
        )

    # ======================================================
    # 4) multipos（基于时间区间）: i2t
    # ======================================================
    qs = query_start_time_id.to(device)
    qe = query_end_time_id.to(device)
    ps = pos_start_time_id.to(device)
    pe = pos_end_time_id.to(device)

    # gather 所有 rank 的时间范围
    qs_all = gather_embeddings(qs)
    qe_all = gather_embeddings(qe)
    ps_all = gather_embeddings(ps)
    pe_all = gather_embeddings(pe)

    # start <= end（保持和你 13 号版本一致）
    q_min = torch.minimum(qs, qe)             # [B]
    q_max = torch.maximum(qs, qe)
    p_min_all = torch.minimum(ps_all, pe_all) # [B_global]
    p_max_all = torch.maximum(ps_all, pe_all)

    # 有交集就视为正例（只针对 pos_all）
    pos_mask_i2t_pos = (q_min[:, None] <= p_max_all[None, :]) & \
                       (q_max[:, None] >= p_min_all[None, :])   # [B, B_global]

    # 扩展到 full gallery：neg 部分全部 False
    pos_mask_i2t = torch.zeros(B, G, dtype=torch.bool, device=device)
    pos_mask_i2t[:, :B_global] = pos_mask_i2t_pos

    # 保证每行至少一个正例：如果这一行全 False，补上自己的 global_diag_idx
    row_has_pos = pos_mask_i2t.any(dim=1)
    if (~row_has_pos).any():
        bad_rows = torch.where(~row_has_pos)[0]
        pos_mask_i2t[bad_rows, global_diag_idx[bad_rows]] = True

    # InfoNCE: -log (sum_pos exp / sum_all exp)
    log_prob_i2t = logits_i2t - torch.logsumexp(logits_i2t, dim=1, keepdim=True)
    pos_log_prob_i2t = log_prob_i2t.masked_fill(~pos_mask_i2t, float("-inf"))
    loss_i2t = -torch.logsumexp(pos_log_prob_i2t, dim=1).mean()

    if not symmetric:
        return loss_i2t

    # ======================================================
    # 5) multipos：t2i，对称方向（只在 pos_all <-> q_all 范围）
    # ======================================================
    p_min = torch.minimum(ps, pe)
    p_max = torch.maximum(ps, pe)
    q_min_all = torch.minimum(qs_all, qe_all)
    q_max_all = torch.maximum(qs_all, qe_all)

    pos_mask_t2i = (p_min[:, None] <= q_max_all[None, :]) & \
                   (p_max[:, None] >= q_min_all[None, :])   # [B, B_global]

    col_has_pos = pos_mask_t2i.any(dim=1)
    if (~col_has_pos).any():
        bad_rows = torch.where(~col_has_pos)[0]
        pos_mask_t2i[bad_rows, global_diag_idx[bad_rows]] = True

    log_prob_t2i = logits_t2i - torch.logsumexp(logits_t2i, dim=1, keepdim=True)
    pos_log_prob_t2i = log_prob_t2i.masked_fill(~pos_mask_t2i, float("-inf"))
    loss_t2i = -torch.logsumexp(pos_log_prob_t2i, dim=1).mean()

    return 0.5 * (loss_i2t + loss_t2i)



# ============ 对比学习相关损失函数 ============

def last_token_pool(last_hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Qwen 官方 last_token_pool：
    - left padding: 直接取最后一个 token
    - 否则根据 attention_mask 取实际序列最后一个 token
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]

def contrastive_loss_inbatch(
    q_emb: torch.Tensor,
    p_emb: torch.Tensor,
    neg_emb: torch.Tensor = None,        # ★ 新增：可选负样本
    temperature: float = 0.05,
    time_bias: torch.Tensor = None,      # (B,B) additive bias for pos-pos block
    # ===== supervised multipos options (任选其一) =====
    pos_mask: torch.Tensor = None,       # (B,B) bool, True 表示 q_i 与 p_j 是正例（只对正样本块）
    query_start_time_id: torch.Tensor = None,  # (B,)
    query_end_time_id:   torch.Tensor = None,  # (B,)
    pos_start_time_id:   torch.Tensor = None,  # (B,)
    pos_end_time_id:     torch.Tensor = None,  # (B,)
    symmetric: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    - 无监督（默认）：与原版一致，labels = arange(B)
    - 有监督（multipos）：每个 query 在正样本块 (B,B) 里可以有多个正例；
      neg_emb 只在 denominator 里当额外负样本，不会被当成正例。
    """

    # ---- normalize ----
    q = F.normalize(q_emb, p=2, dim=-1)
    p = F.normalize(p_emb, p=2, dim=-1)
    B = q.size(0)
    device = q.device

    # ---- 拼 gallery：正样本 + 可选负样本 ----
    use_extra_negs = (neg_emb is not None) and (neg_emb.numel() > 0)
    if use_extra_negs:
        n = F.normalize(neg_emb, p=2, dim=-1)
        gallery = torch.cat([p, n], dim=0)   # (B + Bn, D)
    else:
        gallery = p                          # (B, D)

    G = gallery.size(0)

    # ---- logits: i2t (query vs full gallery) ----
    logits_i2t = (q @ gallery.t()) / temperature  # (B, G)

    # time_bias 只作用在正样本块 (B,B) 上
    if time_bias is not None:
        assert time_bias.shape == (B, B)
        logits_i2t[:, :B] = logits_i2t[:, :B] + time_bias

    # ==========================================================
    # 1) 无监督分支：完全保持原来的 InfoNCE 语义
    # ==========================================================
    has_supervision = (
        (pos_mask is not None) or
        (query_start_time_id is not None and query_end_time_id is not None and
         pos_start_time_id is not None and pos_end_time_id is not None)
    )

    if not has_supervision:
        targets = torch.arange(B, device=device)  # 正例都在 pos-block 的对角线上
        loss_i2t = F.cross_entropy(logits_i2t, targets)

        if not symmetric:
            return loss_i2t

        # t2i 方向：只在正样本块 (B,B) 上做（不使用 neg_emb）
        logits_t2i = (p @ q.t()) / temperature   # (B,B)
        if time_bias is not None:
            logits_t2i = logits_t2i + time_bias.t()
        loss_t2i = F.cross_entropy(logits_t2i, targets)

        return 0.5 * (loss_i2t + loss_t2i)

    # ==========================================================
    # 2) 有监督分支：multi-positive InfoNCE / supervised contrastive
    # ==========================================================

    # ---- 先在正样本块 (B,B) 上构造 pos_mask_pos ----
    if pos_mask is None:
        qs = query_start_time_id.to(device)
        qe = query_end_time_id.to(device)
        ps = pos_start_time_id.to(device)
        pe = pos_end_time_id.to(device)

        # 保证 start <= end（防数据脏）
        q_min = torch.minimum(qs, qe)
        q_max = torch.maximum(qs, qe)
        p_min = torch.minimum(ps, pe)
        p_max = torch.maximum(ps, pe)

        # overlap: [q_min,q_max] 与 [p_min,p_max] 有交集
        pos_mask_pos = (q_min[:, None] <= p_max[None, :]) & \
                       (q_max[:, None] >= p_min[None, :])     # (B,B)
    else:
        pos_mask_pos = pos_mask.to(dtype=torch.bool, device=device)
        assert pos_mask_pos.shape == (B, B)

    # ---- safety: 保证每行至少一个正例（只在正样本块里补对角线）----
    diag = torch.eye(B, dtype=torch.bool, device=device)
    row_has_pos = pos_mask_pos.any(dim=1)
    if (~row_has_pos).any():
        pos_mask_pos = pos_mask_pos | (diag & (~row_has_pos)[:, None])

    # ---- 如果有额外 neg，则把 pos_mask_pos 扩展到 full gallery ----
    if use_extra_negs:
        pos_mask_i2t = torch.zeros(B, G, dtype=torch.bool, device=device)
        pos_mask_i2t[:, :B] = pos_mask_pos
    else:
        pos_mask_i2t = pos_mask_pos

    # ---- i2t loss: -log( sum_pos exp(logit) / sum_all exp(logit) ) ----
    log_prob_i2t = logits_i2t - torch.logsumexp(logits_i2t, dim=1, keepdim=True)  # (B,G)
    pos_log_prob_i2t = log_prob_i2t.masked_fill(~pos_mask_i2t, float("-inf"))
    loss_i2t = -torch.logsumexp(pos_log_prob_i2t, dim=1).mean()

    if not symmetric:
        return loss_i2t

    # ---- t2i loss: 仍然只对正样本块 (B,B) 做对称方向 ----
    logits_t2i = (p @ q.t()) / temperature      # (B,B)
    if time_bias is not None:
        logits_t2i = logits_t2i + time_bias.t()

    pos_mask_t = pos_mask_pos.t()
    col_has_pos = pos_mask_t.any(dim=1)
    if (~col_has_pos).any():
        pos_mask_t = pos_mask_t | (diag & (~col_has_pos)[:, None])

    log_prob_t2i = logits_t2i - torch.logsumexp(logits_t2i, dim=1, keepdim=True)
    pos_log_prob_t2i = log_prob_t2i.masked_fill(~pos_mask_t, float("-inf"))
    loss_t2i = -torch.logsumexp(pos_log_prob_t2i, dim=1).mean()

    return 0.5 * (loss_i2t + loss_t2i)


# ============ Triplet loss ============

def triplet_loss(q_emb, pos_emb, neg_emb, margin: float = 0.2):
    """
    最简单形式：用 cosine 相似度 + hinge
    loss = max(0, margin - sim(q, pos) + sim(q, neg))
    """
    # q_emb / pos / neg 已经做了 L2 norm，所以 dot = cosine
    pos_sim = (q_emb * pos_emb).sum(dim=-1)  # (B,)
    neg_sim = (q_emb * neg_emb).sum(dim=-1)  # (B,)
    loss = F.relu(margin - pos_sim + neg_sim).mean()
    return loss


def encode_texts_bert(
    texts,
    tokenizer,
    encoder,      # 可以是 AutoModel，也可以是 BertDualEncoder / QwenDualEncoder
    device,
    max_length=128,
    batch_size=64,
    pool_mode="cls",  # "cls" / "mean" / "last_token"
):
    """
    用当前 encoder 编码一批文本，返回 L2-normalized 向量 (N, D).

    兼容两种情况：
      1) encoder 是 HF AutoModel（没有 .encode 方法）
      2) encoder 是我们自定义的 DualEncoder（有 .encode 方法）
    """
    was_training = getattr(encoder, "training", False)
    if hasattr(encoder, "eval"):
        encoder.eval()

    all_embs = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            if hasattr(encoder, "encode"):
                # DualEncoder：直接用它的 encode（里边已经做了 pooling）
                emb = encoder.encode(
                    batch["input_ids"],
                    batch["attention_mask"],
                )  # (B, D)
            else:
                # HF AutoModel：手动 pooling
                outputs = encoder(**batch)
                hidden = outputs.last_hidden_state  # (B, L, D)

                if pool_mode == "cls":
                    emb = hidden[:, 0]
                elif pool_mode == "mean":
                    mask = batch["attention_mask"].unsqueeze(-1)  # (B, L, 1)
                    summed = (hidden * mask).sum(dim=1)
                    lengths = mask.sum(dim=1).clamp(min=1)
                    emb = summed / lengths
                elif pool_mode == "last_token":
                    emb = last_token_pool(hidden, batch["attention_mask"])
                else:
                    raise ValueError(f"Unknown pool_mode: {pool_mode}")

                emb = torch.nn.functional.normalize(emb, p=2, dim=1)

            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embs.append(emb.cpu())

    if was_training and hasattr(encoder, "train"):
        encoder.train()

    return torch.cat(all_embs, dim=0)



# ========== 工具函数 ==========

def simple_collate(batch):
    """Eval 阶段：保持 list[dict] 的形式。"""
    return batch


# def compute_retrieval_metrics(
#     query_embs: torch.Tensor,          # (Q, D)
#     gallery_embs: torch.Tensor,        # (G, D)
#     gold_indices_list,                 # List[List[int]]
#     ks=(1, 5, 10),
# ):
#     q = query_embs
#     g = gallery_embs
#     Q = q.size(0)

#     recalls = {k: 0 for k in ks}
#     mrr_at_10 = 0.0

#     with torch.no_grad():
#         sims = torch.matmul(q, g.t())  # (Q, G)
#         for qi in range(Q):
#             sim_row = sims[qi]
#             gold_indices = gold_indices_list[qi]
#             if not gold_indices:
#                 continue

#             topv, topi = torch.topk(sim_row, k=min(10, sim_row.size(0)))
#             topi = topi.tolist()

#             # Recall@k
#             for k in ks:
#                 k_cut = min(k, len(topi))
#                 if any(idx in topi[:k_cut] for idx in gold_indices):
#                     recalls[k] += 1

#             # MRR@10
#             rank = None
#             for r, idx in enumerate(topi):
#                 if idx in gold_indices:
#                     rank = r + 1
#                     break
#             if rank is not None:
#                 mrr_at_10 += 1.0 / rank

#     metrics = {}
#     for k in ks:
#         metrics[f"Recall@{k}"] = recalls[k] / Q if Q > 0 else 0.0
#     metrics["MRR@10"] = mrr_at_10 / Q if Q > 0 else 0.0
#     return metrics

def compute_retrieval_metrics(
    query_embs: torch.Tensor,          # (Q, D)
    gallery_embs: torch.Tensor,        # (G, D)
    gold_indices_list,                 # List[List[int]]
    ks=(1, 5, 10),
):
    q = query_embs
    g = gallery_embs
    Q = q.size(0)

    recalls = {k: 0 for k in ks}
    mrr_at_10 = 0.0
    ndcg_at_10 = 0.0   # NEW

    # 预计算 IDCG@10：只依赖 gold 数量（取 min(|gold|,10) 个 1）
    def ideal_dcg_10(num_gold: int) -> float:
        m = min(num_gold, 10)
        if m <= 0:
            return 0.0
        return sum(1.0 / math.log2(i + 2) for i in range(m))  # i=0..m-1 -> rank=1..m

    with torch.no_grad():
        sims = torch.matmul(q, g.t())  # (Q, G)
        for qi in range(Q):
            sim_row = sims[qi]
            gold_indices = gold_indices_list[qi]
            if not gold_indices:
                continue

            topv, topi = torch.topk(sim_row, k=min(10, sim_row.size(0)))
            topi = topi.tolist()

            gold_set = set(gold_indices)

            # Recall@k (Hit/Success@k)
            for k in ks:
                k_cut = min(k, len(topi))
                if any(idx in gold_set for idx in topi[:k_cut]):
                    recalls[k] += 1

            # MRR@10
            rank = None
            for r, idx in enumerate(topi):
                if idx in gold_set:
                    rank = r + 1
                    break
            if rank is not None:
                mrr_at_10 += 1.0 / rank

            # nDCG@10 (binary relevance)
            dcg = 0.0
            for r, idx in enumerate(topi, start=1):  # r=1..10
                if idx in gold_set:
                    dcg += 1.0 / math.log2(r + 1)
            idcg = ideal_dcg_10(len(gold_set))
            if idcg > 0:
                ndcg_at_10 += (dcg / idcg)

    metrics = {}
    for k in ks:
        metrics[f"Recall@{k}"] = recalls[k] / Q if Q > 0 else 0.0
    metrics["MRR@10"] = mrr_at_10 / Q if Q > 0 else 0.0
    metrics["nDCG@10"] = ndcg_at_10 / Q if Q > 0 else 0.0  # NEW
    return metrics

def compute_retrieval_per_query(
    query_embs: torch.Tensor,          # (Q, D)
    gallery_embs: torch.Tensor,        # (G, D)
    gold_indices_list,                 # List[List[int]]
    ks=(1, 5, 10),
    topk=10,
):
    """
    返回逐 query 的：
      - hit@k (0/1)
      - rr@topk (float)
      - first_rank (1..topk or 0 if miss)
    不改变任何 evaluation split / 指标定义，只是把逐条结果保存出来。
    """
    q = query_embs
    g = gallery_embs
    Q = q.size(0)
    K = min(topk, g.size(0))

    hit = {k: torch.zeros(Q, dtype=torch.int32) for k in ks}
    rr = torch.zeros(Q, dtype=torch.float32)
    first_rank = torch.zeros(Q, dtype=torch.int32)

    with torch.no_grad():
        sims = torch.matmul(q, g.t())  # (Q, G)
        for qi in range(Q):
            gold = gold_indices_list[qi]
            if not gold:
                continue

            _, topi = torch.topk(sims[qi], k=K)
            topi = topi.tolist()

            for k in ks:
                k_cut = min(k, len(topi))
                if any(idx in topi[:k_cut] for idx in gold):
                    hit[k][qi] = 1

            rank = 0
            for r, idx in enumerate(topi, start=1):
                if idx in gold:
                    rank = r
                    break
            if rank > 0:
                first_rank[qi] = rank
                rr[qi] = 1.0 / rank

    out = {
        "hit": {k: hit[k] for k in ks},
        "rr": rr,
        "first_rank": first_rank,
    }
    return out



# def _point_singlepos_loss_direction(
#     anchor_emb: torch.Tensor,
#     other_emb_all: torch.Tensor,
#     extra_neg_all: torch.Tensor = None,      # ★ 新增：可选额外负样本 (G_neg, D)
#     anchor_start: torch.Tensor = None,
#     anchor_end: torch.Tensor = None,
#     other_start_all: torch.Tensor = None,
#     other_end_all: torch.Tensor = None,
#     temperature: float = 0.05,
# ) -> torch.Tensor:
#     """
#     单方向的 point 单正例 InfoNCE:

#     - anchor_emb: 当前 rank 的 query/doc 向量，形状 [B, D]
#     - other_emb_all: 全局的 doc/query 向量，形状 [B_global, D]
#     - extra_neg_all: 额外负样本（不需要时间标签），形状 [G_neg, D]，可以为 None
#     - anchor_start/end: 当前 rank 的时间 id，形状 [B]
#     - other_start_all/end_all: 全局时间 id，形状 [B_global]

#     只使用下面这些 anchor 做 loss：
#       * anchor 是 point: anchor_start == anchor_end
#       * other 是 point: other_start_all == other_end_all
#       * 且该 anchor 的时间戳在 other_all 里「恰好出现一次」

#     对这些合法 anchor:
#       正例 = 唯一时间相同的 other（point-doc）
#       负例 = 所有其它 point-type other + 所有 extra_neg_all
#              （extra_neg_all 不参与时间筛选，只当纯负例）
#     """

#     device = anchor_emb.device
#     B = anchor_emb.size(0)
#     B_global = other_emb_all.size(0)

#     if anchor_start is None or anchor_end is None \
#        or other_start_all is None or other_end_all is None:
#         # 没时间标签就直接返回 0（保持图连通）
#         return anchor_emb.new_tensor(0.0)

#     # 归一化
#     a = F.normalize(anchor_emb, p=2, dim=-1)           # [B, D]
#     o_all = F.normalize(other_emb_all, p=2, dim=-1)    # [B_global, D]

#     as_ = anchor_start.to(device)      # [B]
#     ae_ = anchor_end.to(device)        # [B]
#     os_all = other_start_all.to(device)  # [B_global]
#     oe_all = other_end_all.to(device)    # [B_global]

#     # 只把 point-type 的 other(doc/query) 纳入“候选正例”
#     other_is_point = (os_all == oe_all)                  # [B_global]
#     point_other_idx = torch.where(other_is_point)[0]     # [G_point]
#     G_point = point_other_idx.size(0)
#     if G_point == 0:
#         return anchor_emb.new_tensor(0.0)

#     gallery_pos = o_all[point_other_idx]                 # [G_point, D]

#     # ====== 把额外负样本并入 gallery ======
#     if extra_neg_all is not None and extra_neg_all.numel() > 0:
#         extra = F.normalize(extra_neg_all.to(device), p=2, dim=-1)   # [G_neg, D]
#         gallery_all = torch.cat([gallery_pos, extra], dim=0)         # [G_point + G_neg, D]
#     else:
#         gallery_all = gallery_pos                                    # [G_point, D]

#     G_all = gallery_all.size(0)

#     # 预先算好 anchor vs full gallery 的 logits
#     logits_all = (a @ gallery_all.t()) / temperature                 # [B, G_all]

#     # global_index -> gallery_pos_index 的映射
#     # （注意：只给 point-doc 建映射，extra_neg_all 没有对应的时间，也不需要映射）
#     global_to_point = torch.full(
#         (B_global,),
#         -1,
#         dtype=torch.long,
#         device=device,
#     )
#     global_to_point[point_other_idx] = torch.arange(
#         G_point, device=device
#     )  # 只映射到 [0, G_point-1]，即正例/point-doc 区域

#     # 对每个 anchor 找「唯一一个」时间完全相同的 point-doc
#     is_anchor_point = (as_ == ae_)                      # [B]
#     pos_global_idx = torch.full(
#         (B,),
#         -1,
#         dtype=torch.long,
#         device=device,
#     )

#     for i in range(B):
#         if not is_anchor_point[i]:
#             continue

#         # 候选：point-doc 且时间完全相等
#         same_time = other_is_point & (os_all == as_[i]) & (oe_all == ae_[i])
#         idxs = torch.where(same_time)[0]

#         # 只保留“恰好一个”匹配的样本
#         if idxs.numel() == 1:
#             pos_global_idx[i] = idxs.item()

#     valid_mask = (pos_global_idx >= 0)                  # [B]
#     if valid_mask.sum() == 0:
#         # 没有任何干净的 point 样本
#         return anchor_emb.new_tensor(0.0)

#     # 只用合法 anchor 的 logits
#     logits = logits_all[valid_mask]                     # [B_valid, G_all]
#     # 正例 index 仍然只能落在 [0, G_point-1] 的“point-doc 区域”
#     targets = global_to_point[pos_global_idx[valid_mask]]  # [B_valid]
#     assert (targets >= 0).all() and (targets < G_point).all()

#     loss = F.cross_entropy(logits, targets)
#     return loss


# def point_singlepos_loss_inbatch(
#     q_emb: torch.Tensor,
#     p_emb: torch.Tensor,
#     neg_emb: torch.Tensor = None,                # ★ 现在真正使用：作为额外负样本
#     temperature: float = 0.05,
#     query_start_time_id: torch.Tensor = None,
#     query_end_time_id:   torch.Tensor = None,
#     pos_start_time_id:   torch.Tensor = None,
#     pos_end_time_id:     torch.Tensor = None,
#     symmetric: bool = True,
# ) -> torch.Tensor:
#     """
#     单卡版本的 point-type 单正例 InfoNCE 损失：

#     - 只用：
#         * point-type query (start == end)
#         * point-type doc (start == end)
#         * 且某个 query 在当前 batch 的 doc 里，时间戳恰好匹配一个 doc
#       这些样本参与损失。
#     - 正例 = 该唯一 doc；
#       负例 = 所有其它 point-type doc + neg_emb（如果提供）。
#     - window-type doc 完全忽略。
#     """

#     device = q_emb.device

#     if query_start_time_id is None or query_end_time_id is None \
#        or pos_start_time_id is None or pos_end_time_id is None:
#         # 没有监督时间标签，直接返回 0
#         return q_emb.new_tensor(0.0)

#     # 方向 1：query -> doc
#     extra_neg_local = neg_emb if (neg_emb is not None and neg_emb.numel() > 0) else None

#     loss_q2d = _point_singlepos_loss_direction(
#         anchor_emb=q_emb,
#         other_emb_all=p_emb,
#         extra_neg_all=extra_neg_local,              # ★ 把本 batch 的 neg_emb 当作额外负例
#         anchor_start=query_start_time_id,
#         anchor_end=query_end_time_id,
#         other_start_all=pos_start_time_id,
#         other_end_all=pos_end_time_id,
#         temperature=temperature,
#     )

#     if not symmetric:
#         return loss_q2d

#     # 方向 2：doc -> query
#     # 为了和你原来的 contrastive_loss_inbatch 一致，这一方向暂不使用 neg_emb
#     loss_d2q = _point_singlepos_loss_direction(
#         anchor_emb=p_emb,
#         other_emb_all=q_emb,
#         extra_neg_all=None,                        # ★ 不传 neg
#         anchor_start=pos_start_time_id,
#         anchor_end=pos_end_time_id,
#         other_start_all=query_start_time_id,
#         other_end_all=query_end_time_id,
#         temperature=temperature,
#     )

#     return 0.5 * (loss_q2d + loss_d2q)


# def point_singlepos_loss_global_inbatch(
#     q_emb: torch.Tensor,
#     p_emb: torch.Tensor,
#     neg_emb: torch.Tensor = None,                # ★ 同样真正使用：作为额外负样本
#     temperature: float = 0.05,
#     query_start_time_id: torch.Tensor = None,
#     query_end_time_id:   torch.Tensor = None,
#     pos_start_time_id:   torch.Tensor = None,
#     pos_end_time_id:     torch.Tensor = None,
#     symmetric: bool = True,
# ) -> torch.Tensor:
#     """
#     DDP 版本的 point-type 单正例 InfoNCE：

#     - 若未初始化 DDP：退回 point_singlepos_loss_inbatch
#     - 若已初始化 DDP：
#         * anchor = 当前 rank 的 query_i / doc_i
#         * other_all = 全局 gather 的 doc / query
#         * extra_neg_all = 全局 gather 的 neg_emb，用作额外负样本（不参与时间匹配，只当负例）
#         * 只用 (start == end) 的 point-type query/doc
#         * 对于某个 point-query，如果在「全局 doc」里时间戳恰好匹配一个 point-doc，
#           则该 doc 为正例，其它 point-doc + 所有 extra_neg_all 为负例；
#           若 0 个或 >1 个匹配则忽略该 query。
#         * window-type doc 完全忽略。
#     """

#     # 单机 / 未启用 DDP：直接走本地版
#     if not dist.is_initialized():
#         return point_singlepos_loss_inbatch(
#             q_emb=q_emb,
#             p_emb=p_emb,
#             neg_emb=neg_emb,
#             temperature=temperature,
#             query_start_time_id=query_start_time_id,
#             query_end_time_id=query_end_time_id,
#             pos_start_time_id=pos_start_time_id,
#             pos_end_time_id=pos_end_time_id,
#             symmetric=symmetric,
#         )

#     device = q_emb.device

#     if query_start_time_id is None or query_end_time_id is None \
#        or pos_start_time_id is None or pos_end_time_id is None:
#         return q_emb.new_tensor(0.0)

#     # 先 normalize 一下本 rank 的特征（方便 gather）
#     q_norm = F.normalize(q_emb, p=2, dim=-1)
#     p_norm = F.normalize(p_emb, p=2, dim=-1)

#     # ====== gather 全局 doc/query 及其时间标签 ======
#     with torch.no_grad():
#         # 全局 doc
#         p_all = gather_embeddings(p_norm.detach())                     # [B_global, D]
#         ps_all = gather_embeddings(pos_start_time_id.to(device))       # [B_global]
#         pe_all = gather_embeddings(pos_end_time_id.to(device))         # [B_global]

#         # 全局 query
#         q_all = gather_embeddings(q_norm.detach())                     # [B_global, D]
#         qs_all = gather_embeddings(query_start_time_id.to(device))     # [B_global]
#         qe_all = gather_embeddings(query_end_time_id.to(device))       # [B_global]

#         # 全局 neg（如果有的话）
#         if neg_emb is not None and neg_emb.numel() > 0:
#             n_norm = F.normalize(neg_emb.to(device), p=2, dim=-1)
#             n_all = gather_embeddings(n_norm.detach())                 # [G_neg_global, D]
#         else:
#             n_all = None

#     # 当前 rank 的本地时间标签
#     qs_local = query_start_time_id.to(device)
#     qe_local = query_end_time_id.to(device)
#     ps_local = pos_start_time_id.to(device)
#     pe_local = pos_end_time_id.to(device)

#     # 方向 1：query(local) -> doc(global)（用全局 neg 作为额外负例）
#     loss_q2d = _point_singlepos_loss_direction(
#         anchor_emb=q_emb,
#         other_emb_all=p_all,
#         extra_neg_all=n_all,                           # ★ 使用全局 neg_emb
#         anchor_start=qs_local,
#         anchor_end=qe_local,
#         other_start_all=ps_all,
#         other_end_all=pe_all,
#         temperature=temperature,
#     )

#     if not symmetric:
#         return loss_q2d

#     # 方向 2：doc(local) -> query(global)（不使用 neg，以保持和原对比损失的风格一致）
#     loss_d2q = _point_singlepos_loss_direction(
#         anchor_emb=p_emb,
#         other_emb_all=q_all,
#         extra_neg_all=None,                            # ★ 不使用 neg
#         anchor_start=ps_local,
#         anchor_end=pe_local,
#         other_start_all=qs_all,
#         other_end_all=qe_all,
#         temperature=temperature,
#     )

#     return 0.5 * (loss_q2d + loss_d2q)


def point_singlepos_loss_inbatch(
    q_emb: torch.Tensor,
    p_emb: torch.Tensor,
    neg_emb: torch.Tensor = None,                # 现在会真正参与为负例
    temperature: float = 0.05,
    query_start_time_id: torch.Tensor = None,
    query_end_time_id:   torch.Tensor = None,
    pos_start_time_id:   torch.Tensor = None,
    pos_end_time_id:     torch.Tensor = None,
    symmetric: bool = True,                      # 这里实际上忽略，只做单向 q_point -> gallery
) -> torch.Tensor:
    """
    单卡版本的 point-type 时间辅助 InfoNCE：

    - anchor：所有 point-type query（start == end）。
    - gallery：统一的一份 [Q_window, P_doc, neg_emb]。
      * Q_window：所有 window-type query（qs != qe），带时间区间 [qs, qe]
      * P_doc：所有 doc（pos_emb），带时间区间 [ps, pe]
      * neg_emb：没有时间标签，永远是负例
    - 对于每个 point-anchor i（时间 t_i），正例集合是：
        * 所有满足其时间区间覆盖 t_i 的 Q_window
        * 所有满足其时间区间覆盖 t_i 的 P_doc
      其它（含 neg）全部视作负例。
    - 如果某个 point-anchor 找不到任何正例，则忽略该 anchor。
    """

    device = q_emb.device

    # 没有时间监督信息，直接返回 0
    if (
        query_start_time_id is None or query_end_time_id is None
        or pos_start_time_id   is None or pos_end_time_id   is None
    ):
        return q_emb.new_tensor(0.0)

    # ====== 归一化 ======
    q = F.normalize(q_emb, p=2, dim=-1)    # [B, D]
    p = F.normalize(p_emb, p=2, dim=-1)    # [B, D]

    use_extra_negs = (neg_emb is not None) and (neg_emb.numel() > 0)
    if use_extra_negs:
        n = F.normalize(neg_emb, p=2, dim=-1)   # [B_neg, D]
        B_neg = n.size(0)
    else:
        n = None
        B_neg = 0

    # ====== 拆 query: point / window ======
    qs = query_start_time_id.to(device)    # [B]
    qe = query_end_time_id.to(device)      # [B]
    ps = pos_start_time_id.to(device)      # [B]
    pe = pos_end_time_id.to(device)        # [B]

    is_point_q  = (qs == qe)
    is_window_q = ~is_point_q

    if not is_point_q.any():
        # 这个 batch 没有 point-type query，loss=0
        return q_emb.new_tensor(0.0)

    # anchors: 所有 point-query
    q_point = q[is_point_q]                # [B_point, D]
    t_point = qs[is_point_q]               # [B_point]
    B_point = q_point.size(0)

    # window-query 作为 gallery 的一部分
    q_window   = q[is_window_q]            # [B_win, D]
    qs_window  = qs[is_window_q]           # [B_win]
    qe_window  = qe[is_window_q]           # [B_win]
    B_win      = q_window.size(0)

    # doc 部分
    p_all = p                              # [B_doc, D]
    B_doc = p_all.size(0)

    # ====== 构造统一 gallery: [Q_window, P_doc, neg] ======
    gallery_parts = []
    if B_win > 0:
        gallery_parts.append(q_window)
    gallery_parts.append(p_all)
    if use_extra_negs:
        gallery_parts.append(n)

    gallery = torch.cat(gallery_parts, dim=0)  # [G, D]
    G = gallery.size(0)

    # ====== 为 point-anchor 构造 positive mask ======
    # 只对 Q_window 和 P_doc 有时间标签；neg 没有，天然都是负例
    pos_mask = torch.zeros(B_point, G, dtype=torch.bool, device=device)

    col = 0
    # --- Q_window 段: [0 : B_win] ---
    if B_win > 0:
        # 条件：qs_window <= t_point <= qe_window
        # t_point: [B_point] -> [B_point, 1]
        # qs_window, qe_window: [B_win] -> [1, B_win]
        t_point_ = t_point[:, None]                    # [B_point, 1]
        qs_w_    = qs_window[None, :]                  # [1, B_win]
        qe_w_    = qe_window[None, :]                  # [1, B_win]

        pos_win = (qs_w_ <= t_point_) & (t_point_ <= qe_w_)   # [B_point, B_win]
        pos_mask[:, col : col + B_win] = pos_win
        col += B_win

    # --- P_doc 段: [col : col + B_doc] ---
    t_point_ = t_point[:, None]                        # [B_point, 1]
    ps_      = ps[None, :]                             # [1, B_doc]
    pe_      = pe[None, :]                             # [1, B_doc]

    pos_doc = (ps_ <= t_point_) & (t_point_ <= pe_)    # [B_point, B_doc]
    pos_mask[:, col : col + B_doc] = pos_doc
    col += B_doc

    # neg 段（如果有）天然全 False，不用动

    # ====== 去掉没有任何正例的 anchor ======
    row_has_pos = pos_mask.any(dim=1)                  # [B_point]
    if not row_has_pos.any():
        return q_emb.new_tensor(0.0)

    q_point_valid   = q_point[row_has_pos]             # [B_valid, D]
    pos_mask_valid  = pos_mask[row_has_pos]            # [B_valid, G]

    # ====== 计算单向 InfoNCE: q_point_valid -> gallery ======
    logits = (q_point_valid @ gallery.t()) / temperature   # [B_valid, G]

    log_prob      = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_log_prob  = log_prob.masked_fill(~pos_mask_valid, float("-inf"))
    loss_per_row  = -torch.logsumexp(pos_log_prob, dim=1)  # [B_valid]
    loss          = loss_per_row.mean()

    return loss


# def point_singlepos_loss_global_inbatch(
#     q_emb: torch.Tensor,
#     p_emb: torch.Tensor,
#     neg_emb: torch.Tensor = None,                # 参与为负例
#     temperature: float = 0.05,
#     query_start_time_id: torch.Tensor = None,
#     query_end_time_id:   torch.Tensor = None,
#     pos_start_time_id:   torch.Tensor = None,
#     pos_end_time_id:     torch.Tensor = None,
#     symmetric: bool = True,                      # 同样忽略，只做 q_point -> gallery
# ) -> torch.Tensor:
#     """
#     DDP 版本的 point-type 时间辅助 InfoNCE：

#     - 若未初始化 DDP：直接退回单卡版 point_singlepos_loss_inbatch。
#     - 若已初始化 DDP：
#         * anchor = 当前 rank 的所有 point-query
#         * gallery = [全局 window-query, 全局 doc, 全局 neg]（统一一块）
#         * 正例 = 时间区间覆盖该 point 的那些 gallery 元素（window-query + doc）
#         * 其它（含 neg）全部负例。
#     """

#     # 非 DDP 直接用单卡实现
#     if not dist.is_initialized():
#         return point_singlepos_loss_inbatch(
#             q_emb=q_emb,
#             p_emb=p_emb,
#             neg_emb=neg_emb,
#             temperature=temperature,
#             query_start_time_id=query_start_time_id,
#             query_end_time_id=query_end_time_id,
#             pos_start_time_id=pos_start_time_id,
#             pos_end_time_id=pos_end_time_id,
#             symmetric=symmetric,
#         )

#     device = q_emb.device

#     # 没有时间监督信息，直接返回 0
#     if (
#         query_start_time_id is None or query_end_time_id is None
#         or pos_start_time_id   is None or pos_end_time_id   is None
#     ):
#         return q_emb.new_tensor(0.0)

#     # ====== 本地归一化 ======
#     q_local = F.normalize(q_emb, p=2, dim=-1)    # [B_local, D]
#     p_local = F.normalize(p_emb, p=2, dim=-1)    # [B_local, D]

#     use_extra_negs = (neg_emb is not None) and (neg_emb.numel() > 0)
#     if use_extra_negs:
#         n_local = F.normalize(neg_emb, p=2, dim=-1)  # [B_local, D]
#     else:
#         n_local = None

#     qs_local = query_start_time_id.to(device)    # [B_local]
#     qe_local = query_end_time_id.to(device)      # [B_local]
#     ps_local = pos_start_time_id.to(device)      # [B_local]
#     pe_local = pos_end_time_id.to(device)        # [B_local]

#     # ====== 全局 gather ======
#     with torch.no_grad():
#         # queries
#         q_all  = gather_embeddings(q_local.detach())         # [B_global, D]
#         qs_all = gather_embeddings(qs_local)                 # [B_global]
#         qe_all = gather_embeddings(qe_local)                 # [B_global]

#         # docs
#         p_all  = gather_embeddings(p_local.detach())         # [B_global, D]
#         ps_all = gather_embeddings(ps_local)                 # [B_global]
#         pe_all = gather_embeddings(pe_local)                 # [B_global]

#         # neg
#         if use_extra_negs:
#             n_all = gather_embeddings(n_local.detach())      # [N_global, D]
#         else:
#             n_all = None

#     # ====== 本 rank 的 point-query 作为 anchor ======
#     is_point_local = (qs_local == qe_local)                  # [B_local]
#     if not is_point_local.any():
#         # 本 rank 没有 point-query
#         return q_emb.new_tensor(0.0)

#     q_point = q_local[is_point_local]                        # [B_point, D]
#     t_point = qs_local[is_point_local]                       # [B_point]
#     B_point = q_point.size(0)

#     # ====== 全局 window-query 作为 gallery 的一部分 ======
#     is_point_all   = (qs_all == qe_all)                      # [B_global]
#     is_window_all  = ~is_point_all

#     q_window_all   = q_all[is_window_all]                    # [B_win_g, D]
#     qs_window_all  = qs_all[is_window_all]                   # [B_win_g]
#     qe_window_all  = qe_all[is_window_all]                   # [B_win_g]
#     B_win_g        = q_window_all.size(0)

#     # 全局 doc
#     p_all  = p_all                                           # [B_doc_g, D]
#     ps_all = ps_all                                          # [B_doc_g]
#     pe_all = pe_all                                          # [B_doc_g]
#     B_doc_g = p_all.size(0)

#     # ====== 构造统一全局 gallery: [Q_window_all, P_all, neg_all] ======
#     gallery_parts = []
#     if B_win_g > 0:
#         gallery_parts.append(q_window_all)
#     gallery_parts.append(p_all)
#     if use_extra_negs and n_all is not None:
#         gallery_parts.append(n_all)

#     gallery_all = torch.cat(gallery_parts, dim=0)            # [G, D]
#     G = gallery_all.size(0)

#     # ====== 为本 rank 的 point-anchor 构造 positive mask ======
#     pos_mask = torch.zeros(B_point, G, dtype=torch.bool, device=device)

#     col = 0
#     # --- Q_window_all 段 ---
#     if B_win_g > 0:
#         t_point_ = t_point[:, None]                          # [B_point, 1]
#         qs_wg_   = qs_window_all[None, :]                    # [1, B_win_g]
#         qe_wg_   = qe_window_all[None, :]                    # [1, B_win_g]

#         pos_win = (qs_wg_ <= t_point_) & (t_point_ <= qe_wg_)    # [B_point, B_win_g]
#         pos_mask[:, col : col + B_win_g] = pos_win
#         col += B_win_g

#     # --- P_all 段 ---
#     t_point_ = t_point[:, None]                              # [B_point, 1]
#     ps_g_    = ps_all[None, :]                               # [1, B_doc_g]
#     pe_g_    = pe_all[None, :]                               # [1, B_doc_g]

#     pos_doc = (ps_g_ <= t_point_) & (t_point_ <= pe_g_)      # [B_point, B_doc_g]
#     pos_mask[:, col : col + B_doc_g] = pos_doc
#     col += B_doc_g

#     # neg_all 段天然全 False

#     # ====== 丢掉没有任何正例的 anchor ======
#     row_has_pos = pos_mask.any(dim=1)                        # [B_point]
#     if not row_has_pos.any():
#         return q_emb.new_tensor(0.0)

#     q_point_valid  = q_point[row_has_pos]                    # [B_valid, D]
#     pos_mask_valid = pos_mask[row_has_pos]                   # [B_valid, G]

#     # ====== 单向 InfoNCE: q_point_valid -> gallery_all ======
#     logits = (q_point_valid @ gallery_all.t()) / temperature # [B_valid, G]

#     log_prob      = logits - torch.logsumexp(logits, dim=1, keepdim=True)
#     pos_log_prob  = log_prob.masked_fill(~pos_mask_valid, float("-inf"))
#     loss_per_row  = -torch.logsumexp(pos_log_prob, dim=1)
#     loss          = loss_per_row.mean()

#     return loss


def point_singlepos_loss_global_inbatch(
    q_emb: torch.Tensor,
    p_emb: torch.Tensor,
    neg_emb: torch.Tensor = None,                # 参与为负例
    temperature: float = 0.05,
    query_start_time_id: torch.Tensor = None,
    query_end_time_id:   torch.Tensor = None,
    pos_start_time_id:   torch.Tensor = None,
    pos_end_time_id:     torch.Tensor = None,
    symmetric: bool = True,                      # 仍然忽略，只做 q_point -> gallery
    query_is_pure_no_event : torch.Tensor = None,  # 新增：标记某些 query 是否为 pure no_event
) -> torch.Tensor:
    """
    DDP 版本的 point-type 时间辅助 InfoNCE（单向）：

    - anchor = 当前 rank 的所有 point-query（qs == qe）
    - gallery = [全局 window-query, 全局 doc, 全局 neg]
        * window-query 全部视为负例（语义不同的 query 模板，作为 hard negative）
        * doc 中时间区间覆盖该 point 的样本视为正例
        * neg 全部为负例
    """

    # ==== 非 DDP：退回单卡版 ====
    if not dist.is_initialized():
        return point_singlepos_loss_inbatch(
            q_emb=q_emb,
            p_emb=p_emb,
            neg_emb=neg_emb,
            temperature=temperature,
            query_start_time_id=query_start_time_id,
            query_end_time_id=query_end_time_id,
            pos_start_time_id=pos_start_time_id,
            pos_end_time_id=pos_end_time_id,
            symmetric=symmetric,
        )

    device = q_emb.device

    # 没有时间信息，直接不加这条 loss
    if (
        query_start_time_id is None or query_end_time_id is None
        or pos_start_time_id   is None or pos_end_time_id   is None
    ):
        return q_emb.new_tensor(0.0)

    # ====== 本地归一化 ======
    q_local = F.normalize(q_emb, p=2, dim=-1)    # [B_local, D]
    p_local = F.normalize(p_emb, p=2, dim=-1)    # [B_local, D]

    use_extra_negs = (neg_emb is not None) and (neg_emb.numel() > 0)
    if use_extra_negs:
        n_local = F.normalize(neg_emb, p=2, dim=-1)  # [B_local, D]
    else:
        n_local = None

    # 本地时间戳
    qs_local = query_start_time_id.to(device)    # [B_local]
    qe_local = query_end_time_id.to(device)      # [B_local]
    ps_local = pos_start_time_id.to(device)      # [B_local]
    pe_local = pos_end_time_id.to(device)        # [B_local]

    # ====== 全局 gather ======
    with torch.no_grad():
        # queries（用于构造 window 部分）
        q_all  = gather_embeddings(q_local.detach())         # [B_global, D]
        qs_all = gather_embeddings(qs_local)                 # [B_global]
        qe_all = gather_embeddings(qe_local)                 # [B_global]

        # docs
        p_all  = gather_embeddings(p_local.detach())         # [B_global, D]
        ps_all = gather_embeddings(ps_local)                 # [B_global]
        pe_all = gather_embeddings(pe_local)                 # [B_global]

        # neg
        if use_extra_negs:
            n_all = gather_embeddings(n_local.detach())      # [N_global, D]
        else:
            n_all = None

    # ====== 当前 rank 的 point-query 作为 anchor ======
    # 注意：这里用的是 query 的时间（qs_local, qe_local），完全没有拿 doc 的时间来当 query
    is_point_local = (qs_local == qe_local)                  # [B_local]

    if query_is_pure_no_event is not None:
        # ★ 剔掉空月 query：只保留 "point 且不是 pure no_event" 的 anchor
        is_point_local = is_point_local & (~query_is_pure_no_event)

    if not is_point_local.any():
        return q_emb.new_tensor(0.0)

    q_point = q_local[is_point_local]                        # [B_point, D]
    t_point = qs_local[is_point_local]                       # [B_point]  # 单点时间 id
    B_point = q_point.size(0)

    # ====== 全局 window-query 作为 gallery 的一部分（纯负例） ======
    is_point_all   = (qs_all == qe_all)                      # [B_global]
    is_window_all  = ~is_point_all

    q_window_all   = q_all[is_window_all]                    # [B_win_g, D]
    qs_window_all  = qs_all[is_window_all]                   # [B_win_g]
    qe_window_all  = qe_all[is_window_all]                   # [B_win_g]
    B_win_g        = q_window_all.size(0)

    # 全局 doc
    p_all  = p_all                                           # [B_doc_g, D]
    ps_all = ps_all                                          # [B_doc_g]
    pe_all = pe_all                                          # [B_doc_g]
    B_doc_g = p_all.size(0)

    # ====== 构造统一全局 gallery: [Q_window_all, P_all, neg_all] ======
    gallery_parts = []
    if B_win_g > 0:
        gallery_parts.append(q_window_all)
    gallery_parts.append(p_all)
    if use_extra_negs and n_all is not None:
        gallery_parts.append(n_all)

    gallery_all = torch.cat(gallery_parts, dim=0)            # [G, D]
    G = gallery_all.size(0)

    # ====== 为本 rank 的 point-anchor 构造 positive mask ======
    pos_mask = torch.zeros(B_point, G, dtype=torch.bool, device=device)

    col = 0
    # --- Q_window_all 段：全部视为负例，只移动指针 ---
    if B_win_g > 0:
        # 不再把 window-query 当作正例
        col += B_win_g

    # --- P_all 段：时间覆盖 t_point 的 doc 是正例 ---
    # 为了稳妥，跟主 loss 一样用 min/max，防止 start/end 颠倒
    p_min_all = torch.minimum(ps_all, pe_all)                # [B_doc_g]
    p_max_all = torch.maximum(ps_all, pe_all)                # [B_doc_g]

    t_point_  = t_point[:, None]                             # [B_point, 1]
    p_min_g_  = p_min_all[None, :]                           # [1, B_doc_g]
    p_max_g_  = p_max_all[None, :]                           # [1, B_doc_g]

    # 单点时间落在 doc 区间内，就视为正例
    pos_doc = (p_min_g_ <= t_point_) & (t_point_ <= p_max_g_)  # [B_point, B_doc_g]
    pos_mask[:, col : col + B_doc_g] = pos_doc
    col += B_doc_g

    # neg_all 段天然全 False，剩下的列保持 0 即可

    # ====== 丢掉没有任何正例的 anchor（只保留有 doc 覆盖的 point） ======
    row_has_pos = pos_mask.any(dim=1)                        # [B_point]
    if not row_has_pos.any():
        return q_emb.new_tensor(0.0)

    q_point_valid  = q_point[row_has_pos]                    # [B_valid, D]
    pos_mask_valid = pos_mask[row_has_pos]                   # [B_valid, G]

    # ====== 单向 InfoNCE: q_point_valid -> gallery_all ======
    logits = (q_point_valid @ gallery_all.t()) / temperature # [B_valid, G]

    log_prob      = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_log_prob  = log_prob.masked_fill(~pos_mask_valid, float("-inf"))
    loss_per_row  = -torch.logsumexp(pos_log_prob, dim=1)
    loss          = loss_per_row.mean()

    return loss


