import torch
import torch.nn as nn
import torch.nn.functional as F
from .models_temporal_dual import BertDualEncoder
# ============ 时间多任务相关损失函数 ============

def time_ce_loss_with_neighbor_smoothing(
    logits: torch.Tensor,     # (B, C)
    targets: torch.Tensor,    # (B,) 0-based
    num_classes: int,
    use_neighbor_smoothing: bool = False,
    smoothing_eps: float = 0.1,
):
    """
    如果 use_neighbor_smoothing=False 或 eps<=0，就退化成标准 CrossEntropy。
    否则：给 target 类分配 (1 - eps)，左右相邻类共享 eps。
    """
    if (not use_neighbor_smoothing) or smoothing_eps <= 0.0 or num_classes <= 1:
        return F.cross_entropy(logits, targets)

    with torch.no_grad():
        B, C = logits.size()
        smooth_targets = torch.zeros_like(logits)  # (B, C)
        for i in range(B):
            c = int(targets[i].item())
            # 中心类
            smooth_targets[i, c] = 1.0 - smoothing_eps

            # 邻居：c-1, c+1
            neighbors = []
            if c - 1 >= 0:
                neighbors.append(c - 1)
            if c + 1 < num_classes:
                neighbors.append(c + 1)

            if len(neighbors) > 0:
                share = smoothing_eps / len(neighbors)
                for n in neighbors:
                    smooth_targets[i, n] += share

    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
    return loss


def symmetric_kl_from_logits(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """
    对称 KL(p || q) + KL(q || p)，其中 p, q 都是 logits (B, C)。
    """
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    p = log_p.exp()
    q = log_q.exp()

    kl_pq = F.kl_div(log_p, q, reduction="batchmean")
    kl_qp = F.kl_div(log_q, p, reduction="batchmean")
    return kl_pq + kl_qp

def build_time_scalar_from_labels(
    gong_label: torch.Tensor,   # (B,) 0-based
    year_label: torch.Tensor,   # (B,) 0-based
    month_label: torch.Tensor,  # (B,) 0-based
    num_gong: int,
    num_year: int,
    num_month: int,
):
    """
    把 (gong, year, month) 合成一个连续标量 t 并归一到 [0, 1]。
    这里假设类索引本身已经保留了原来的时间顺序（你的 sort_key 就是）。
    """
    g = gong_label.float()
    y = year_label.float()
    m = month_label.float()

    # 先映射到一个全局 index
    # 每个 gong 下有 num_year * num_month 个位置
    t = g * (num_year * num_month) + y * num_month + m  # (B,)

    max_t = float(num_gong * num_year * num_month - 1)
    if max_t > 0:
        t = t / max_t  # 归一化到 [0, 1]
    return t


def compute_time_losses(
    model: BertDualEncoder,
    q_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    gong_logits_q: torch.Tensor,
    year_logits_q: torch.Tensor,
    month_logits_q: torch.Tensor,
    gong_logits_p: torch.Tensor,
    year_logits_p: torch.Tensor,
    month_logits_p: torch.Tensor,
    batch_t: dict,
    use_neighbor_smoothing: bool = False,
    smoothing_eps: float = 0.1,
    use_time_align: bool = False,
    use_time_regression: bool = False,
    time_align_weight: float = 1.0,
    time_reg_weight: float = 1.0,
):
    """
    返回:
      time_loss_total, time_loss_doc_ce, time_loss_align, time_loss_reg
    如果没开时间头，四个都是 0。
    """
    device = q_emb.device

    if not model.use_time_heads:
        zero = torch.tensor(0.0, device=device)
        return zero, zero, zero, zero

    # # --------- 1) 时间分类 logits ----------
    # gong_logits_q, year_logits_q, month_logits_q = model.time_logits(q_emb)      # (B, G/Y/M)
    # gong_logits_p, year_logits_p, month_logits_p = model.time_logits(pos_emb)

    # --------- 2) doc 侧 CE（带可选邻近 smoothing） ----------
    pos_gong = batch_t["pos_gong_label"]    # 0-based
    pos_year = batch_t["pos_year_label"]
    pos_month = batch_t["pos_month_label"]

    num_gong = model.num_gong
    num_year = model.num_year
    num_month = model.num_month

    loss_gong_doc = time_ce_loss_with_neighbor_smoothing(
        gong_logits_p, pos_gong, num_gong,
        use_neighbor_smoothing=use_neighbor_smoothing,
        smoothing_eps=smoothing_eps,
    )
    loss_year_doc = time_ce_loss_with_neighbor_smoothing(
        year_logits_p, pos_year, num_year,
        use_neighbor_smoothing=use_neighbor_smoothing,
        smoothing_eps=smoothing_eps,
    )
    loss_month_doc = time_ce_loss_with_neighbor_smoothing(
        month_logits_p, pos_month, num_month,
        use_neighbor_smoothing=use_neighbor_smoothing,
        smoothing_eps=smoothing_eps,
    )

    time_loss_doc_ce = (loss_gong_doc + loss_year_doc + loss_month_doc) / 3.0

    # --------- 3) query ↔ doc 时间分布对齐 (KL) ----------
    time_loss_align = torch.tensor(0.0, device=device)
    if use_time_align:
        # symmetric KL on each head
        kl_gong = symmetric_kl_from_logits(gong_logits_q, gong_logits_p)
        kl_year = symmetric_kl_from_logits(year_logits_q, year_logits_p)
        kl_month = symmetric_kl_from_logits(month_logits_q, month_logits_p)
        time_loss_align = (kl_gong + kl_year + kl_month) / 3.0
        time_loss_align = time_align_weight * time_loss_align

    # --------- 4) 连续时间回归 ----------
    time_loss_reg = torch.tensor(0.0, device=device)
    if use_time_regression and model.use_time_regression:
        # 构造 query / doc 的连续时间标量
        query_gong = batch_t["query_gong_label"]
        query_year = batch_t["query_year_label"]
        query_month = batch_t["query_month_label"]

        t_doc = build_time_scalar_from_labels(
            pos_gong, pos_year, pos_month,
            num_gong=num_gong, num_year=num_year, num_month=num_month,
        )  # (B,)
        t_query = build_time_scalar_from_labels(
            query_gong, query_year, query_month,
            num_gong=num_gong, num_year=num_year, num_month=num_month,
        )  # (B,)

        # 预测值
        t_doc_hat = model.time_scalar(pos_emb)   # (B,)
        t_query_hat = model.time_scalar(q_emb)   # (B,)

        # SmoothL1 比 MSE 更稳定一点
        loss_reg_doc = F.smooth_l1_loss(t_doc_hat, t_doc)
        loss_reg_query = F.smooth_l1_loss(t_query_hat, t_query)
        time_loss_reg = time_reg_weight * 0.5 * (loss_reg_doc + loss_reg_query)

    # --------- 5) 汇总 ----------
    time_loss_total = time_loss_doc_ce + time_loss_align + time_loss_reg
    return time_loss_total, time_loss_doc_ce, time_loss_align, time_loss_reg

