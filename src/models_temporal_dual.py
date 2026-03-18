import torch
import math
import torch.nn as nn
import random
from torch import Tensor


from transformers import AutoModel
import torch.nn.functional as F

def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    """
    Qwen3-Embedding 官方推荐的 pooling：
    - 如果是 left padding，则取最后一个 token；
    - 否则根据 attention_mask 取有效序列的最后一个 token。
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

def wrap_query_with_instruction(task_description: str, query: str) -> str:
    """
    Qwen3-Embedding 推荐格式：
    Instruct: ...
    Query:...
    """
    return f"Instruct: {task_description}\nQuery:{query}"


class BertDualEncoder(nn.Module):
    """
    双塔检索 + 时间建模：

    - 一个 BERT 编码 query / doc（共享权重）
    - pooling 用 [CLS]，然后 L2 norm

    - 时间头：
        · 三个分类头：gong / year / month
        · 只用于时间 CE / label smoothing / 预测分布，不做回归

    - 绝对时间上下文（可选，use_time_context_pred=True）：
        · 用时间头的预测分布 (gong/year/month)
        · 对各自的 Fourier 编码表做期望 → 时间上下文向量
        · 三块拼接 + Linear 映射回 hidden_size
        · 残差加回文本 emb，带一个可学习 gate

    - 相对时间 Fourier bias（可选，use_time_rel_bias=True）：
        · 用时间头预测分布，得到连续时间标量 t ∈ [0,1]
        · Δt = t_doc - t_query（保留方向）
        · Fourier(Δt) → 小 MLP → 标量 bias
        · bias = alpha * bias，加在 in-batch 对比学习 logits 上
    """

    def __init__(
        self,
        model_name_or_path: str,
        pooling: str = "cls",
        normalize: bool = True,
        num_gong: int = None,
        num_year: int = None,
        num_month: int = None,
        # 绝对时间：基于预测分布的 Fourier 上下文
        use_time_context_pred: bool = False,
        time_emb_dim: int = 64,
        # 相对时间：基于 Δt 的 Fourier bias
        use_time_rel_bias: bool = False,
        time_rel_dim: int = 32,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.pooling = pooling
        self.normalize = normalize

        hidden_size = self.encoder.config.hidden_size

        # ---------- 时间多任务：时间分类头 ----------
        self.num_gong = num_gong
        self.num_year = num_year
        self.num_month = num_month

        self.use_time_heads = (
            num_gong is not None and num_year is not None and num_month is not None
        )

        if self.use_time_heads:
            # 注意：num_* 已经是类别数（0-based label 对应 [0, num_* - 1]）
            self.time_head_gong = nn.Linear(hidden_size, num_gong)
            self.time_head_year = nn.Linear(hidden_size, num_year)
            self.time_head_month = nn.Linear(hidden_size, num_month)

        # ---------- 绝对时间上下文（预测分布 + Fourier） ----------
        self.use_time_context_pred = use_time_context_pred
        self.time_emb_dim = time_emb_dim

        if self.use_time_heads and self.use_time_context_pred:
            # 三块 time context 拼接后投回 hidden_size
            self.time_ctx_proj = nn.Linear(time_emb_dim * 3, hidden_size)
            # 可学习 gate，控制时间上下文注入强度；模型可以主动把它调小趋近 0
            self.time_ctx_gate = nn.Parameter(torch.tensor(1.0))

        # ---------- 基于 Δt 的 relative-time Fourier bias（logits 上的加性偏置） ----------
        self.use_time_rel_bias = use_time_rel_bias
        self.time_rel_dim = time_rel_dim

        if self.use_time_heads and self.use_time_rel_bias:
            # Fourier 特征 -> 小 MLP -> 标量 bias
            self.time_rel_mlp = nn.Sequential(
                nn.Linear(time_rel_dim, time_rel_dim),
                nn.ReLU(),
                nn.Linear(time_rel_dim, 1),
            )
            # 可学习缩放系数，初始为 0：最坏情况退回没有时间 bias
            self.time_rel_alpha = nn.Parameter(torch.tensor(0.0))

    # ----------------- encode / forward -----------------

    def encode_base(self, input_ids, attention_mask):
        """
        只做文本编码，不加任何时间上下文，用于：
        - 计算 time CE / KL loss
        - 为后续的时间上下文 / 相对时间核生成打底表示
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state

        if self.pooling == "cls":
            emb = last_hidden[:, 0]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            summed = (last_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            emb = summed / denom
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return emb  # (B, H)

    def add_time_context_from_pred(self, emb, tau: float = 1.0):
        """
        emb: (B, H) 文本基础向量
        返回: (B, H) 融合了“预测时间分布”的上下文后的检索向量
        """
        # 如果没开时间头或者没开基于预测分布的时间上下文，直接 norm + 返回
        if not (self.use_time_heads and self.use_time_context_pred):
            out = emb
            if self.normalize:
                out = F.normalize(out, p=2, dim=1)
            return out

        device = emb.device

        # 1) 用基础 emb 做时间预测
        gong_logits, year_logits, month_logits = self.time_logits_from_emb(emb)

        # 2) softmax 得到预测分布（可选 temperature）
        gong_p = F.softmax(gong_logits / tau, dim=-1)   # (B, G)
        year_p = F.softmax(year_logits / tau, dim=-1)   # (B, Y)
        month_p = F.softmax(month_logits / tau, dim=-1) # (B, M)

        # 3) 构造 gong/year/month 对应的 Fourier 编码表
        gong_table = self._build_fourier_table(self.num_gong,  self.time_emb_dim, device)  # (G, D)
        year_table = self._build_fourier_table(self.num_year,  self.time_emb_dim, device)  # (Y, D)
        month_table = self._build_fourier_table(self.num_month, self.time_emb_dim, device) # (M, D)

        # 4) 预测分布对编码表做期望（mixture-of-Fourier-features）
        gong_ctx = gong_p @ gong_table        # (B, D)
        year_ctx = year_p @ year_table        # (B, D)
        month_ctx = month_p @ month_table     # (B, D)

        # 5) 拼接 + 映射到 hidden_size
        time_ctx = torch.cat([gong_ctx, year_ctx, month_ctx], dim=-1)  # (B, 3D)
        time_ctx = self.time_ctx_proj(time_ctx)                        # (B, H)

        # 6) 残差融合 + gate 控制强度
        out = emb + self.time_ctx_gate * time_ctx

        if self.normalize:
            out = F.normalize(out, p=2, dim=1)
        return out

    def encode(self, input_ids, attention_mask):
        base = self.encode_base(input_ids, attention_mask)
        out = self.add_time_context_from_pred(base)
        return out

    def forward(self, batch):
        """
        训练用前向：

        返回：
          - q_emb, pos_emb, neg_emb：检索向量（已加绝对时间上下文 & L2）
          - q_base, pos_base：纯文本 CLS，用于 time-loss / time-rel-bias
          - [可选] time_rel_bias: (B, B) 时间相对性 bias 矩阵
        """
        q_base = self.encode_base(batch["query_input_ids"], batch["query_attention_mask"])
        pos_base = self.encode_base(batch["pos_input_ids"], batch["pos_attention_mask"])
        neg_base = self.encode_base(batch["neg_input_ids"], batch["neg_attention_mask"])

        # 检索向量：根据配置决定是否加时间上下文
        q_emb = self.add_time_context_from_pred(q_base)
        pos_emb = self.add_time_context_from_pred(pos_base)
        neg_emb = self.add_time_context_from_pred(neg_base)

        # ---------- 新增：时间分类 head 的 logits（完全在 forward 内部） ----------
        gong_logits_q = year_logits_q = month_logits_q = None
        gong_logits_p = year_logits_p = month_logits_p = None
        if self.use_time_heads:
            gong_logits_q, year_logits_q, month_logits_q = self.time_logits(q_base)
            gong_logits_p, year_logits_p, month_logits_p = self.time_logits(pos_base)
        
        # ---------- 相对时间 bias（保持你原来的写法） ----------
        time_rel_bias = None
        if self.use_time_heads and self.use_time_rel_bias:
            time_rel_bias = self.compute_time_rel_bias(q_base, pos_base, tau=1.0)

        # 统一返回：emb + base CLS + 时间 logits + 时间 bias
        return (
            q_emb, pos_emb, neg_emb,
            q_base, pos_base,
            gong_logits_q, year_logits_q, month_logits_q,
            gong_logits_p, year_logits_p, month_logits_p,
            time_rel_bias,
        )

        # if self.use_time_heads and self.use_time_rel_bias:
        #     time_rel_bias = self.compute_time_rel_bias(q_base, pos_base, tau=1.0)
        #     return q_emb, pos_emb, neg_emb, q_base, pos_base, time_rel_bias
        # else:
        #     return q_emb, pos_emb, neg_emb, q_base, pos_base

    # ---------- 时间头 helper ----------

    def time_logits(self, emb):
        """
        输入: emb (B, H)
        输出: 三个 logits (gong/year/month)，如果未启用时间头则返回 None
        """
        if not self.use_time_heads:
            return None, None, None
        gong_logits = self.time_head_gong(emb)
        year_logits = self.time_head_year(emb)
        month_logits = self.time_head_month(emb)
        return gong_logits, year_logits, month_logits

    def time_logits_from_emb(self, emb):
        """
        和 time_logits 类似，只是直接吃 (B, H) emb，
        避免一定要经过 encode() 那一层。
        """
        if not self.use_time_heads:
            raise RuntimeError("time heads not enabled")
        gong_logits = self.time_head_gong(emb)
        year_logits = self.time_head_year(emb)
        month_logits = self.time_head_month(emb)
        return gong_logits, year_logits, month_logits

    # ----------------- 绝对时间：Fourier 表（按 label index） -----------------

    def _build_fourier_table(self, num_classes: int, emb_dim: int, device: torch.device):
        """
        为 0..num_classes-1 构造一个 (num_classes, emb_dim) 的 Fourier / sinusoidal 编码表。
        每个类 c 先归一成 [0,1] 上的 position，然后做一组从低频到高频的 sin/cos。
        """
        positions = torch.arange(num_classes, device=device).float()  # (C,)

        if num_classes > 1:
            positions = positions / (num_classes - 1)  # 映射到 [0,1]
        else:
            positions = torch.zeros_like(positions)

        half_dim = emb_dim // 2
        if half_dim == 0:
            # 极端情况：维度太小，直接返回一个列向量
            return positions.unsqueeze(1)

        # 类似 Transformer 的 1 / 10000^{2k/D} 频率布置，让高维对应更高频
        div_term = torch.exp(
            torch.arange(half_dim, device=device).float()
            * (-math.log(10000.0) / max(1, half_dim - 1))
        )  # (half_dim,)

        # (C, half_dim)
        angles = positions.unsqueeze(1) * div_term.unsqueeze(0)

        sin = torch.sin(angles)
        cos = torch.cos(angles)
        table = torch.cat([sin, cos], dim=-1)  # (C, 2*half_dim)

        # 根据 emb_dim 截断或 padding
        if table.size(1) < emb_dim:
            pad = emb_dim - table.size(1)
            pad_tensor = torch.zeros(num_classes, pad, device=device)
            table = torch.cat([table, pad_tensor], dim=-1)
        elif table.size(1) > emb_dim:
            table = table[:, :emb_dim]

        return table  # (C, emb_dim)

    # ----------------- 相对时间：连续标量 t & Fourier 特征 -----------------

    def _build_time_scalar_from_probs(self, gong_p, year_p, month_p):
        """
        gong_p:  (B, G)
        year_p:  (B, Y)
        month_p: (B, M)
        返回 t: (B,), 归一化到 [0, 1]
        """
        device = gong_p.device

        g_idx = torch.arange(self.num_gong,  device=device).float()  # (G,)
        y_idx = torch.arange(self.num_year,  device=device).float()  # (Y,)
        m_idx = torch.arange(self.num_month, device=device).float()  # (M,)

        g = (gong_p  * g_idx.unsqueeze(0)).sum(-1)  # (B,)
        y = (year_p  * y_idx.unsqueeze(0)).sum(-1)  # (B,)
        m = (month_p * m_idx.unsqueeze(0)).sum(-1)  # (B,)

        # 展平成一个全局时间 index
        t = g * (self.num_year * self.num_month) + y * self.num_month + m
        max_t = float(self.num_gong * self.num_year * self.num_month - 1)
        if max_t > 0:
            t = t / max_t
        return t  # (B,), in [0, 1]

    def _build_rel_fourier_features(self, delta_t):
        """
        delta_t: (Bq, Bd)，Δt 已在 [-1, 1] 之类的范围
        返回: (Bq, Bd, time_rel_dim) 的 Fourier 特征
        """
        D = self.time_rel_dim
        K = max(1, D // 2)

        # (Bq, Bd, 1)
        delta = delta_t.unsqueeze(-1)

        # log-scale 频率，类似 pos-encoding
        freqs = torch.exp(
            torch.linspace(0, 1, K, device=delta_t.device) * (-math.log(10000.0))
        )  # (K,)

        # (Bq, Bd, K)
        angles = delta * freqs

        sin_feat = torch.sin(angles)
        cos_feat = torch.cos(angles)
        feat = torch.cat([sin_feat, cos_feat], dim=-1)  # (Bq, Bd, 2K)

        # 对齐到 D 维
        if feat.size(-1) > D:
            feat = feat[..., :D]
        elif feat.size(-1) < D:
            pad = feat.new_zeros(*feat.shape[:-1], D - feat.size(-1))
            feat = torch.cat([feat, pad], dim=-1)
        return feat  # (Bq, Bd, D)

    def compute_time_rel_bias(self, q_base, p_base, tau: float = 1.0):
        """
        q_base, p_base: (B, H)，都是 encode_base 的输出（未加时间上下文的 CLS）
        返回: (B, B) 的时间相对性 bias 矩阵，或者 None
        """
        if not (self.use_time_heads and self.use_time_rel_bias):
            return None

        # 1) 时间分布（query / doc 各自）
        g_log_q, y_log_q, m_log_q = self.time_logits_from_emb(q_base)
        g_log_p, y_log_p, m_log_p = self.time_logits_from_emb(p_base)

        g_p_q = F.softmax(g_log_q / tau, dim=-1)
        y_p_q = F.softmax(y_log_q / tau, dim=-1)
        m_p_q = F.softmax(m_log_q / tau, dim=-1)

        g_p_p = F.softmax(g_log_p / tau, dim=-1)
        y_p_p = F.softmax(y_log_p / tau, dim=-1)
        m_p_p = F.softmax(m_log_p / tau, dim=-1)

        # 2) 得到连续时间标量 t_q, t_p in [0, 1]
        t_q = self._build_time_scalar_from_probs(g_p_q, y_p_q, m_p_q)  # (B,)
        t_p = self._build_time_scalar_from_probs(g_p_p, y_p_p, m_p_p)  # (B,)

        # 3) 计算 Δt，广播成 (Bq, Bd)
        delta_t = t_p.unsqueeze(0) - t_q.unsqueeze(1)  # (Bq, Bd)
        delta_t = delta_t.clamp(min=-1.0, max=1.0)

        # 4) Δt -> Fourier 特征 -> 标量 bias
        rel_feat = self._build_rel_fourier_features(delta_t)       # (Bq, Bd, D_rel)
        bias = self.time_rel_mlp(rel_feat).squeeze(-1)             # (Bq, Bd)

        # 5) 可学习 alpha 控制强度
        return self.time_rel_alpha * bias  # (Bq, Bd)


class QwenTimeDualEncoder(nn.Module):
    """
    用 Qwen3-Embedding 做 backbone 的时间感知双塔：
    - backbone: Qwen3-Embedding (0.6B / 4B)
    - pooling: last_token_pool（官方推荐）
    - 时间模块完全沿用 BERT 版本：
      * 时间分类头：gong / year / month
      * 绝对时间 Fourier 上下文（基于预测分布）
      * 相对时间 Fourier kernel bias（Δt -> Fourier -> MLP -> bias）
    """

    def __init__(
        self,
        model_name_or_path: str,
        normalize: bool = True,
        num_gong: int = None,
        num_year: int = None,
        num_month: int = None,
        # 绝对时间上下文（基于预测分布的 Fourier）
        use_time_context_pred: bool = False,
        time_emb_dim: int = 64,
        # 相对时间 Fourier bias
        use_time_rel_bias: bool = False,
        time_rel_dim: int = 32,
    ):
        super().__init__()
        # ★ Qwen 需要 trust_remote_code=True
        self.encoder = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        self.normalize = normalize

        hidden_size = self.encoder.config.hidden_size

        # ---------- 时间多任务：时间分类头 ----------
        self.num_gong = num_gong
        self.num_year = num_year
        self.num_month = num_month

        self.use_time_heads = (
            num_gong is not None and num_year is not None and num_month is not None
        )

        if self.use_time_heads:
            self.time_head_gong = nn.Linear(hidden_size, num_gong)
            self.time_head_year = nn.Linear(hidden_size, num_year)
            self.time_head_month = nn.Linear(hidden_size, num_month)

        # ---------- 绝对时间上下文（Fourier + 预测分布） ----------
        self.use_time_context_pred = use_time_context_pred
        self.time_emb_dim = time_emb_dim
        print("use_time_heads:", self.use_time_heads)

        if self.use_time_heads and self.use_time_context_pred:
            # gong/year/month 分别各自一套 Fourier 表（在 forward 动态构造）
            # 然后 concat -> proj -> gate
            self.time_ctx_proj = nn.Linear(time_emb_dim * 3, hidden_size)
            self.time_ctx_gate = nn.Parameter(torch.tensor(1.0))

        # ---------- 相对时间 Fourier kernel bias ----------
        self.use_time_rel_bias = use_time_rel_bias
        self.time_rel_dim = time_rel_dim

        if self.use_time_heads and self.use_time_rel_bias:
            self.time_rel_mlp = nn.Sequential(
                nn.Linear(time_rel_dim, time_rel_dim),
                nn.ReLU(),
                nn.Linear(time_rel_dim, 1),
            )
            self.time_rel_alpha = nn.Parameter(torch.tensor(0.0))  # 全局缩放

    # ===== 工具：从预测分布得到连续时间标量 t \in [0, 1] =====
    def _build_time_scalar_from_probs(self, gong_p, year_p, month_p):
        device = gong_p.device

        g_idx = torch.arange(self.num_gong,  device=device).float()  # (G,)
        y_idx = torch.arange(self.num_year,  device=device).float()  # (Y,)
        m_idx = torch.arange(self.num_month, device=device).float()  # (M,)

        g = (gong_p * g_idx.unsqueeze(0)).sum(-1)   # (B,)
        y = (year_p * y_idx.unsqueeze(0)).sum(-1)   # (B,)
        m = (month_p * m_idx.unsqueeze(0)).sum(-1)  # (B,)

        t = g * (self.num_year * self.num_month) + y * self.num_month + m
        max_t = float(self.num_gong * self.num_year * self.num_month - 1)
        if max_t > 0:
            t = t / max_t
        return t  # (B,), in [0, 1]

    # ===== 相对时间 Fourier 特征 =====
    def _build_rel_fourier_features(self, delta_t):
        """
        delta_t: (Bq, Bd)，Δt 已在 [-1, 1]
        返回: (Bq, Bd, time_rel_dim)
        """
        D = self.time_rel_dim
        K = max(1, D // 2)

        delta = delta_t.unsqueeze(-1)  # (Bq, Bd, 1)

        freqs = torch.exp(
            torch.linspace(0, 1, K, device=delta_t.device) * (-math.log(10000.0))
        )  # (K,)

        angles = delta * freqs  # (Bq, Bd, K)

        sin_feat = torch.sin(angles)
        cos_feat = torch.cos(angles)
        feat = torch.cat([sin_feat, cos_feat], dim=-1)  # (Bq, Bd, 2K)

        if feat.size(-1) > D:
            feat = feat[..., :D]
        elif feat.size(-1) < D:
            pad = feat.new_zeros(*feat.shape[:-1], D - feat.size(-1))
            feat = torch.cat([feat, pad], dim=-1)
        return feat  # (Bq, Bd, D)

    def compute_time_rel_bias(self, q_base, p_base, tau: float = 1.0):
        """
        返回: (Bq, Bd) 时间相对性 bias 矩阵
        """
        if not (self.use_time_heads and self.use_time_rel_bias):
            return None

        # 1) 时间分布（query / doc）
        g_log_q, y_log_q, m_log_q = self.time_logits_from_emb(q_base)
        g_log_p, y_log_p, m_log_p = self.time_logits_from_emb(p_base)

        g_p_q = F.softmax(g_log_q / tau, dim=-1)
        y_p_q = F.softmax(y_log_q / tau, dim=-1)
        m_p_q = F.softmax(m_log_q / tau, dim=-1)

        g_p_p = F.softmax(g_log_p / tau, dim=-1)
        y_p_p = F.softmax(y_log_p / tau, dim=-1)
        m_p_p = F.softmax(m_log_p / tau, dim=-1)

        # 2) 连续时间标量 t_q, t_p
        t_q = self._build_time_scalar_from_probs(g_p_q, y_p_q, m_p_q)  # (B,)
        t_p = self._build_time_scalar_from_probs(g_p_p, y_p_p, m_p_p)  # (B,)

        # 3) Δt ∈ [-1, 1]
        delta_t = t_p.unsqueeze(0) - t_q.unsqueeze(1)  # (Bq, Bd)
        delta_t = delta_t.clamp(min=-1.0, max=1.0)

        # 4) Fourier 特征 → MLP → 标量 bias
        rel_feat = self._build_rel_fourier_features(delta_t)  # (Bq, Bd, D_rel)
        bias = self.time_rel_mlp(rel_feat).squeeze(-1)        # (Bq, Bd)

        # 5) α * bias
        return self.time_rel_alpha * bias

    # ===== 绝对时间：构造 Fourier table =====
    def _build_fourier_table(self, num_classes: int, emb_dim: int, device: torch.device):
        positions = torch.arange(num_classes, device=device).float()  # (C,)
        if num_classes > 1:
            positions = positions / (num_classes - 1)
        else:
            positions = torch.zeros_like(positions)

        half_dim = emb_dim // 2
        if half_dim == 0:
            return positions.unsqueeze(1)

        div_term = torch.exp(
            torch.arange(half_dim, device=device).float()
            * (-math.log(10000.0) / max(1, half_dim - 1))
        )  # (half_dim,)

        angles = positions.unsqueeze(1) * div_term.unsqueeze(0)  # (C, half_dim)
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        table = torch.cat([sin, cos], dim=-1)  # (C, 2*half_dim)

        if table.size(1) < emb_dim:
            pad = emb_dim - table.size(1)
            pad_tensor = torch.zeros(num_classes, pad, device=device)
            table = torch.cat([table, pad_tensor], dim=-1)
        elif table.size(1) > emb_dim:
            table = table[:, :emb_dim]

        return table  # (C, emb_dim)

    # ----------------- encode / forward -----------------
    def encode_base(self, input_ids, attention_mask):
        """
        Qwen3-Embedding 版的 base：
        - encoder -> last_hidden_state
        - last_token_pool
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        emb = last_token_pool(last_hidden, attention_mask)  # (B, H)
        return emb

    def add_time_context_from_pred(self, emb, tau: float = 1.0):
        """
        绝对时间上下文（Fourier mixture）：
        用 gong/year/month 的预测分布对 Fourier table 做期望，再 residual 加回 emb。
        """
        if not (self.use_time_heads and self.use_time_context_pred):
            out = emb
            if self.normalize:
                out = F.normalize(out, p=2, dim=1)
            return out

        device = emb.device

        gong_logits, year_logits, month_logits = self.time_logits_from_emb(emb)

        gong_p = F.softmax(gong_logits / tau, dim=-1)   # (B, G)
        year_p = F.softmax(year_logits / tau, dim=-1)   # (B, Y)
        month_p = F.softmax(month_logits / tau, dim=-1) # (B, M)

        gong_table = self._build_fourier_table(self.num_gong,  self.time_emb_dim, device)
        year_table = self._build_fourier_table(self.num_year,  self.time_emb_dim, device)
        month_table = self._build_fourier_table(self.num_month, self.time_emb_dim, device)

        gong_ctx = gong_p @ gong_table
        year_ctx = year_p @ year_table
        month_ctx = month_p @ month_table

        time_ctx = torch.cat([gong_ctx, year_ctx, month_ctx], dim=-1)  # (B, 3D)
        time_ctx = self.time_ctx_proj(time_ctx)                        # (B, H)

        out = emb + self.time_ctx_gate * time_ctx
        if self.normalize:
            out = F.normalize(out, p=2, dim=1)
        return out

    def encode(self, input_ids, attention_mask):
        base = self.encode_base(input_ids, attention_mask)
        out = self.add_time_context_from_pred(base)
        return out

    def forward(self, batch):
        """
        训练用前向：

        返回：
          - q_emb, pos_emb, neg_emb：检索向量（已加绝对时间上下文 & L2）
          - q_base, pos_base：纯文本 CLS，用于 time-loss / time-rel-bias
          - [可选] time_rel_bias: (B, B) 时间相对性 bias 矩阵
        """
        q_base = self.encode_base(batch["query_input_ids"], batch["query_attention_mask"])
        pos_base = self.encode_base(batch["pos_input_ids"], batch["pos_attention_mask"])
        neg_base = self.encode_base(batch["neg_input_ids"], batch["neg_attention_mask"])

        # 检索向量：根据配置决定是否加时间上下文
        q_emb = self.add_time_context_from_pred(q_base)
        pos_emb = self.add_time_context_from_pred(pos_base)
        neg_emb = self.add_time_context_from_pred(neg_base)

        # ---------- 新增：时间分类 head 的 logits（完全在 forward 内部） ----------
        gong_logits_q = year_logits_q = month_logits_q = None
        gong_logits_p = year_logits_p = month_logits_p = None
        if self.use_time_heads:
            gong_logits_q, year_logits_q, month_logits_q = self.time_logits(q_base)
            gong_logits_p, year_logits_p, month_logits_p = self.time_logits(pos_base)
        
        # ---------- 相对时间 bias（保持你原来的写法） ----------
        time_rel_bias = None
        if self.use_time_heads and self.use_time_rel_bias:
            time_rel_bias = self.compute_time_rel_bias(q_base, pos_base, tau=1.0)

        # 统一返回：emb + base CLS + 时间 logits + 时间 bias
        return (
            q_emb, pos_emb, neg_emb,
            q_base, pos_base,
            gong_logits_q, year_logits_q, month_logits_q,
            gong_logits_p, year_logits_p, month_logits_p,
            time_rel_bias,
        )

    # ---------- 时间头 helper ----------
    def time_logits(self, emb):
        if not self.use_time_heads:
            return None, None, None
        gong_logits = self.time_head_gong(emb)
        year_logits = self.time_head_year(emb)
        month_logits = self.time_head_month(emb)
        return gong_logits, year_logits, month_logits

    def time_logits_from_emb(self, emb):
        if not self.use_time_heads:
            raise RuntimeError("time heads not enabled")
        gong_logits = self.time_head_gong(emb)
        year_logits = self.time_head_year(emb)
        month_logits = self.time_head_month(emb)
        return gong_logits, year_logits, month_logits


class RetrievalCollator:
    def __init__(
        self,
        tokenizer,
        max_query_len=64,
        max_doc_len=128,
        use_instruction: bool = False,
        task_description: str = "",
    ):
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len
        self.use_instruction = use_instruction
        self.task_description = task_description

    def __call__(self, batch):
        raw_queries = [item["query"] for item in batch]
        pos_texts = [item["pos_text"] for item in batch]

        if self.use_instruction and self.task_description:
            queries = [
                wrap_query_with_instruction(self.task_description, q)
                for q in raw_queries
            ]
        else:
            queries = raw_queries

        # 每样本采 1 个 neg（维持原逻辑）
        neg_texts = []
        for item in batch:
            if item.get("neg_texts"):
                neg_texts.append(random.choice(item["neg_texts"]))
            else:
                neg_texts.append(item["pos_text"])

        # ---- 时间标签：1-based -> 0-based ----
        query_gong_labels = torch.tensor(
            [ex["query_gong_label"] - 1 for ex in batch],
            dtype=torch.long,
        )
        query_year_labels = torch.tensor(
            [ex["query_year_label"] - 1 for ex in batch],
            dtype=torch.long,
        )
        query_month_labels = torch.tensor(
            [ex["query_month_label"] - 1 for ex in batch],
            dtype=torch.long,
        )

        pos_gong_labels = torch.tensor(
            [ex["pos_gong_label"] - 1 for ex in batch],
            dtype=torch.long,
        )
        pos_year_labels = torch.tensor(
            [ex["pos_year_label"] - 1 for ex in batch],
            dtype=torch.long,
        )
        pos_month_labels = torch.tensor(
            [ex["pos_month_label"] - 1 for ex in batch],
            dtype=torch.long,
        )

        # ---- time_id（连续时间轴上的 bin id，不减 1）----
        query_time_ids = torch.tensor(
            [ex["query_time_id"] for ex in batch],
            dtype=torch.long,
        )
        pos_time_ids = torch.tensor(
            [ex["pos_time_id"] for ex in batch],
            dtype=torch.long,
        )

        # ✅ range id（dataset 加了就用；没加就退化成 point）
        query_start_time_ids = torch.tensor(
            [ex.get("query_start_time_id", ex["query_time_id"]) for ex in batch],
            dtype=torch.long,
        )
        query_end_time_ids = torch.tensor(
            [ex.get("query_end_time_id", ex["query_time_id"]) for ex in batch],
            dtype=torch.long,
        )

        # pos range：默认单月句子，= pos_time_id
        pos_start_time_ids = torch.tensor(
            [ex.get("pos_start_time_id", ex["pos_time_id"]) for ex in batch],
            dtype=torch.long,
        )
        pos_end_time_ids = torch.tensor(
            [ex.get("pos_end_time_id", ex["pos_time_id"]) for ex in batch],
            dtype=torch.long,
        )

        # ✅ NEW：query 是否是“纯 no_event” 月份（用于 point_loss 里过滤）
        query_is_pure_no_event = torch.tensor(
            [bool(ex.get("is_pure_no_event", False)) for ex in batch],
            dtype=torch.bool,
        )

        # ---- Tokenization ----
        q_tok = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.max_query_len,
            return_tensors="pt",
        )
        pos_tok = self.tokenizer(
            pos_texts,
            padding=True,
            truncation=True,
            max_length=self.max_doc_len,
            return_tensors="pt",
        )
        neg_tok = self.tokenizer(
            neg_texts,
            padding=True,
            truncation=True,
            max_length=self.max_doc_len,
            return_tensors="pt",
        )

        return {
            "query_input_ids": q_tok["input_ids"],
            "query_attention_mask": q_tok["attention_mask"],
            "pos_input_ids": pos_tok["input_ids"],
            "pos_attention_mask": pos_tok["attention_mask"],
            "neg_input_ids": neg_tok["input_ids"],
            "neg_attention_mask": neg_tok["attention_mask"],

            # CE labels (0-based)
            "query_gong_label": query_gong_labels,
            "query_year_label": query_year_labels,
            "query_month_label": query_month_labels,
            "pos_gong_label": pos_gong_labels,
            "pos_year_label": pos_year_labels,
            "pos_month_label": pos_month_labels,

            # time id (bin)
            "query_time_id": query_time_ids,
            "pos_time_id": pos_time_ids,

            # range（给 multipos / point-type 时间 loss 用）
            "query_start_time_id": query_start_time_ids,
            "query_end_time_id": query_end_time_ids,
            "pos_start_time_id": pos_start_time_ids,
            "pos_end_time_id": pos_end_time_ids,

            # ✅ NEW：纯 no_event 标记
            "query_is_pure_no_event": query_is_pure_no_event,

            # debug
            "raw_queries": raw_queries,
            "wrapped_queries": queries,
            "raw_pos": pos_texts,
            "raw_neg": neg_texts,
        }
