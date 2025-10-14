"""
TSMixer + SGA (Scalable Global Attention) for RUL prediction
在原始 TSMixer 的 Mixer 堆栈之后，串接一个“双轴 squeeze + 自适应融合”的 SGA 门控模块，
同时对时间维与特征维进行全局加权，最后接回归头。
"""

import torch
import torch.nn as nn
from .base_model import BaseRULModel


# ======================== SGA: 双轴可扩展全局注意力 ========================
class SGAGate(nn.Module):
    """
    输入:  x ∈ R^{B×T×F}
    步骤:
      1) 时间轴 squeeze（沿F聚合） -> A_time ∈ R^{B×T×1}，用小MLP产生时间权重
      2) 特征轴 squeeze（沿T聚合） -> A_feat ∈ R^{B×1×F}，用小MLP产生特征权重
      3) 融合:  A = σ( γ · (A_time ⊕ A_feat) + β )    # ⊕ 为广播相加或Hadamard
      4) 加权:  Y = X ⊙ A
    说明:
      - rr_time / rr_feat 控制隐藏层降维比 (reduction ratio)
      - fuse: 'add' 或 'hadamard'
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        rr_time: int = 4,
        rr_feat: int = 4,
        dropout: float = 0.05,
        fuse: str = "add",  # ["add", "hadamard"]
    ):
        super().__init__()
        assert rr_time >= 1 and rr_feat >= 1
        assert fuse in ["add", "hadamard"]

        self.fuse = fuse

        t_hid = max(1, seq_len // rr_time)
        f_hid = max(1, num_features // rr_feat)

        # 时间轴 MLP：对每个时间步的“全局特征均值”做非线性映射
        self.time_mlp = nn.Sequential(
            nn.Linear(1, t_hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(t_hid, 1),
        )

        # 特征轴 MLP：对每个特征的“全局时间均值”做非线性映射
        self.feat_mlp = nn.Sequential(
            nn.Linear(1, f_hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(f_hid, 1),
        )

        # 可学习融合参数
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

        # 最后做一个轻量 LayerNorm 稳定数值（可选）
        self.post_norm = nn.LayerNorm(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        B, T, F = x.shape

        # ---- 1) 时间轴 squeeze：沿特征维求均值 -> [B, T, 1] ----
        t_ctx = x.mean(dim=2, keepdim=True)                  # [B, T, 1]
        a_time = self.time_mlp(t_ctx)                        # [B, T, 1]

        # ---- 2) 特征轴 squeeze：沿时间维求均值 -> [B, 1, F] ----
        f_ctx = x.mean(dim=1, keepdim=True)                  # [B, 1, F]
        # 为了复用线性层，按特征逐点映射 (B,1,F)->(B*F,1)->(B,1,F)
        a_feat = self.feat_mlp(f_ctx.transpose(1, 2)).transpose(1, 2)  # [B, 1, F]

        # ---- 3) 融合 ----
        if self.fuse == "add":
            fused = a_time + a_feat                           # 广播到 [B,T,F]
        else:
            fused = a_time * a_feat                           # Hadamard 广播

        gate = torch.sigmoid(self.gamma * fused + self.beta)  # [B, T, F]

        # ---- 4) 加权并归一化 ----
        y = x * gate
        # 按特征做 LayerNorm，保持分布平稳
        y = self.post_norm(y)
        return y


# ======================== 原始 TSMixer Blocks ========================
class TSMixerBlock(nn.Module):
    """TSMixer block with time and feature mixing"""

    def __init__(self, seq_len, num_features, hidden_dim, dropout=0.1):
        super(TSMixerBlock, self).__init__()

        # Time mixing (across time steps)
        self.time_mixing = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, seq_len),
            nn.Dropout(dropout),
        )

        # Feature mixing (across features/sensors)
        self.feature_mixing = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_features),
            nn.Dropout(dropout),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(num_features)

    def forward(self, x):
        """
        x: [B, T, F]
        """
        # Time mixing with residual connection
        residual = x
        x = self.norm1(x)
        x_t = x.transpose(1, 2)             # [B, F, T]
        x_t = self.time_mixing(x_t)         # [B, F, T]
        x = x_t.transpose(1, 2)             # [B, T, F]
        x = x + residual

        # Feature mixing with residual connection
        residual = x
        x = self.norm2(x)
        x = self.feature_mixing(x)          # [B, T, F]
        x = x + residual

        return x


class TSMixerBackbone(nn.Module):
    """
    仅 TSMixer 主干（输入投影 + 若干 Mixer 块）
    """

    def __init__(self, seq_len, num_features, hidden_dim=64, num_blocks=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(num_features, num_features)
        self.blocks = nn.ModuleList(
            [TSMixerBlock(seq_len, num_features, hidden_dim, dropout) for _ in range(num_blocks)]
        )

    def forward(self, x):
        # x: [B, T, F] 或 [B, Td, P, F]
        if x.dim() == 4:
            B, Td, P, F = x.shape
            x = x.reshape(B, Td * P, F)
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        return x  # [B, T, F]


# ======================== 带 SGA 的 TSMixer 架构 ========================
class TSMixerWithSGA(nn.Module):
    """
    TSMixer 主干 + SGA 门控 + 回归头
    - SGA 放在 Mixer 堆栈之后，作为"轻头盔"做双轴全局门控
    - 支持简单线性头或MLP头
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        hidden_dim: int = 64,
        num_blocks: int = 4,
        dropout: float = 0.1,
        # SGA 超参
        sga_time_rr: int = 4,
        sga_feat_rr: int = 4,
        sga_dropout: float = 0.05,
        sga_fuse: str = "add",          # ["add", "hadamard"]
        # 头部
        head_pool: str = "mean",        # ["mean", "last", "weighted", "none"]
        use_mlp_head: bool = True,      # 是否使用MLP输出头
    ):
        super().__init__()

        self.backbone = TSMixerBackbone(
            seq_len=seq_len,
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout,
        )

        self.sga = SGAGate(
            seq_len=seq_len,
            num_features=num_features,
            rr_time=sga_time_rr,
            rr_feat=sga_feat_rr,
            dropout=sga_dropout,
            fuse=sga_fuse,
        )

        self.head_pool = head_pool
        self.use_mlp_head = use_mlp_head
        
        # 构建输出头
        if use_mlp_head:
            # MLP输出头（类似TSMixerRUL）
            if head_pool == "none":
                # 不池化，Flatten所有特征
                input_size = seq_len * num_features
            else:
                # 池化后再接MLP
                input_size = num_features
            
            self.reg_head = nn.Sequential(
                nn.Linear(input_size, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
        else:
            # 简单线性头
            self.reg_head = nn.Linear(num_features, 1)

    def forward(self, x):
        """
        x: [B, T, F] 或 [B, Td, P, F]
        """
        h = self.backbone(x)     # [B, T, F]
        h = self.sga(h)          # [B, T, F]

        # 池化策略
        if self.head_pool == "mean":
            h = h.mean(dim=1)                    # [B, F]
        elif self.head_pool == "last":
            h = h[:, -1, :]                      # [B, F]
        elif self.head_pool == "weighted":
            # 线性递增权重（越接近末尾权重越大）
            B, T, F = h.shape
            w = torch.linspace(0.1, 1.0, steps=T, device=h.device).view(1, T, 1)
            h = (h * w).sum(dim=1) / w.sum(dim=1)  # [B, F]
        elif self.head_pool == "none":
            # 不池化，Flatten所有特征
            h = h.flatten(1)                     # [B, T*F]
        # else: 保持 [B, T, F] 或其他自定义

        # 通过输出头
        if self.use_mlp_head:
            y = self.reg_head(h).squeeze(-1)     # [B]
        else:
            y = self.reg_head(h).squeeze(-1)     # [B]
        
        return y


# ======================== 对接 BaseRULModel 的封装 ========================
class TSMixerSGARUL(BaseRULModel):
    """
    TSMixer + SGA for RUL prediction
    符合 BaseRULModel 接口规范
    
    默认配置：使用 flatten + MLP 输出头（对标TSMixerRUL）
    
    用法示例：
        # 对标TSMixer（默认）
        model = TSMixerSGARUL(
            patch_size=5, time_denpen_len=6, num_sensor=14,
            hidden_dim=64, num_blocks=4, dropout=0.1
        )
        
        # 轻量版本（使用池化）
        model = TSMixerSGARUL(
            patch_size=5, time_denpen_len=6, num_sensor=14,
            hidden_dim=64, num_blocks=4, dropout=0.1,
            head_pool='mean', use_mlp_head=False
        )
    """

    def __init__(
        self,
        patch_size: int = 5,
        time_denpen_len: int = 6,
        num_sensor: int = 14,
        hidden_dim: int = 64,
        num_blocks: int = 4,
        dropout: float = 0.1,
        # SGA 超参数
        sga_time_rr: int = 4,
        sga_feat_rr: int = 4,
        sga_dropout: float = 0.05,
        sga_fuse: str = "add",
        # 输出头配置（默认对标TSMixer）
        head_pool: str = "none",
        use_mlp_head: bool = True,
    ):
        super().__init__()  # BaseRULModel.__init__() 不接受参数

        seq_len = time_denpen_len * patch_size
        
        self.model = TSMixerWithSGA(
            seq_len=seq_len,
            num_features=num_sensor,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout,
            sga_time_rr=sga_time_rr,
            sga_feat_rr=sga_feat_rr,
            sga_dropout=sga_dropout,
            sga_fuse=sga_fuse,
            head_pool=head_pool,
            use_mlp_head=use_mlp_head,
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, time_denpen_len, patch_size, num_sensor]
        Returns:
            [batch_size] (squeezed output)
        """
        return self.model(x)
