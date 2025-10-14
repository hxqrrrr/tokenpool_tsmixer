"""
Multi-Scale TokenPool + TSMixer for RUL prediction

核心思想：并联两路 TokenPool
- 短尺度路径：小 patch_size，更多 tokens（捕捉快速变化/故障征兆）
- 长尺度路径：大 patch_size，较少 tokens（捕捉慢漂移/退化趋势）

融合方式：
1. Concat（首选）：通道维对齐后拼接
2. 门控融合（可选）：学习自适应权重

优势：
- 兼顾快变征兆与慢漂移
- 对长多工况数据集（FD002/004）收益显著
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base_model import BaseRULModel


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class TokenPoolA(nn.Module):
    """
    Learnable attention-based token pooling
    [B, T, C] -> [B, N, D]
    """
    def __init__(self, 
                 input_dim,
                 output_dim,
                 num_tokens=10,
                 num_heads=4,
                 temperature=1.5,
                 attn_dropout=0.1,
                 use_pos_encoding=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.temperature = temperature
        self.head_dim = output_dim // num_heads
        
        assert output_dim % num_heads == 0
        
        # Key, Value 投影
        self.k_proj = nn.Linear(input_dim, output_dim)
        self.v_proj = nn.Linear(input_dim, output_dim)
        
        # 可学习的查询向量
        self.queries = nn.Parameter(torch.randn(num_tokens, output_dim))
        nn.init.normal_(self.queries, std=0.02)
        
        # 位置编码
        self.use_pos_encoding = use_pos_encoding
        if use_pos_encoding:
            self.pos_encoding = PositionalEncoding(output_dim)
        
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        """
        Args:
            x: [B, T, C]
        Returns:
            pooled: [B, N, D]
            attn_weights: [B, H, N, T]
        """
        B, T, C = x.shape
        
        K = self.k_proj(x)  # [B, T, D]
        V = self.v_proj(x)  # [B, T, D]
        
        if self.use_pos_encoding:
            K = self.pos_encoding(K)
        
        Q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, N, D]
        
        # Multi-head attention
        Q = Q.view(B, self.num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(self.head_dim) * self.temperature)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        pooled = torch.matmul(attn_weights, V)
        pooled = pooled.transpose(1, 2).contiguous().view(B, self.num_tokens, self.output_dim)
        
        pooled = self.out_proj(pooled)
        pooled = self.norm(pooled)
        
        return pooled, attn_weights


class TSMixerBlock(nn.Module):
    """TSMixer block with time and feature mixing"""
    def __init__(self, seq_len, num_features, hidden_dim, dropout=0.1):
        super().__init__()
        
        # Time mixing
        self.time_mixing = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, seq_len),
            nn.Dropout(dropout)
        )
        
        # Feature mixing
        self.feature_mixing = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_features),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(num_features)
        
    def forward(self, x):
        # Time mixing
        residual = x
        x = self.norm1(x)
        x_t = x.transpose(1, 2)
        x_t = self.time_mixing(x_t)
        x = x_t.transpose(1, 2)
        x = x + residual
        
        # Feature mixing
        residual = x
        x = self.norm2(x)
        x = self.feature_mixing(x)
        x = x + residual
        
        return x


class TSMixer(nn.Module):
    """Stack of TSMixer blocks"""
    def __init__(self, seq_len, num_features, hidden_dim=64, num_blocks=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(num_features, num_features)
        self.blocks = nn.ModuleList([
            TSMixerBlock(seq_len, num_features, hidden_dim, dropout)
            for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return x


class GatedFusion(nn.Module):
    """
    门控融合：学习自适应权重融合短尺度和长尺度特征
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, short_scale, long_scale):
        """
        Args:
            short_scale: [B, N_short, D]
            long_scale: [B, N_long, D]
        Returns:
            fused: [B, N_short+N_long, D]
        """
        # 全局池化获取代表性特征
        short_global = short_scale.mean(dim=1)  # [B, D]
        long_global = long_scale.mean(dim=1)    # [B, D]
        
        # 计算门控权重
        concat = torch.cat([short_global, long_global], dim=-1)  # [B, 2D]
        gate_weights = self.gate(concat)  # [B, 2]
        
        # 加权融合（先concat再加权）
        fused = torch.cat([short_scale, long_scale], dim=1)  # [B, N_short+N_long, D]
        
        # 也可以返回门控权重用于分析
        return fused, gate_weights


class MultiScaleTokenPoolTSMixerRUL(BaseRULModel):
    """
    Multi-Scale TokenPool + TSMixer for RUL prediction
    
    Architecture:
        Input [B, time_denpen_len, patch_size, C]
        ↓
        Reshape to two scales
        ├─ Short Scale: [B, T_short, C] → TokenPool(N_short tokens)
        └─ Long Scale:  [B, T_long, C]  → TokenPool(N_long tokens)
        ↓
        Align dimensions & Concat/Fuse → [B, N_total, D]
        ↓
        TSMixer → Head → Output [B]
    
    Args:
        short_patch_size: 短尺度 patch 大小（捕捉快速变化）
        long_patch_size: 长尺度 patch 大小（捕捉慢漂移）
        num_short_tokens: 短尺度 token 数量
        num_long_tokens: 长尺度 token 数量
        fusion_mode: 'concat' 或 'gated'
    """
    def __init__(self, 
                 # 输入参数
                 patch_size=5,           # 原始 patch_size（用于兼容，实际不用）
                 time_denpen_len=6,
                 num_sensor=14,
                 # 多尺度参数
                 short_patch_size=5,     # 短尺度 patch
                 long_patch_size=9,      # 长尺度 patch
                 num_short_tokens=8,     # 短尺度 token 数
                 num_long_tokens=4,      # 长尺度 token 数
                 # TokenPool 参数
                 token_dim=128,
                 num_heads=4,
                 short_temperature=1.3,  # 短尺度温度（略低，突出尖峰）
                 long_temperature=1.8,   # 长尺度温度（略高，避免塌缩）
                 attn_dropout=0.1,
                 use_pos_encoding=True,
                 # 融合参数
                 fusion_mode='concat',   # 'concat' 或 'gated'
                 # TSMixer 参数
                 hidden_dim=64,
                 num_blocks=4,
                 dropout=0.1):
        super().__init__()
        
        self.time_denpen_len = time_denpen_len
        self.num_sensor = num_sensor
        self.short_patch_size = short_patch_size
        self.long_patch_size = long_patch_size
        self.num_short_tokens = num_short_tokens
        self.num_long_tokens = num_long_tokens
        self.fusion_mode = fusion_mode
        
        # 计算每个尺度的序列长度
        self.total_len = time_denpen_len * max(short_patch_size, long_patch_size)
        
        # 短尺度 TokenPool
        self.short_tokenpool = TokenPoolA(
            input_dim=num_sensor,
            output_dim=token_dim,
            num_tokens=num_short_tokens,
            num_heads=num_heads,
            temperature=short_temperature,
            attn_dropout=attn_dropout,
            use_pos_encoding=use_pos_encoding
        )
        
        # 长尺度 TokenPool
        self.long_tokenpool = TokenPoolA(
            input_dim=num_sensor,
            output_dim=token_dim,
            num_tokens=num_long_tokens,
            num_heads=num_heads,
            temperature=long_temperature,
            attn_dropout=attn_dropout,
            use_pos_encoding=use_pos_encoding
        )
        
        # 门控融合（可选）
        if fusion_mode == 'gated':
            self.fusion_gate = GatedFusion(token_dim, dropout)
        else:
            self.fusion_gate = None
        
        # TSMixer（作用在融合后的 tokens 上）
        total_tokens = num_short_tokens + num_long_tokens
        self.tsmixer = TSMixer(
            seq_len=total_tokens,
            num_features=token_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout
        )
        
        # Regression head
        self.output_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_tokens * token_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # 用于监控
        self.last_short_attn = None
        self.last_long_attn = None
        self.last_gate_weights = None
        
    def _reshape_to_scale(self, x, patch_size):
        """
        将输入重塑为指定 patch size 的序列
        Args:
            x: [B, time_denpen_len, orig_patch_size, C]
            patch_size: 目标 patch size
        Returns:
            [B, T, C] where T = time_denpen_len * orig_patch_size / patch_size
        """
        B, L, P, C = x.shape
        # 先展平为 [B, total_len, C]
        x_flat = x.reshape(B, L * P, C)
        
        total_len = L * P
        # 确保能整除
        if total_len % patch_size != 0:
            # 截断到能整除的长度
            valid_len = (total_len // patch_size) * patch_size
            x_flat = x_flat[:, :valid_len, :]
            total_len = valid_len
        
        # 重塑为 [B, T, patch_size, C] 然后平均池化
        T = total_len // patch_size
        x_reshaped = x_flat.reshape(B, T, patch_size, C)
        # 在 patch 维度上平均
        x_pooled = x_reshaped.mean(dim=2)  # [B, T, C]
        
        return x_pooled
    
    def forward(self, x):
        """
        Args:
            x: [B, time_denpen_len, patch_size, num_sensor]
        Returns:
            [B] (squeezed output)
        """
        # 1. 重塑为两个尺度
        x_short = self._reshape_to_scale(x, self.short_patch_size)  # [B, T_short, C]
        x_long = self._reshape_to_scale(x, self.long_patch_size)    # [B, T_long, C]
        
        # 2. 分别通过 TokenPool
        short_tokens, short_attn = self.short_tokenpool(x_short)  # [B, N_short, D]
        long_tokens, long_attn = self.long_tokenpool(x_long)      # [B, N_long, D]
        
        # 保存注意力用于分析
        self.last_short_attn = short_attn.detach()
        self.last_long_attn = long_attn.detach()
        
        # 3. 融合两个尺度
        if self.fusion_mode == 'gated' and self.fusion_gate is not None:
            fused_tokens, gate_weights = self.fusion_gate(short_tokens, long_tokens)
            self.last_gate_weights = gate_weights.detach()
        else:
            # 简单拼接
            fused_tokens = torch.cat([short_tokens, long_tokens], dim=1)  # [B, N_short+N_long, D]
        
        # 4. TSMixer
        mixed = self.tsmixer(fused_tokens)  # [B, N_total, D]
        
        # 5. 输出
        output = self.output_head(mixed)  # [B, 1]
        
        return output.squeeze(-1)
    
    def get_attention_stats(self):
        """获取多尺度注意力统计信息"""
        if self.last_short_attn is None or self.last_long_attn is None:
            return None
        
        stats = {
            'short_attn_mean': self.last_short_attn.mean().item(),
            'short_attn_std': self.last_short_attn.std().item(),
            'long_attn_mean': self.last_long_attn.mean().item(),
            'long_attn_std': self.last_long_attn.std().item(),
            'short_tokens': self.num_short_tokens,
            'long_tokens': self.num_long_tokens,
        }
        
        if self.last_gate_weights is not None:
            stats['gate_weights'] = self.last_gate_weights.mean(dim=0).cpu().numpy()
        
        return stats

