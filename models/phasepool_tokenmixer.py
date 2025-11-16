"""
TokenMixer model for RUL prediction with PhasePool
Token Mixer: An all-MLP Architecture for Time Series Forecasting
PhasePool: Learnable attention-based token pooling for sequence compression
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base_model import BaseRULModel


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time series"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            x with positional encoding added
        """
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class PhasePoolA(nn.Module):
    """
    Learnable attention-based token pooling
    Compresses sequence [B, T, C] -> [B, N, D] using learnable queries
    
    PhasePool 通过可学习的查询向量，对输入序列进行注意力加权池化：
    - 每个 query 是一个"主题探针"，自适应地从时间序列中聚合关键信息
    - 多头注意力机制实现并行的多视角聚合
    - 输出更紧凑但信息丰富的 token 序列
    """
    
    def __init__(self, 
                 input_dim,           # 输入特征维度 C
                 output_dim,          # 输出特征维度 D
                 num_tokens=10,       # 输出 token 数量 N
                 num_heads=4,         # 多头注意力头数
                 temperature=1.5,     # 注意力温度，越大越平滑
                 attn_dropout=0.1,    # 注意力 dropout
                 use_pos_encoding=True):  # 是否使用位置编码
        super(PhasePoolA, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.temperature = temperature
        self.head_dim = output_dim // num_heads
        
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"
        
        # Key, Value 投影
        self.k_proj = nn.Linear(input_dim, output_dim)
        self.v_proj = nn.Linear(input_dim, output_dim)
        
        # 可学习的查询向量 [N, D]
        self.queries = nn.Parameter(torch.empty(num_tokens, output_dim))
        nn.init.trunc_normal_(self.queries, std=0.02, a=-0.04, b=0.04)
        
        # 位置编码
        self.use_pos_encoding = use_pos_encoding
        if use_pos_encoding:
            self.pos_encoding = PositionalEncoding(output_dim)
        
        # 注意力 dropout
        self.attn_dropout = nn.Dropout(attn_dropout)
        
        # 输出投影和 LayerNorm
        self.out_proj = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            pooled: [batch_size, num_tokens, output_dim]
            attn_weights: [batch_size, num_heads, num_tokens, seq_len] (for monitoring)
        """
        B, T, C = x.shape
        
        # 1. 线性投影得到 K, V
        K = self.k_proj(x)  # [B, T, D]
        V = self.v_proj(x)  # [B, T, D]
        
        # 2. 给 K 加位置编码（保留时序信息）
        if self.use_pos_encoding:
            K = self.pos_encoding(K)
        
        # 3. 获取可学习的 Query，扩展到 batch
        Q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, N, D]
        
        # 4. 重塑为多头形式
        # Q: [B, N, D] -> [B, N, H, head_dim] -> [B, H, N, head_dim]
        # K: [B, T, D] -> [B, T, H, head_dim] -> [B, H, T, head_dim]
        # V: [B, T, D] -> [B, T, H, head_dim] -> [B, H, T, head_dim]
        Q = Q.view(B, self.num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 5. 计算注意力分数（带温度缩放）
        # scores: [B, H, N, T]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(self.head_dim) * self.temperature)
        attn_weights = torch.softmax(scores, dim=-1)  # [B, H, N, T]
        attn_weights = self.attn_dropout(attn_weights)
        
        # 6. 加权聚合 V
        # [B, H, N, T] @ [B, H, T, head_dim] -> [B, H, N, head_dim]
        pooled = torch.matmul(attn_weights, V)
        
        # 7. 拼接多头并投影
        # [B, H, N, head_dim] -> [B, N, H, head_dim] -> [B, N, D]
        pooled = pooled.transpose(1, 2).contiguous().view(B, self.num_tokens, self.output_dim)
        
        # 8. 输出投影和残差连接（这里没有残差，因为维度可能不同）
        pooled = self.out_proj(pooled)
        pooled = self.norm(pooled)
        
        return pooled, attn_weights


class PhasePoolATransformerPE(nn.Module):
    """
    PhasePool with positional encoding on input (Transformer-style)
    This variant applies positional encoding to the input before projection,
    following the standard Transformer architecture.
    
    Key difference from PhasePoolA:
    - PhasePoolA: PE on Key after projection (K̃ = K + PE)
    - PhasePoolATransformerPE: PE on input before projection (X̃ = X_proj + PE, then K = k_proj(X̃))
    """
    
    def __init__(self, 
                 input_dim,           # 输入特征维度 C
                 output_dim,          # 输出特征维度 D
                 num_tokens=10,       # 输出 token 数量 N
                 num_heads=4,         # 多头注意力头数
                 temperature=1.5,     # 注意力温度，越大越平滑
                 attn_dropout=0.1,    # 注意力 dropout
                 use_pos_encoding=True):  # 是否使用位置编码
        super(PhasePoolATransformerPE, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.temperature = temperature
        self.head_dim = output_dim // num_heads
        
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"
        
        # 输入投影层（新增）：将输入从C维投影到D维
        self.input_proj = nn.Linear(input_dim, output_dim)
        
        # 位置编码（加在投影后的输入上）
        self.use_pos_encoding = use_pos_encoding
        if use_pos_encoding:
            self.pos_encoding = PositionalEncoding(output_dim)
        
        # Key, Value 投影（从带位置编码的输入投影）
        self.k_proj = nn.Linear(output_dim, output_dim)
        self.v_proj = nn.Linear(output_dim, output_dim)
        
        # 可学习的查询向量 [N, D]
        self.queries = nn.Parameter(torch.empty(num_tokens, output_dim))
        nn.init.trunc_normal_(self.queries, std=0.02, a=-0.04, b=0.04)
        
        # 注意力 dropout
        self.attn_dropout = nn.Dropout(attn_dropout)
        
        # 输出投影和 LayerNorm
        self.out_proj = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            pooled: [batch_size, num_tokens, output_dim]
            attn_weights: [batch_size, num_heads, num_tokens, seq_len] (for monitoring)
        """
        B, T, C = x.shape
        
        # 1. 输入投影：C -> D
        x_proj = self.input_proj(x)  # [B, T, C] -> [B, T, D]
        
        # 2. 加位置编码（Transformer方式：加在输入上）
        if self.use_pos_encoding:
            x_pe = self.pos_encoding(x_proj)  # [B, T, D] + PE
        else:
            x_pe = x_proj
        
        # 3. 从带位置编码的输入投影K, V
        K = self.k_proj(x_pe)  # [B, T, D]
        V = self.v_proj(x_pe)  # [B, T, D]
        
        # 4. 获取可学习的 Query，扩展到 batch
        Q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, N, D]
        
        # 5. 重塑为多头形式
        # Q: [B, N, D] -> [B, N, H, head_dim] -> [B, H, N, head_dim]
        # K: [B, T, D] -> [B, T, H, head_dim] -> [B, H, T, head_dim]
        # V: [B, T, D] -> [B, T, H, head_dim] -> [B, H, T, head_dim]
        Q = Q.view(B, self.num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 6. 计算注意力分数（带温度缩放）
        # scores: [B, H, N, T]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(self.head_dim) * self.temperature)
        attn_weights = torch.softmax(scores, dim=-1)  # [B, H, N, T]
        attn_weights = self.attn_dropout(attn_weights)
        
        # 7. 加权聚合 V
        # [B, H, N, T] @ [B, H, T, head_dim] -> [B, H, N, head_dim]
        pooled = torch.matmul(attn_weights, V)
        
        # 8. 拼接多头并投影
        # [B, H, N, head_dim] -> [B, N, H, head_dim] -> [B, N, D]
        pooled = pooled.transpose(1, 2).contiguous().view(B, self.num_tokens, self.output_dim)
        
        # 9. 输出投影和残差连接（这里没有残差，因为维度可能不同）
        pooled = self.out_proj(pooled)
        pooled = self.norm(pooled)
        
        return pooled, attn_weights


class PhasePoolSelfAttention(nn.Module):
    """
    Standard T×T self-attention (ablation baseline)
    Keeps sequence length [B, T, C] -> [B, T, D] without compression
    
    This is an ablation variant that replaces learnable query cross-attention
    with standard self-attention to evaluate the benefit of the cross-attention design.
    """
    
    def __init__(self, 
                 input_dim,           # 输入特征维度 C
                 output_dim,          # 输出特征维度 D
                 num_tokens=10,       # 未使用，仅为保持接口一致
                 num_heads=4,         # 多头注意力头数
                 temperature=1.5,     # 注意力温度，越大越平滑
                 attn_dropout=0.1,    # 注意力 dropout
                 use_pos_encoding=True):  # 是否使用位置编码
        super(PhasePoolSelfAttention, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.temperature = temperature
        self.head_dim = output_dim // num_heads
        
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"
        
        # Q, K, V 投影 (standard self-attention)
        self.q_proj = nn.Linear(input_dim, output_dim)
        self.k_proj = nn.Linear(input_dim, output_dim)
        self.v_proj = nn.Linear(input_dim, output_dim)
        
        # 位置编码
        self.use_pos_encoding = use_pos_encoding
        if use_pos_encoding:
            self.pos_encoding = PositionalEncoding(output_dim)
        
        # 注意力 dropout
        self.attn_dropout = nn.Dropout(attn_dropout)
        
        # 输出投影和 LayerNorm
        self.out_proj = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            output: [batch_size, seq_len, output_dim]  # 注意：保持序列长度T
            attn_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        B, T, C = x.shape
        
        # 1. 线性投影得到 Q, K, V
        Q = self.q_proj(x)  # [B, T, D]
        K = self.k_proj(x)  # [B, T, D]
        V = self.v_proj(x)  # [B, T, D]
        
        # 2. 给 Q, K 加位置编码
        if self.use_pos_encoding:
            Q = self.pos_encoding(Q)
            K = self.pos_encoding(K)
        
        # 3. 重塑为多头形式
        # Q, K, V: [B, T, D] -> [B, T, H, head_dim] -> [B, H, T, head_dim]
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 4. 计算注意力分数（带温度缩放）
        # scores: [B, H, T, T]  # 注意：T×T self-attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(self.head_dim) * self.temperature)
        attn_weights = torch.softmax(scores, dim=-1)  # [B, H, T, T]
        attn_weights = self.attn_dropout(attn_weights)
        
        # 5. 加权聚合 V
        # [B, H, T, T] @ [B, H, T, head_dim] -> [B, H, T, head_dim]
        output = torch.matmul(attn_weights, V)
        
        # 6. 拼接多头并投影
        # [B, H, T, head_dim] -> [B, T, H, head_dim] -> [B, T, D]
        output = output.transpose(1, 2).contiguous().view(B, T, self.output_dim)
        
        # 7. 输出投影和 LayerNorm
        output = self.out_proj(output)
        output = self.norm(output)
        
        return output, attn_weights


class TokenMixerBlock(nn.Module):
    """TokenMixer block with time and feature mixing"""
    
    def __init__(self, seq_len, num_features, hidden_dim, dropout=0.1, 
                 use_time_mix=True, use_feature_mix=True):
        super(TokenMixerBlock, self).__init__()
        
        self.use_time_mix = use_time_mix
        self.use_feature_mix = use_feature_mix
        
        # Time mixing (across time steps)
        self.time_mixing = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, seq_len),
            nn.Dropout(dropout)
        )
        
        # Feature mixing (across features/sensors)
        self.feature_mixing = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_features),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(num_features)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, num_features]
        Returns:
            [batch_size, seq_len, num_features]
        """
        # Time mixing with residual connection
        if self.use_time_mix:
            residual = x
            x = self.norm1(x)
            x_t = x.transpose(1, 2)  # [B, F, T]
            x_t = self.time_mixing(x_t)  # [B, F, T]
            x = x_t.transpose(1, 2)  # [B, T, F]
            x = x + residual
        
        # Feature mixing with residual connection
        if self.use_feature_mix:
            residual = x
            x = self.norm2(x)
            x = self.feature_mixing(x)
            x = x + residual
        
        return x


class TokenMixer(nn.Module):
    """TokenMixer model for RUL prediction"""
    
    def __init__(self, seq_len, num_features, hidden_dim=64, num_blocks=4, 
                 dropout=0.1, output_dim=1, use_time_mix=True, use_feature_mix=True):
        super(TokenMixer, self).__init__()
        
        self.seq_len = seq_len
        self.num_features = num_features
        
        # Input projection
        self.input_proj = nn.Linear(num_features, num_features)
        
        # Stack of TokenMixer blocks
        self.blocks = nn.ModuleList([
            TokenMixerBlock(seq_len, num_features, hidden_dim, dropout, 
                        use_time_mix=use_time_mix, use_feature_mix=use_feature_mix)
            for _ in range(num_blocks)
        ])
        
        # Output head for regression
        self.output_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * num_features, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, num_features] or 
               [batch_size, time_denpen_len, patch_size, num_features]
        Returns:
            [batch_size, output_dim]
        """
        # Handle 4D input
        if x.dim() == 4:
            batch_size, time_denpen_len, patch_size, num_features = x.shape
            x = x.reshape(batch_size, time_denpen_len * patch_size, num_features)
        
        # Input projection
        x = self.input_proj(x)
        
        # Pass through TokenMixer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output prediction
        output = self.output_head(x)
        
        return output


class PhasePoolTokenMixerRUL(BaseRULModel):
    """
    PhasePool + TokenMixer for RUL prediction
    
    架构流程:
    1. PhasePool: 将 [B, T, C] 通过可学习注意力池化压缩为 [B, N, D]
       - 每个 token 是一个"主题探针"，自适应聚合时间序列的关键信息
       - 多头注意力实现并行的多视角特征提取
    2. TokenMixer: 在压缩后的 token 序列上进行 Time-Mix 和 Feature-Mix
       - Time-Mix: 跨 token（时间维度）的全局混合
       - Feature-Mix: 跨通道（特征维度）的全局混合
    3. Regression Head: 最终回归预测 RUL
    
    优势:
    - 自适应选时: PhasePool 能自动放大重要时段、弱化噪声
    - 更短序列: 压缩后的序列让 Mixer 训练更稳定、计算更高效
    - 软分块: 比固定 patch 更灵活，跨块信息也能被聚合
    """
    
    def __init__(self, 
                 patch_size=5, 
                 time_denpen_len=6, 
                 num_sensor=14,
                 # PhasePool 参数
                 num_tokens=10,          # 压缩到的 token 数量（T -> N）
                 token_dim=128,          # token 的特征维度
                 num_heads=4,            # 多头注意力头数
                 temperature=1.5,        # 注意力温度
                 attn_dropout=0.1,       # 注意力 dropout
                 use_pos_encoding=True,  # 是否使用位置编码
                 pe_on_input=False,      # 位置编码是否加在输入上（Transformer方式，默认False即当前方式）
                 # TokenMixer 参数
                 hidden_dim=64, 
                 num_blocks=4, 
                 dropout=0.1,
                 # 消融实验参数
                 use_phasepool=True,     # 是否使用PhasePool（False则用平均池化）
                 use_self_attention=False,  # 是否使用Self-Attention替代Cross-Attention
                 use_time_mix=True,      # 是否使用Time-Mix
                 use_feature_mix=True,   # 是否使用Feature-Mix
                 use_tokenmixer=True):   # 是否使用TokenMixer（False则用简单MLP）
        super(PhasePoolTokenMixerRUL, self).__init__()
        
        # 保存所有初始化参数为实例属性（用于checkpoint保存）
        self.patch_size = patch_size
        self.time_denpen_len = time_denpen_len
        self.num_sensor = num_sensor
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.temperature = temperature
        self.attn_dropout = attn_dropout
        self.use_pos_encoding = use_pos_encoding
        self.pe_on_input = pe_on_input  # 保存新参数
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.use_phasepool = use_phasepool
        self.use_self_attention = use_self_attention
        self.use_time_mix = use_time_mix
        self.use_feature_mix = use_feature_mix
        self.use_tokenmixer = use_tokenmixer
        
        # PhasePool: 选择Cross-Attention或Self-Attention
        if use_self_attention:
            # 使用标准Self-Attention (T×T)，不压缩序列
            self.phase_pool = PhasePoolSelfAttention(
                input_dim=num_sensor,
                output_dim=token_dim,
                num_tokens=num_tokens,  # 未使用，仅保持接口一致
                num_heads=num_heads,
                temperature=temperature,
                attn_dropout=attn_dropout,
                use_pos_encoding=use_pos_encoding
            )
        else:
            # 使用Cross-Attention with learnable queries (N×T)，压缩序列
            if pe_on_input:
                # 标准Transformer方式：位置编码加在输入上
                self.phase_pool = PhasePoolATransformerPE(
                    input_dim=num_sensor,
                    output_dim=token_dim,
                    num_tokens=num_tokens,
                    num_heads=num_heads,
                    temperature=temperature,
                    attn_dropout=attn_dropout,
                    use_pos_encoding=use_pos_encoding
                )
            else:
                # 当前方式：位置编码加在投影后的Key上
                self.phase_pool = PhasePoolA(
                    input_dim=num_sensor,
                    output_dim=token_dim,
                    num_tokens=num_tokens,
                    num_heads=num_heads,
                    temperature=temperature,
                    attn_dropout=attn_dropout,
                    use_pos_encoding=use_pos_encoding
                )
        
        # 平均池化的备用方案（消融实验用）
        if not use_phasepool:
            # 投影层：将输入特征维度从num_sensor投影到token_dim
            self.avg_pool_proj = nn.Linear(num_sensor, token_dim)
        
        # TokenMixer: 在压缩后的 token 序列上操作
        # 如果使用self-attention，序列长度为time_denpen_len * patch_size；否则为num_tokens
        actual_seq_len = time_denpen_len * patch_size if use_self_attention else num_tokens
        self.tokenmixer = TokenMixer(
            seq_len=actual_seq_len,   # Self-attention保持T=time_denpen_len*patch_size，Cross-attention压缩到N
            num_features=token_dim,    # 特征维度变为 token_dim
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout,
            output_dim=1,
            use_time_mix=use_time_mix,
            use_feature_mix=use_feature_mix
        )
        
        # 简单回归头（用于无TokenMixer消融实验）
        if not use_tokenmixer:
            simple_head_input_dim = actual_seq_len * token_dim  # 适配self-attention的序列长度
            self.simple_head = nn.Sequential(
                nn.Linear(simple_head_input_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
        
        # 用于监控注意力权重的钩子（可选）
        self.last_attn_weights = None
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, time_denpen_len, patch_size, num_sensor]
        Returns:
            [batch_size] (squeezed output)
        """
        # 1. 重塑为 [B, T, C] 格式
        if x.dim() == 4:
            batch_size, time_denpen_len, patch_size, num_sensor = x.shape
            x = x.reshape(batch_size, time_denpen_len * patch_size, num_sensor)
        
        # 2. 序列压缩: [B, T, C] -> [B, N, D]
        if self.use_phasepool:
            # PhasePool: 注意力池化
            x, attn_weights = self.phase_pool(x)
            # 保存注意力权重用于可视化/监控（可选）
            self.last_attn_weights = attn_weights.detach()
        else:
            # 平均池化: 将序列均匀分成num_tokens段，每段取平均
            # [B, T, C] -> [B, N, T//N, C] -> [B, N, C] -> [B, N, D]
            B, T, C = x.shape
            # 确保T能被num_tokens整除，如果不能，截断
            T_new = (T // self.num_tokens) * self.num_tokens
            x = x[:, :T_new, :]  # [B, T_new, C]
            # 重塑为 [B, N, T//N, C]
            x = x.reshape(B, self.num_tokens, T_new // self.num_tokens, C)
            # 在时间维度上平均
            x = x.mean(dim=2)  # [B, N, C]
            # 投影到token_dim
            x = self.avg_pool_proj(x)  # [B, N, D]
            self.last_attn_weights = None
        
        # 3. TokenMixer或简单MLP
        if self.use_tokenmixer:
            # 使用TokenMixer处理: [B, N, D] -> [B, 1]
            output = self.tokenmixer(x)
        else:
            # 展平后使用简单MLP: [B, N, D] -> [B, N*D] -> [B, 1]
            x_flat = x.reshape(x.size(0), -1)  # [B, N*D]
            output = self.simple_head(x_flat)  # [B, 1]
        
        # 4. 返回标量预测
        return output.squeeze(-1)
    
    def get_attention_stats(self):
        """
        获取注意力权重的统计信息，用于监控和调试
        Returns:
            dict: 包含注意力统计的字典
        """
        if self.last_attn_weights is None:
            return None
        
        # [B, H, N, T]
        attn = self.last_attn_weights
        
        stats = {
            'attn_mean': attn.mean().item(),
            'attn_std': attn.std().item(),
            'attn_max': attn.max().item(),
            'attn_min': attn.min().item(),
            # 注意力峰值：每个 token 最关注的时间步的平均权重
            'attn_peak': attn.max(dim=-1)[0].mean().item(),
            # 注意力覆盖度：平均有多少时间步被显著关注（权重 > 均值）
            'attn_coverage': (attn > attn.mean(dim=-1, keepdim=True)).float().mean().item(),
        }
        
        return stats

