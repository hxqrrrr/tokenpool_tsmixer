"""
Configuration management for RUL prediction experiments
支持为每个模型在每个数据集上设置不同的配置
"""


from torch.nn.modules.conv import F


class Config:
    """Configuration class for training"""
    
    def __init__(self):
        # Dataset settings
        self.data_root = './'
        self.dataset_name = 'FD001'
        self.max_rul = 125
        self.patch_size = 5
        self.time_denpen_len = 6
        self.window_sample = 30
        self.num_sensor = 14
        
        # Training settings
        self.epochs = 50
        self.batch_size = 128
        self.lr = 1e-3
        self.weight_decay = 1e-5
        self.lr_scheduler = True
        self.patience = 10              # Early stopping patience
        self.lr_decay_factor = 0.5      # Learning rate decay factor
        self.lr_patience = 5            # Learning rate scheduler patience
        self.show_interval = 1
        
        # Model settings (TSMixer)
        self.model_name = 'TSMixer'
        self.hidden_dim = 64
        self.num_blocks = 4
        self.dropout = 0.1
        
        # SGA settings (for TSMixerSGA)
        self.sga_time_rr = 4
        self.sga_feat_rr = 4
        self.sga_dropout = 0.15
        self.sga_fuse = 'add'
        self.head_pool = 'none'
        self.use_mlp_head = True
        
        # TokenPool settings (for TokenPoolTSMixer)
        self.num_tokens = 10
        self.token_dim = 128
        self.num_heads = 4
        self.temperature = 1.5
        self.attn_dropout = 0.1
        self.use_pos_encoding = True
        
        # Experiment settings
        self.checkpoint_dir = './checkpoints'
        self.results_dir = './results'
        self.log_dir = './logs'
        self.seed = 42
        self.best_model_metric = 'score'  # Options: 'val_loss', 'rmse', 'score'
    
    def update(self, **kwargs):
        """Update configuration with keyword arguments"""
        for key, value in kwargs.items():
                setattr(self, key, value)
        return self
    
    def __repr__(self):
        """String representation"""
        config_str = "Configuration:\n"
        config_str += "="*50 + "\n"
        for key, value in self.__dict__.items():
            config_str += f"  {key}: {value}\n"
        config_str += "="*50
        return config_str


# ==================== 数据集基础配置 ====================
DATASET_BASE_CONFIG = {
    'FD001': {
        'patch_size': 5,
        'time_denpen_len': 6,
        'window_sample': 30,
        'max_rul': 125,
    },
    'FD002': {
        'patch_size': 5,
        'time_denpen_len': 10,      # 保持50步窗口（最优配置）
        'window_sample': 50,        # 10 × 5 = 50
        'max_rul': 125,
    },
    'FD003': {
        'patch_size': 5,
        'time_denpen_len': 10,
        'window_sample': 50,
        'max_rul': 125,
    },
    'FD004': {
        'patch_size': 5,
        'time_denpen_len': 10,
        'window_sample': 50,
        'max_rul': 125,
    }
}


# ==================== 模型-数据集特定配置 ====================
MODEL_DATASET_CONFIGS = {
    'TSMixer': {
        'FD001': {
            'hidden_dim': 64,
            'num_blocks': 4,
            'dropout': 0.1,
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'epochs': 50,
        },
        'FD002': {
            'hidden_dim': 64,
            'num_blocks': 4,
            'dropout': 0.15,
            'lr': 8e-4,
            'weight_decay': 2e-5,
            'epochs': 50,
        },
        'FD003': {
            'hidden_dim': 96,
            'num_blocks': 6,
            'dropout': 0.08,
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'epochs': 50,
        },
        'FD004': {
            'hidden_dim': 96,
            'num_blocks': 6,
            'dropout': 0.12,
            'lr': 8e-4,
            'weight_decay': 2e-5,
            'epochs': 50,
        },
    },
    
    'TSMixerSGA': {
        'FD001': {
            'hidden_dim': 64,
            'num_blocks': 4,
            'dropout': 0.1,
            'sga_time_rr': 4,
            'sga_feat_rr': 4,
            'sga_dropout': 0.15,
            'sga_fuse': 'add',
            'head_pool': 'none',
            'use_mlp_head': True,
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'epochs': 50,
        },
        'FD002': {
            'hidden_dim': 64,
            'num_blocks': 4,
            'dropout': 0.15,
            'sga_time_rr': 4,
            'sga_feat_rr': 4,
            'sga_dropout': 0.15,
            'sga_fuse': 'add',
            'head_pool': 'none',
            'use_mlp_head': True,
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'epochs': 50,
        },
        'FD003': {
            'hidden_dim': 96,
            'num_blocks': 6,
            'dropout': 0.08,
            'sga_time_rr': 2,
            'sga_feat_rr': 2,
            'sga_dropout': 0.1,
            'sga_fuse': 'add',
            'head_pool': 'none',
            'use_mlp_head': True,
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'epochs': 50,
        },
        'FD004': {
            'hidden_dim': 96,
            'num_blocks': 6,
            'dropout': 0.12,
            'sga_time_rr': 2,
            'sga_feat_rr': 2,
            'sga_dropout': 0.12,
            'sga_fuse': 'add',
            'head_pool': 'none',
            'use_mlp_head': True,
            'lr': 8e-4,
            'weight_decay': 2e-5,
            'epochs': 50,
        },
    },
    
    'STGNN': {
        'FD001': {
            # 训练参数
            'lr': 5e-4,
            'weight_decay': 1e-5,
            'epochs': 41,
            'batch_size': 100,
            # 数据参数（覆盖数据集基础配置）
            'patch_size': 5,
            'time_denpen_len': 6,
            'window_sample': 30,
            # STGNN特定参数
            'stgnn_conv_kernel': 2,
            'stgnn_conv_out': 7,
            'stgnn_num_windows': 8,
            'stgnn_lstmout_dim': 32,
            'stgnn_lstmhidden_dim': 8,
            'stgnn_hidden_dim': 8,
            'stgnn_moving_window': [2, 2],
            'stgnn_stride': [1, 2],
            'stgnn_pool_choice': 'mean',
            'stgnn_decay': 0.7,
        },
        'FD002': {
            # 训练参数
            'lr': 5e-4,
            'weight_decay': 1e-5,
            'epochs': 41,
            'batch_size': 100,
            # 数据参数
            'patch_size': 5,
            'time_denpen_len': 10,      # 与基础配置对齐
            'window_sample': 50,        # 与基础配置对齐
            # STGNN特定参数
            'stgnn_conv_kernel': 2,
            'stgnn_conv_out': 7,
            'stgnn_num_windows': 14,
            'stgnn_lstmout_dim': 12,
            'stgnn_lstmhidden_dim': 8,
            'stgnn_hidden_dim': 8,
            'stgnn_moving_window': [2, 2],
            'stgnn_stride': [1, 2],
            'stgnn_pool_choice': 'mean',
            'stgnn_decay': 0.7,
        },
        'FD003': {
            # 训练参数（严格按照论文）
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'epochs': 41,
            'batch_size': 100,
            # 数据参数（FD003使用特殊的patch_size）
            'patch_size': 2,
            'time_denpen_len': 25,
            'window_sample': 50,
            # STGNN特定参数
            'stgnn_conv_kernel': 2,
            'stgnn_conv_out': 4,
            'stgnn_num_windows': 36,
            'stgnn_lstmout_dim': 6,
            'stgnn_lstmhidden_dim': 8,
            'stgnn_hidden_dim': 24,
            'stgnn_moving_window': [2, 2],
            'stgnn_stride': [1, 2],
            'stgnn_pool_choice': 'mean',
            'stgnn_decay': 0.7,
        },
        'FD004': {
            # 训练参数
            'lr': 5e-4,
            'weight_decay': 1e-5,
            'epochs': 41,
            'batch_size': 100,
            # 数据参数
            'patch_size': 5,
            'time_denpen_len': 10,
            'window_sample': 50,
            # STGNN特定参数
            'stgnn_conv_kernel': 2,
            'stgnn_conv_out': 7,
            'stgnn_num_windows': 14,
            'stgnn_lstmout_dim': 6,
            'stgnn_lstmhidden_dim': 8,
            'stgnn_hidden_dim': 8,
            'stgnn_moving_window': [2, 2],
            'stgnn_stride': [1, 2],
            'stgnn_pool_choice': 'mean',
            'stgnn_decay': 0.7,
        },
    },
    
    'MixerGNN': {
        'FD001': {
            # 训练参数
            'lr': 1e-3,
            'weight_decay': 2e-5,
            'epochs': 40,  # 配合early stop，减少无效训练
            'batch_size': 128,
            'lr_scheduler': False,  # 🔑 禁用scheduler，保持固定LR
            # 数据参数
            'patch_size': 5,
            'time_denpen_len': 6,
            'window_sample': 30,
            # MixerGNN特定参数 - V5优化配置
            'depth_pre': 2,
            'depth_post': 2,
            'time_expansion': 4,
            'feat_expansion': 4,
            'dropout': 0.15,
            'gcn_hidden': 28,
            't_kernel': 7,
            't_dilation': 2,
            'graph_pool': 'ema',
            'graph_temperature': 1.0,
            'use_channel_ffn': True,
            'ffn_expand': 2,
            'pool': 'weighted',
            'use_mlp_head': True,
            'head_hidden_dim': 64,
            # 对齐TSMixer的关键改进
            'use_input_proj': True,  # ✅ 证明有效！提升8.2%
            'use_flatten_head': False,  # 保持Pool模式
            # Early stopping
            'early_stop_patience': 10,
        },
        'FD002': {
            # 训练参数（V3：长序列专用优化）
            'lr': 1e-3,  # 0.0008→0.001 提高LR，加快长序列优化
            'weight_decay': 3e-5,  # 2e-5→3e-5 进一步增强正则
            'epochs': 50,
            'batch_size': 128,
            'lr_scheduler': False,
            # 数据参数
            'patch_size': 5,
            'time_denpen_len': 10,      # 与基础配置对齐
            'window_sample': 50,        # 与基础配置对齐
            # MixerGNN特定参数 - 平衡容量与正则
            'depth_pre': 2,  # 3→2 减小容量，降低长序列优化难度
            'depth_post': 2,  # 3→2 同上
            'time_expansion': 3,  # 4→3 减小time mixing复杂度
            'feat_expansion': 4,
            'dropout': 0.15,  # 0.12→0.15 增强正则，防过拟合
            'gcn_hidden': 32,  # 保持GCN容量
            't_kernel': 7,  # 9→7 减小卷积核
            't_dilation': 2,
            'graph_pool': 'ema',
            'graph_temperature': 0.8,
            'use_channel_ffn': True,
            'ffn_expand': 2,  # 3→2 减小FFN
            'pool': 'weighted',
            'use_mlp_head': True,
            'head_hidden_dim': 64,  # 96→64 简化输出头
            'use_input_proj': True,  # ✅ 保持关键改进
            'use_flatten_head': False,
            'early_stop_patience': 8,  # 10→8 更激进的early stop
        },
        'FD003': {
            # 训练参数（优秀配置，仅微调early stop）
            'lr': 8e-4,  # 已验证有效
            'weight_decay': 1e-5,  # 已验证有效
            'epochs': 60,  # 足够
            'batch_size': 128,
            'lr_scheduler': False,
            # 数据参数
            'patch_size': 5,
            'time_denpen_len': 10,
            'window_sample': 50,
            # MixerGNN特定参数 - 已验证有效的容量配置
            'depth_pre': 3,  # ✅ 验证有效
            'depth_post': 3,  # ✅ 验证有效
            'time_expansion': 4,
            'feat_expansion': 4,
            'dropout': 0.10,  # ✅ 验证有效
            'gcn_hidden': 32,  # ✅ 验证有效
            't_kernel': 9,  # ✅ 验证有效
            't_dilation': 2,
            'graph_pool': 'ema',
            'graph_temperature': 0.8,  # ✅ 验证有效
            'use_channel_ffn': True,
            'ffn_expand': 3,  # ✅ 验证有效
            'pool': 'weighted',
            'use_mlp_head': True,
            'head_hidden_dim': 96,
            'use_input_proj': True,  # ✅ 关键改进
            'use_flatten_head': False,
            'early_stop_patience': 12,  # 15→12 避免验证集过拟合，在Epoch 31左右停止
        },
        'FD004': {
            # 训练参数（V5：最优平衡点）
            'lr': 7e-4,  # V1最优LR，已验证有效
            'weight_decay': 1e-5,  # V1配置，正则适中
            'epochs': 60,
            'batch_size': 128,
            'lr_scheduler': False,
            # 数据参数
            'patch_size': 5,
            'time_denpen_len': 10,
            'window_sample': 50,
            # MixerGNN特定参数 - V1验证有效的配置
            'depth_pre': 3,
            'depth_post': 3,
            'time_expansion': 4,
            'feat_expansion': 4,
            'dropout': 0.10,  # ✅ V1最优值，0.12略过度
            'gcn_hidden': 32,
            't_kernel': 9,
            't_dilation': 2,
            'graph_pool': 'ema',
            'graph_temperature': 0.8,
            'use_channel_ffn': True,
            'ffn_expand': 3,
            'pool': 'weighted',
            'use_mlp_head': True,
            'head_hidden_dim': 96,
            'use_input_proj': True,  # ✅ 保持关键改进
            'use_flatten_head': False,
            'early_stop_patience': 8,  # ✅ V4验证有效，在Epoch 28停止
        },
    },
    
    'TokenPoolTSMixer': {
        'FD001': {
            # 训练参数
            'hidden_dim': 64,
            'num_blocks': 4,
            'dropout': 0.1,
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'epochs': 50,
            # TokenPool 参数
            'num_tokens': 10,          # 压缩到10个token（30步→10token）
            'token_dim': 128,          # token维度，提供足够的表征能力
            'num_heads': 4,            # 4头注意力，平衡多视角与计算效率
            'temperature': 1.5,        # 温度1.5，避免注意力过度集中
            'attn_dropout': 0.1,       # 注意力dropout
            'use_pos_encoding': True,  # 使用位置编码，保留时序信息
        },
        'FD002': {
            # ==================== 实验验证最佳配置 ====================
            # 🏆 最终性能: Score 618.23, RMSE 12.94 (超越TSMixer baseline)
            # 📊 实验历程: 8次系统实验，temperature=1.6是关键sweet spot
            
            # -------------------- TSMixer 主体参数 --------------------
            'hidden_dim': 96,           # TSMixer隐藏层维度
            'num_blocks': 3,            # TSMixer层数 (浅网络泛化更好)
            'dropout': 0.15,            # 主dropout率 (防止过拟合)
            
            # -------------------- TokenPool 核心参数 --------------------
            'num_tokens': 16,           # 🔥 输出token数量 (50步→16tokens，3.1倍压缩)
                                        #    实验值: 16(618.23) < 18(638.68) ✅
            'token_dim': 64,            # 🔥 Token维度 (特征表示能力)
                                        #    实验值: 64(618.23) << 56(649.48) ✅
            'num_heads': 4,             # 多头注意力头数 (token_dim必须被整除)
            'temperature': 1.6,         # 🔥🔥🔥 注意力温度 (最关键参数！)
                                        #    实验曲线: 1.5(639) > 1.6(618) < 1.8(630) < 2.0(644)
                                        #    Sweet Spot: 1.6 ✅
            'attn_dropout': 0.15,       # 🔥 注意力dropout (配合temperature=1.6最优)
                                        #    实验值: 0.15 > 0.12
            'use_pos_encoding': True,   # 位置编码 (保留时序信息)
            
            # -------------------- 训练超参数 --------------------
            'lr': 8e-4,                 # 学习率 (Adam优化器)
            'weight_decay': 2e-5,       # L2正则化系数
            'epochs': 50,               # 最大训练轮数
            'batch_size': 128,          # 批次大小 (显存允许下尽量大)
            'lr_scheduler': True,       # 是否使用学习率衰减
            'patience': 10,             # Early stopping耐心值
            'lr_decay_factor': 0.5,     # 学习率衰减因子
            'lr_patience': 5,           # 学习率衰减的耐心值
            
            # -------------------- 实验注释 --------------------
            # 关键发现：
            # 1. temperature是最敏感参数，0.1变化→20分差异
            # 2. token_dim=64是必须的，降到56会损失30分
            # 3. num_tokens=16足够，增加到18反而变差
            # 4. attn_dropout=0.15配合temp=1.6达到最佳平衡
            # 5. 参数量194K，相比TSMixer(132K)增加47%但性能提升0.9%
        },
        'FD003': {
            # 训练参数
            'hidden_dim': 96,
            'num_blocks': 6,
            'dropout': 0.08,
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'epochs': 50,
            # TokenPool 参数
            'num_tokens': 12,
            'token_dim': 160,          # 更大的token维度，匹配更大的hidden_dim
            'num_heads': 8,            # 更多头，捕捉多工况复杂模式
            'temperature': 1.6,
            'attn_dropout': 0.08,
            'use_pos_encoding': True,
        },
        'FD004': {
            # ==================== 移植自FD002最优配置 (Score: 593.13) ====================
            # 🔄 配置来源: FD002经过14次实验验证的最优参数
            # 📊 FD004特点: 6工况 + 多故障模式，数据复杂度最高
            
            # -------------------- TSMixer 主体参数 --------------------
            'hidden_dim': 64,           # ✅ 从FD002移植：浅而宽架构
            'num_blocks': 4,            # ✅ 从FD002移植：3层最优
            'dropout': 0.15,            # ✅ 从FD002移植：最佳正则化
            
            # -------------------- TokenPool 核心参数 --------------------
            'num_tokens': 16,           # ✅ 从FD002移植：最优压缩比
            'token_dim': 64,            # ✅ 从FD002移植：最优Token维度
            'num_heads': 4,             # ✅ 从FD002移植：4头注意力
            'temperature': 1.8,         # ✅ 从FD002移植：Sweet Spot
                                        #    注：虽然FD004更复杂，但先测试1.6
                                        #    如果性能不佳可尝试1.8或2.0
            'attn_dropout': 0.15,       # ✅ 从FD002移植：配合dropout
            'use_pos_encoding': True,   # ✅ 位置编码
            
            # -------------------- 训练超参数 --------------------
            'lr': 8e-4,                 # ✅ 从FD002移植
            'weight_decay': 2e-5,       # ✅ 从FD002移植
            'epochs': 50,
            'batch_size': 128,
            'lr_scheduler': True,
            'patience': 10,
            'lr_decay_factor': 0.5,
            'lr_patience': 5,
            
            # -------------------- 预期与后续调整 --------------------
            # 预期: FD004比FD002更难，性能可能略差
            # 如果结果不理想，优先尝试:
            #   1. temperature: 1.6 → 1.8 (多工况可能需要更平滑注意力)
            #   2. hidden_dim: 96 → 128 (更复杂数据可能需要更大容量)
            #   3. dropout: 0.15 → 0.18 (防止过拟合)
        },
    },
    
    
    # ==================== MultiScaleTokenPool (多尺度 TokenPool + TSMixer) ====================
    'MultiScaleTokenPool': {
        'FD001': {
            # 训练参数
            'hidden_dim': 64,
            'num_blocks': 4,
            'dropout': 0.1,
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'epochs': 50,
            # 多尺度参数（FD001: 简单单工况）
            'short_patch_size': 5,          # 短尺度：捕捉快速变化/故障征兆
            'long_patch_size': 9,           # 长尺度：捕捉慢漂移/退化趋势
            'num_short_tokens': 8,          # 短尺度更多 token
            'num_long_tokens': 4,           # 长尺度较少 token（总 12 tokens）
            # TokenPool 参数
            'token_dim': 128,
            'num_heads': 4,
            'short_temperature': 1.3,       # 短尺度温度略低，突出尖峰
            'long_temperature': 1.8,        # 长尺度温度略高，避免塌缩
            'attn_dropout': 0.1,
            'use_pos_encoding': True,
            # 融合参数
            'fusion_mode': 'concat',        # 'concat' 或 'gated'
        },
        'FD002': {
            # 训练参数（优化版 V2 - 目标冲击 <660）
            'hidden_dim': 64,
            'num_blocks': 4,
            'dropout': 0.20,                # 0.18 → 0.20 (更强正则，降低波动)
            'lr': 5e-4,                     # 7e-4 → 5e-4 (更低初始LR，更稳定)
            'weight_decay': 4e-5,           # 3e-5 → 4e-5 (更强L2正则)
            'epochs': 70,                   # 60 → 70 (更多epochs，配合低LR)
            # 多尺度参数（Ensemble配置）
            'short_patch_size': 4,          
            'long_patch_size': 4,           
            'num_short_tokens': 12,         
            'num_long_tokens': 12,           
            # TokenPool 参数（增强正则）
            'token_dim': 128,
            'num_heads': 4,
            'short_temperature': 1.4,       # 1.3 → 1.4 (略高，更平滑)
            'long_temperature': 2.1,        # 2.0 → 2.1 (略高，更平滑)
            'attn_dropout': 0.18,           # 0.15 → 0.18 (更强注意力dropout)
            'use_pos_encoding': True,
            # 融合参数
            'fusion_mode': 'concat',
        },
        'FD003': {
            # 训练参数（多工况）
            'hidden_dim': 96,
            'num_blocks': 6,
            'dropout': 0.08,
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'epochs': 50,
            # 多尺度参数（FD003: 与 FD001 类似）
            'short_patch_size': 5,
            'long_patch_size': 9,
            'num_short_tokens': 8,
            'num_long_tokens': 4,           # 总 12 tokens
            # TokenPool 参数
            'token_dim': 160,               # 更大容量
            'num_heads': 8,
            'short_temperature': 1.3,
            'long_temperature': 1.8,
            'attn_dropout': 0.08,
            'use_pos_encoding': True,
            # 融合参数
            'fusion_mode': 'concat',
        },
        'FD004': {
            # 训练参数（最复杂 6 工况，最大收益点）
            'hidden_dim': 64,
            'num_blocks': 4,
            'dropout': 0.15,
            'lr': 8e-4,
            'weight_decay': 2e-5,
            'epochs': 50,
            # 多尺度参数（FD004: 与 FD002 对齐）
            'short_patch_size': 5,
            'long_patch_size': 15,          # 长尺度 15 步
            'num_short_tokens': 10,
            'num_long_tokens': 6,           # 总 16 tokens
            # TokenPool 参数
            'token_dim': 128,
            'num_heads': 4,
            'short_temperature': 1.3,
            'long_temperature': 2.0,        # 高温度
            'attn_dropout': 0.12,
            'use_pos_encoding': True,
            # 融合参数
            'fusion_mode': 'concat',
        },
    },
}


def get_config(model_name='TSMixer', dataset_name='FD001', **kwargs):
    """
    获取指定模型和数据集的配置
    
    Args:
        model_name: 模型名称 (TSMixer, TSMixerSGA, STGNN, TokenPoolTSMixer, MultiScaleTokenPool)
        dataset_name: 数据集名称 (FD001, FD002, FD003, FD004)
        **kwargs: 额外的配置参数，用于覆盖默认值
        
    Returns:
        Config object with model and dataset-specific settings
    """
    config = Config()
    config.model_name = model_name
    config.dataset_name = dataset_name
    
    # 1. 应用数据集基础配置
    if dataset_name in DATASET_BASE_CONFIG:
        config.update(**DATASET_BASE_CONFIG[dataset_name])
    
    # 2. 应用模型-数据集特定配置
    if model_name in MODEL_DATASET_CONFIGS:
        if dataset_name in MODEL_DATASET_CONFIGS[model_name]:
            config.update(**MODEL_DATASET_CONFIGS[model_name][dataset_name])
        else:
            print(f"Warning: No specific config for {model_name} on {dataset_name}, using default")
    else:
        print(f"Warning: Unknown model {model_name}, using default config")
    
    # 3. 应用用户自定义参数（最高优先级）
    if kwargs:
        config.update(**kwargs)
    
    return config


def get_default_config(dataset_name='FD001'):
    """
    获取默认配置（向后兼容）
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        Config object
    """
    return get_config('TSMixer', dataset_name)


def print_all_configs():
    """打印所有模型-数据集的配置概览"""
    print("\n" + "="*80)
    print("所有模型-数据集配置概览")
    print("="*80)
    
    models = ['TSMixer', 'TSMixerSGA', 'STGNN', 'TokenPoolTSMixer', 'MultiScaleTokenPool']
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    
    for model in models:
        print(f"\n{'='*80}")
        print(f"模型: {model}")
        print(f"{'='*80}")
        
        for dataset in datasets:
            config = get_config(model, dataset)
            print(f"\n{dataset}:")
            print(f"  数据: patch_size={config.patch_size}, time_len={config.time_denpen_len}, "
                  f"window={config.window_sample}")
            print(f"  训练: lr={config.lr}, epochs={config.epochs}, batch_size={config.batch_size}")
            
            if model in ['TSMixer', 'TSMixerSGA']:
                print(f"  模型: hidden_dim={config.hidden_dim}, num_blocks={config.num_blocks}, "
                      f"dropout={config.dropout}")
            
            if model == 'TSMixerSGA':
                print(f"  SGA: time_rr={config.sga_time_rr}, feat_rr={config.sga_feat_rr}, "
                      f"dropout={config.sga_dropout}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    # 测试配置系统
    print_all_configs()
