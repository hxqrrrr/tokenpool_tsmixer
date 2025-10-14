"""
TSMixer model for RUL prediction
Time Series Mixer: An all-MLP Architecture for Time Series Forecasting
"""
import torch
import torch.nn as nn
from .base_model import BaseRULModel


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
        residual = x
        x = self.norm1(x)
        x_t = x.transpose(1, 2)  # [B, F, T]
        x_t = self.time_mixing(x_t)  # [B, F, T]
        x = x_t.transpose(1, 2)  # [B, T, F]
        x = x + residual
        
        # Feature mixing with residual connection
        residual = x
        x = self.norm2(x)
        x = self.feature_mixing(x)
        x = x + residual
        
        return x


class TSMixer(BaseRULModel):
    """TSMixer model for RUL prediction"""
    
    def __init__(self, seq_len, num_features, hidden_dim=64, num_blocks=4, 
                 dropout=0.1, output_dim=1):
        super(TSMixer, self).__init__()
        
        self.seq_len = seq_len
        self.num_features = num_features
        
        # Input projection
        self.input_proj = nn.Linear(num_features, num_features)
        
        # Stack of TSMixer blocks
        self.blocks = nn.ModuleList([
            TSMixerBlock(seq_len, num_features, hidden_dim, dropout)
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
        
        # Pass through TSMixer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output prediction
        output = self.output_head(x)
        
        return output


class TSMixerRUL(BaseRULModel):
    """TSMixer specifically designed for RUL prediction task"""
    
    def __init__(self, 
                 patch_size=5, 
                 time_denpen_len=6, 
                 num_sensor=14, 
                 hidden_dim=64, 
                 num_blocks=4, 
                 dropout=0.1):
        super(TSMixerRUL, self).__init__()
        
        seq_len = time_denpen_len * patch_size
        
        self.tsmixer = TSMixer(
            seq_len=seq_len,
            num_features=num_sensor,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout,
            output_dim=1
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, time_denpen_len, patch_size, num_sensor]
        Returns:
            [batch_size] (squeezed output)
        """
        output = self.tsmixer(x)
        return output.squeeze(-1)

