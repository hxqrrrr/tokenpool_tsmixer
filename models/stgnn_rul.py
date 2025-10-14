"""
Spatial-Temporal Graph Neural Network for RUL Prediction
时空图神经网络用于剩余使用寿命预测
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
from .base_model import BaseRULModel


# ==================== 基础组件 ====================

class Feature_extractor_1DCNN_RUL(nn.Module):
    """1D CNN特征提取器"""
    def __init__(self, input_channels, num_hidden, out_dim, kernel_size=8, stride=1, dropout=0):
        super(Feature_extractor_1DCNN_RUL, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, num_hidden, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=(kernel_size//2)),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(num_hidden, out_dim, kernel_size=kernel_size, stride=1, bias=False, padding=(kernel_size//2)),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x_in):
        ### input dim is (bs, tlen, feature_dim)
        x = torch.transpose(x_in, -1, -2)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return x


class Dot_Graph_Construction_weights(nn.Module):
    """点积图构造（带权重映射）"""
    def __init__(self, input_dim):
        super().__init__()
        self.mapping = nn.Linear(input_dim, input_dim)

    def forward(self, node_features):
        node_features = self.mapping(node_features)
        bs, N, dimen = node_features.size()

        node_features_1 = torch.transpose(node_features, 1, 2)
        Adj = torch.bmm(node_features, node_features_1)

        device = node_features.device
        eyes_like = torch.eye(N).repeat(bs, 1, 1).to(device)
        eyes_like_inf = eyes_like * 1e8
        Adj = F.leaky_relu(Adj - eyes_like_inf)
        Adj = F.softmax(Adj, dim=-1)
        Adj = Adj + eyes_like

        return Adj


class MPNN_mk_v2(nn.Module):
    """消息传递神经网络（带批归一化）"""
    def __init__(self, input_dimension, output_dimension, k):
        super(MPNN_mk_v2, self).__init__()
        self.way_multi_field = 'sum'
        self.k = k
        theta = []
        for kk in range(self.k):
            theta.append(nn.Linear(input_dimension, output_dimension))
        self.theta = nn.ModuleList(theta)
        self.bn1 = nn.BatchNorm1d(output_dimension)

    def forward(self, X, A):
        ## size of X is (bs, N, A)
        ## size of A is (bs, N, N)
        GCN_output_ = []
        for kk in range(self.k):
            if kk == 0:
                A_ = A
            else:
                A_ = torch.bmm(A_, A)
            out_k = self.theta[kk](torch.bmm(A_, X))
            GCN_output_.append(out_k)

        if self.way_multi_field == 'cat':
            GCN_output_ = torch.cat(GCN_output_, -1)
        elif self.way_multi_field == 'sum':
            GCN_output_ = sum(GCN_output_)

        GCN_output_ = torch.transpose(GCN_output_, -1, -2)
        GCN_output_ = self.bn1(GCN_output_)
        GCN_output_ = torch.transpose(GCN_output_, -1, -2)

        return F.leaky_relu(GCN_output_)


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(100.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def Conv_GraphST(input, time_window_size, stride):
    """时空图卷积"""
    ## input size is (bs, time_length, num_sensors, feature_dim)
    ## output size is (bs, num_windows, num_sensors, time_window_size, feature_dim)
    bs, time_length, num_sensors, feature_dim = input.size()
    x_ = torch.transpose(input, 1, 3)

    y_ = F.unfold(x_, (num_sensors, time_window_size), stride=stride)

    y_ = torch.reshape(y_, [bs, feature_dim, num_sensors, time_window_size, -1])
    y_ = torch.transpose(y_, 1, -1)

    return y_


def Mask_Matrix(num_node, time_length, decay_rate, device='cpu'):
    """掩码矩阵（带时间衰减）"""
    Adj = torch.ones(num_node * time_length, num_node * time_length).to(device)
    for i in range(time_length):
        v = 0
        for r_i in range(i, time_length):
            idx_s_row = i * num_node
            idx_e_row = (i + 1) * num_node
            idx_s_col = (r_i) * num_node
            idx_e_col = (r_i + 1) * num_node
            Adj[idx_s_row:idx_e_row, idx_s_col:idx_e_col] = Adj[idx_s_row:idx_e_row, idx_s_col:idx_e_col] * (decay_rate ** v)
            v = v + 1
        v = 0
        for r_i in range(i + 1):
            idx_s_row = i * num_node
            idx_e_row = (i + 1) * num_node
            idx_s_col = (i - r_i) * num_node
            idx_e_col = (i - r_i + 1) * num_node
            Adj[idx_s_row:idx_e_row, idx_s_col:idx_e_col] = Adj[idx_s_row:idx_e_row, idx_s_col:idx_e_col] * (decay_rate ** v)
            v = v + 1

    return Adj


class GraphConvpoolMPNN_block_v6(nn.Module):
    """图卷积池化MPNN块"""
    def __init__(self, input_dim, output_dim, num_sensors, time_length, time_window_size, stride, decay, pool_choice):
        super(GraphConvpoolMPNN_block_v6, self).__init__()
        self.time_window_size = time_window_size
        self.stride = stride
        self.output_dim = output_dim

        self.graph_construction = Dot_Graph_Construction_weights(input_dim)
        self.BN = nn.BatchNorm1d(input_dim)
        self.MPNN = MPNN_mk_v2(input_dim, output_dim, k=1)

        # 延迟初始化pre_relation
        self.num_sensors = num_sensors
        self.decay = decay
        self.pre_relation = None

        self.pool_choice = pool_choice

    def forward(self, input):
        ## input size (bs, time_length, num_nodes, input_dim)
        ## output size (bs, output_node_t, output_node_s, output_dim)

        input_con = Conv_GraphST(input, self.time_window_size, self.stride)
        ## input_con size (bs, num_windows, num_sensors, time_window_size, feature_dim)
        bs, num_windows, num_sensors, time_window_size, feature_dim = input_con.size()
        input_con_ = torch.transpose(input_con, 2, 3)
        input_con_ = torch.reshape(input_con_, [bs * num_windows, time_window_size * num_sensors, feature_dim])

        # 初始化pre_relation（如果还没有）
        if self.pre_relation is None:
            self.pre_relation = Mask_Matrix(self.num_sensors, self.time_window_size, self.decay, device=input.device)

        A_input = self.graph_construction(input_con_)
        A_input = A_input * self.pre_relation

        input_con_ = torch.transpose(input_con_, -1, -2)
        input_con_ = self.BN(input_con_)
        input_con_ = torch.transpose(input_con_, -1, -2)
        X_output = self.MPNN(input_con_, A_input)

        X_output = torch.reshape(X_output, [bs, num_windows, time_window_size, num_sensors, self.output_dim])

        if self.pool_choice == 'mean':
            X_output = torch.mean(X_output, 2)
        elif self.pool_choice == 'max':
            X_output, ind = torch.max(X_output, 2)
        else:
            print('input choice for pooling cannot be read')

        return X_output


# ==================== 主模型 ====================

class FC_STGNN_RUL(nn.Module):
    """时空图神经网络用于RUL预测（底层网络）"""
    def __init__(self, indim_fea, Conv_out, lstmhidden_dim, lstmout_dim, conv_kernel, hidden_dim, 
                 time_length, num_node, num_windows, moving_window, stride, decay, pooling_choice, n_class):
        super(FC_STGNN_RUL, self).__init__()
        
        self.nonlin_map = Feature_extractor_1DCNN_RUL(1, lstmhidden_dim, lstmout_dim, kernel_size=conv_kernel)
        self.nonlin_map2 = nn.Sequential(
            nn.Linear(lstmout_dim * Conv_out, 2 * hidden_dim),
            nn.BatchNorm1d(2 * hidden_dim)
        )

        self.positional_encoding = PositionalEncoding(2 * hidden_dim, 0.1, max_len=5000)

        self.MPNN1 = GraphConvpoolMPNN_block_v6(2 * hidden_dim, hidden_dim, num_node, time_length, 
                                                 time_window_size=moving_window[0], stride=stride[0], 
                                                 decay=decay, pool_choice=pooling_choice)
        self.MPNN2 = GraphConvpoolMPNN_block_v6(2 * hidden_dim, hidden_dim, num_node, time_length, 
                                                 time_window_size=moving_window[1], stride=stride[1], 
                                                 decay=decay, pool_choice=pooling_choice)

        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(hidden_dim * num_windows * num_node, 2 * hidden_dim)),
            ('relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(2 * hidden_dim, 2 * hidden_dim)),
            ('relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(2 * hidden_dim, hidden_dim)),
            ('relu3', nn.ReLU(inplace=True)),
            ('fc4', nn.Linear(hidden_dim, n_class)),
        ]))

    def forward(self, X):
        bs, tlen, num_node, dimension = X.size()

        ### Graph Generation
        A_input = torch.reshape(X, [bs * tlen * num_node, dimension, 1])
        A_input_ = self.nonlin_map(A_input)
        A_input_ = torch.reshape(A_input_, [bs * tlen * num_node, -1])
        A_input_ = self.nonlin_map2(A_input_)
        A_input_ = torch.reshape(A_input_, [bs, tlen, num_node, -1])

        ## positional encoding
        X_ = torch.reshape(A_input_, [bs, tlen, num_node, -1])
        X_ = torch.transpose(X_, 1, 2)
        X_ = torch.reshape(X_, [bs * num_node, tlen, -1])
        X_ = self.positional_encoding(X_)
        X_ = torch.reshape(X_, [bs, num_node, tlen, -1])
        X_ = torch.transpose(X_, 1, 2)
        A_input_ = X_

        MPNN_output1 = self.MPNN1(A_input_)
        MPNN_output2 = self.MPNN2(A_input_)

        features1 = torch.reshape(MPNN_output1, [bs, -1])
        features2 = torch.reshape(MPNN_output2, [bs, -1])

        features = torch.cat([features1, features2], -1)
        features = self.fc(features)

        return features


class STGNNRUL(BaseRULModel):
    """
    时空图神经网络用于RUL预测（框架适配版本）
    
    继承自BaseRULModel，适配CMAPSS数据集
    注意：STGNN的参数需要从config中传入
    """
    def __init__(self, patch_size=5, time_denpen_len=10, conv_kernel=2, 
                 conv_out=7, num_windows=14, lstmout_dim=32, lstmhidden_dim=8,
                 hidden_dim=8, moving_window=None, stride=None, 
                 pool_choice='mean', decay=0.7):
        super().__init__()
        
        # CMAPSS数据集配置
        self.patch_size = patch_size
        self.time_denpen_len = time_denpen_len
        self.num_sensor = 14  # 固定为14个传感器
        
        # 模型超参数（来自原始STGNN论文）
        self.hidden_dim = hidden_dim
        self.lstmhidden_dim = lstmhidden_dim
        self.lstmout_dim = lstmout_dim
        self.conv_kernel = conv_kernel
        self.conv_out = conv_out
        
        # 图卷积参数
        self.moving_window = moving_window if moving_window is not None else [2, 2]
        self.stride = stride if stride is not None else [1, 2]
        self.decay = decay
        self.pooling_choice = pool_choice
        self.num_windows = num_windows
        
        # 构建网络
        self.model = FC_STGNN_RUL(
            indim_fea=1,
            Conv_out=self.conv_out,
            lstmhidden_dim=self.lstmhidden_dim,
            lstmout_dim=self.lstmout_dim,
            conv_kernel=self.conv_kernel,
            hidden_dim=self.hidden_dim,
            time_length=self.time_denpen_len,
            num_node=self.num_sensor,
            num_windows=self.num_windows,
            moving_window=self.moving_window,
            stride=self.stride,
            decay=self.decay,
            pooling_choice=self.pooling_choice,
            n_class=1  # RUL是回归任务，输出1个值
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, time_denpen_len, patch_size, num_sensor) 
               来自CMAPSS数据集的数据格式
        
        Returns:
            (batch_size, 1): RUL预测
        """
        # 输入形状: (batch_size, time_denpen_len, patch_size, num_sensor)
        # STGNN需要: (batch_size, time_length, num_nodes, dimension)
        # 其中 time_length=time_denpen_len, num_nodes=num_sensor, dimension=patch_size
        
        # x已经是正确的形状，直接使用
        # (batch_size, time_denpen_len, patch_size, num_sensor) 
        # -> (batch_size, time_denpen_len, num_sensor, patch_size)
        batch_size, time_len, patch_size, num_sensor = x.shape
        x = x.transpose(2, 3)  # 交换patch_size和num_sensor维度
        
        # STGNN前向传播
        out = self.model(x)
        
        # 输出归一化到[0,1]范围
        out = torch.sigmoid(out)
        
        # 压缩最后一维，从 (batch, 1) -> (batch,)
        out = out.squeeze(-1)
        
        return out
    
    def get_model_name(self):
        """返回模型名称"""
        return 'STGNN-RUL'

