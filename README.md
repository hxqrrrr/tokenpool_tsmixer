# TokenPool-TSMixer RUL Prediction Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

🚀 **TokenPool-TSMixer**: 专注于航空发动机剩余寿命(RUL)预测的实验框架

**核心特性**：JSON配置驱动 | 批量实验管理 | 自动记录可视化 | 专注TokenPool-TSMixer单一模型

---

## 📋 目录

- [项目简介](#项目简介)
- [核心创新](#核心创新)
- [模型架构](#模型架构)
- [实验结果](#实验结果)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [详细使用](#详细使用)
- [引用](#引用)
- [许可证](#许可证)

---

## 🎯 项目简介

本项目针对**航空发动机剩余使用寿命（RUL）预测**任务，提出了一种基于 **TokenPool 自适应序列压缩** 和 **TSMixer 时空混合** 的深度学习架构。在 NASA C-MAPSS 数据集上取得了优异性能，特别是在多工况场景下展现出强大的泛化能力。

### 主要特性

- ✅ **高效压缩**：将 50 步时间序列压缩至 16 个 tokens，降低 68% 计算量
- ✅ **自适应注意力**：可学习的 Query 向量自动识别关键退化时段
- ✅ **多工况泛化**：在 FD004（6 工况 + 多故障）上 Score 达到 760
- ✅ **参数高效**：仅 194K 参数，相比 Transformer 减少 80%
- ✅ **工程友好**：<10ms 推理延迟，支持边缘部署

---

## 💡 核心创新

### 1. TokenPool 自适应压缩模块

基于 **Cross-Attention** 机制的可学习池化层：

```
输入序列 [B, T=50, C=14] 
    ↓ 
Cross-Attention (可学习 Query)
    ↓
压缩 Token 序列 [B, N=16, D=64]
```

**关键技术**：
- 🔥 **可学习 Query 向量**：每个 Query 作为"退化探针"，自适应聚合关键信息
- 🔥 **温度调控注意力**：τ 参数控制注意力平滑度（单工况 τ=1.5，多工况 τ=1.8）
- 🔥 **正弦位置编码**：保留时序顺序信息

### 2. TSMixer 高效时空融合

纯 MLP 架构，实现线性复杂度 O(T)：

- **Time-Mixing**：跨时间步全局混合，捕捉退化趋势
- **Feature-Mixing**：跨传感器通道混合，建模物理关联

### 3. 工况感知归一化

按操作条件分组归一化，避免多工况数据混淆：

```python
for condition in unique_conditions:
    scaler.fit_transform(train_data[condition])
    scaler.transform(test_data[condition])
```

---

## 🏗️ 模型架构

```
┌─────────────────────────────────────────────────┐
│          输入：多传感器时序数据                    │
│            [Batch, 50, 14]                      │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│        TokenPool 自适应压缩                       │
│  • K, V = Linear(X)                             │
│  • Q = Learnable Queries [16, 64]               │
│  • Attn = softmax(Q·K^T / (√d_k · τ))          │
│  • Z = Attn·V  [Batch, 16, 64]                 │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│         TSMixer 时空混合 (3 层)                   │
│  • Time-Mixing: MLP across time                │
│  • Feature-Mixing: MLP across features         │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│            回归头 → RUL 预测                      │
│         Flatten + MLP → [Batch, 1]              │
└─────────────────────────────────────────────────┘
```

---

## 📊 实验结果

### C-MAPSS 基准测试

| 数据集 | 工况数 | 故障模式 | 训练样本 | RMSE ↓ | Score ↓ | 参数量 |
|--------|--------|----------|---------|--------|---------|--------|
| FD001  | 1      | 1        | 20,631  | **10.90** | **177.11** | 180K |
| FD002  | 6      | 1        | 53,759  | **12.62** | **594.33** | 194K |
| FD003  | 1      | 2        | 24,720  | **11.06** | **176.02** | 280K |
| FD004  | 6      | 2        | 61,249  | **13.26** | **760.47** | 194K |

**Score 计算公式**（非对称惩罚）：
```
Score = Σ { exp(-d_i/13) - 1,  if d_i < 0 (早预测)
            exp(d_i/10) - 1,   if d_i ≥ 0 (晚预测) }
其中 d_i = 预测RUL - 真实RUL
```

### 消融实验 (FD002)

| 实验配置 | Temperature | Num Tokens | Token Dim | Score | 相对变化 |
|---------|-------------|------------|-----------|-------|---------|
| **最优配置** | 1.6 | 16 | 64 | **618.23** | baseline |
| Temp=1.5 | 1.5 | 16 | 64 | 639.15 | +3.4% ⬆️ |
| Temp=2.0 | 2.0 | 16 | 64 | 644.02 | +4.2% ⬆️ |
| Tokens=18 | 1.6 | 18 | 64 | 638.68 | +3.3% ⬆️ |
| Dim=56 | 1.6 | 16 | 56 | 649.48 | +5.1% ⬆️ |

**关键发现**：
1. **Temperature 最敏感**：0.1 变化导致 20 分差异（Sweet Spot: τ=1.6）
2. **Token Dim 不可降低**：从 64→56 损失 5% 性能
3. **压缩比最优**：16 tokens（3.1×）优于 18 tokens

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (可选，用于 GPU 加速)

### 安装

```bash
# 克隆仓库
git clone https://github.com/hxqrrrr/tokenpool_tsmixer.git
cd tokenpool_tsmixer

# 安装依赖
pip install -r requirements.txt
```

### 数据准备

1. 下载 [C-MAPSS 数据集](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
2. 解压到 `CMAPSSData/` 目录：

```
CMAPSSData/
├── train_FD001.txt
├── test_FD001.txt
├── RUL_FD001.txt
├── train_FD002.txt
├── ...
```

### 运行实验

#### 方式1：快速实验（使用默认配置）

```bash
# 使用默认配置快速运行
python main.py --quick --dataset FD001

# 快速运行并覆盖参数
python main.py --quick --dataset FD002 --temperature 1.6 --lr 0.0008
```

#### 方式2：使用JSON配置文件

```bash
# 单个实验
python main.py --config experiments/configs/single_experiment.json

# 覆盖配置文件中的参数
python main.py --config experiments/configs/single_experiment.json --temperature 1.8
```

#### 方式3：批量实验

```bash
# 温度参数扫描（4个实验）
python main.py --batch experiments/configs/batch_temperature_sweep.json

# 窗口长度扫描（4个实验）
python main.py --batch experiments/configs/batch_window_sweep.json
```

### 查看实验结果

```bash
# 可视化训练历史
python experiments/visualize.py experiments/runs/<exp_name> --plot history

# 查看预测结果
python experiments/visualize.py experiments/runs/<exp_name> --plot predictions

# 生成所有图表
python experiments/visualize.py experiments/runs/<exp_name> --plot all
```

---

## 📁 项目结构

```
tokenpool_tsmixer/
├── experiments/                # 🔥 实验管理框架
│   ├── __init__.py
│   ├── config_loader.py       # JSON配置加载器
│   ├── experiment_manager.py  # 实验管理和日志
│   ├── visualize.py           # 可视化工具
│   ├── README.md              # 实验框架文档
│   ├── configs/               # 实验配置文件
│   │   ├── single_experiment.json
│   │   ├── batch_temperature_sweep.json
│   │   └── batch_window_sweep.json
│   └── runs/                  # 实验运行记录（自动生成）
│       └── <exp_name>_<timestamp>/
│           ├── checkpoints/   # 模型检查点
│           ├── logs/          # 训练日志
│           ├── results/       # 预测结果
│           ├── config/        # 配置备份
│           └── experiment_summary.json
├── dataset/                    # 数据加载
│   ├── __init__.py
│   └── cmapss_dataset.py      # C-MAPSS 数据集处理
├── models/                     # 模型定义
│   ├── __init__.py
│   ├── base_model.py          # 基类
│   ├── tokenpool_tsmixer.py   # 🔥 核心模型（专注此模型）
│   └── tsmixer.py             # TSMixer 基础架构
├── trainers/                   # 训练器
│   ├── __init__.py
│   └── rul_trainer.py         # RUL 训练流程
├── utils/                      # 工具函数
│   ├── __init__.py
│   ├── logger.py              # 日志记录
│   └── metrics.py             # 评估指标
├── main.py                     # 🚀 主入口（支持JSON配置和批量实验）
├── requirements.txt            # 依赖列表
├── .gitignore
└── README.md
```

---

## 📖 详细使用

### 配置文件

所有超参数在 `configs/config.py` 中管理：

```python
# FD002 最优配置（经过 14 次实验验证）
'FD002': {
    # TokenPool 参数
    'num_tokens': 16,          # 压缩到 16 tokens
    'token_dim': 64,           # Token 维度
    'temperature': 1.6,        # 🔥 Sweet Spot
    'attn_dropout': 0.15,
    
    # TSMixer 参数
    'hidden_dim': 96,
    'num_blocks': 3,
    'dropout': 0.15,
    
    # 训练参数
    'lr': 8e-4,
    'weight_decay': 2e-5,
    'batch_size': 128,
}
```

### 自定义模型

继承 `BaseRULModel` 实现自己的模型：

```python
from models.base_model import BaseRULModel

class MyRULModel(BaseRULModel):
    def __init__(self, config):
        super().__init__()
        # 定义网络层
        
    def forward(self, x):
        # 前向传播
        return predictions
```

### 可视化注意力权重

```python
from models.tokenpool_tsmixer import TokenPoolTSMixerRUL

model = TokenPoolTSMixerRUL(...)
output = model(x)

# 获取注意力权重 [Batch, Heads, N_tokens, T_steps]
attn_weights = model.last_attn_weights

# 可视化哪些时间步对预测贡献最大
import matplotlib.pyplot as plt
avg_attn = attn_weights.mean(dim=(0,1)).cpu().numpy()  # [N, T]
plt.imshow(avg_attn, cmap='hot', aspect='auto')
plt.xlabel('Time Steps')
plt.ylabel('Tokens')
plt.colorbar(label='Attention Weight')
plt.show()
```

---

## 🔬 技术细节

### 数据预处理流程

1. **删除常量传感器**：移除 7 个无效传感器（s1, s5, s6, s10, s16, s18, s19）
2. **RUL 截断**：`RUL = min(RUL, 125)`（NASA 标准）
3. **工况感知归一化**：按操作设置分组 MinMaxScaler
4. **滑动窗口**：
   - 训练集：密集采样（步长=1）
   - 测试集：只取最后一个窗口
5. **Patch 采样**：`[50, 14] → [10 patches, 5 steps, 14 sensors]`

### TokenPool 数学原理

**Cross-Attention 池化**：

$$
\begin{align}
\mathbf{K} &= \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V \quad \in \mathbb{R}^{T \times D} \\
\mathbf{Q} &= [\mathbf{q}_1, ..., \mathbf{q}_N] \quad \in \mathbb{R}^{N \times D} \quad \text{(可学习参数)} \\
\mathbf{A} &= \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k} \cdot \tau}\right) \quad \in \mathbb{R}^{N \times T} \\
\mathbf{Z} &= \mathbf{A} \mathbf{V} \quad \in \mathbb{R}^{N \times D}
\end{align}
$$

**温度参数作用**：
- τ → 0：稀疏注意力（类似 max-pooling）
- τ → ∞：均匀注意力（类似 avg-pooling）
- τ ≈ 1.6：最佳平衡（FD002 实验验证）

---

## 🛠️ 高级功能

### 1. 超参数搜索

使用网格搜索优化配置：

```bash
python main.py --dataset FD002 --model TokenPoolTSMixer \
    --grid_search \
    --param temperature 1.4,1.5,1.6,1.7,1.8 \
    --param num_tokens 12,16,20
```

### 2. 模型集成

多模型投票提升性能：

```python
from trainer import RULTrainer

models = [
    TokenPoolTSMixerRUL(...),
    TSMixerRUL(...),
    STGNNRUL(...)
]

predictions = [model(x) for model in models]
ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
```

### 3. 边缘部署

模型量化与导出：

```python
import torch

# 导出 ONNX
model.eval()
dummy_input = torch.randn(1, 10, 5, 14)
torch.onnx.export(model, dummy_input, "model.onnx")

# 量化为 INT8
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

---

## 📈 性能优化

### 训练加速技巧

1. **混合精度训练**：
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(x)
    loss = criterion(output, y)
scaler.scale(loss).backward()
```

2. **梯度累积**（模拟大 batch）：
```python
accumulation_steps = 4
for i, (x, y) in enumerate(dataloader):
    loss = model(x, y) / accumulation_steps
    loss.backward()
    if (i+1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

3. **数据预加载**：
```python
from torch.utils.data import DataLoader
loader = DataLoader(dataset, num_workers=4, pin_memory=True)
```

---

## 🤝 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📝 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@software{tokenpool_tsmixer_2024,
  author = {Your Name},
  title = {TokenPool-TSMixer: Adaptive Sequence Compression for RUL Prediction},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/hxqrrrr/tokenpool_tsmixer}
}
```

**相关论文**：
- **TSMixer**: Chen et al., "TSMixer: An All-MLP Architecture for Time Series Forecasting", 2023
- **Perceiver**: Jaegle et al., "Perceiver: General Perception with Iterative Attention", ICML 2021
- **C-MAPSS**: Saxena et al., "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM 2008

---

## 📄 许可证

本项目采用 [MIT License](LICENSE)。

---

## 🙏 致谢

- **NASA Ames Research Center** 提供 C-MAPSS 数据集
- **Google Research** 提供 TSMixer 架构启发
- **DeepMind** 提供 Perceiver 思想启发

---

## 📧 联系方式

- **GitHub Issues**: [https://github.com/hxqrrrr/tokenpool_tsmixer/issues](https://github.com/hxqrrrr/tokenpool_tsmixer/issues)
- **Email**: [您的邮箱]

---

## 🌟 Star History

如果觉得本项目有用，请给个 ⭐ Star！

[![Star History Chart](https://api.star-history.com/svg?repos=hxqrrrr/tokenpool_tsmixer&type=Date)](https://star-history.com/#hxqrrrr/tokenpool_tsmixer&Date)

---

**最后更新时间**: 2024年10月

