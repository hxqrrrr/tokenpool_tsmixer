# PhasePool-TokenMixer RUL Prediction

航空发动机剩余使用寿命（RUL）预测框架 - 基于 PhasePool 自适应序列压缩和 TokenMixer 时空混合

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## 📋 目录

- [快速开始](#快速开始)
- [训练模型](#训练模型)
- [项目结构](#项目结构)
- [最优配置](#最优配置)
- [模型架构](#模型架构)

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**依赖包**：`torch` `numpy` `pandas` `matplotlib` `scikit-learn` `scipy`

### 2. 准备数据

将 C-MAPSS 数据集放在 `CMAPSSData/` 目录下（仓库已包含数据文件）

---

## 💻 训练模型

### 使用最佳配置训练

针对每个数据集，我们提供了经过优化的最佳配置文件：

```bash
# FD001 数据集
python main.py --config experiments/configs/fd001_best.json

# FD002 数据集
python main.py --config experiments/configs/fd002_best.json

# FD003 数据集
python main.py --config experiments/configs/fd003_best.json

# FD004 数据集
python main.py --config experiments/configs/fd004_best.json
```

### 命令行参数训练

也可以通过命令行参数直接训练：

```bash
python main.py \
  --dataset FD001 \
  --window_sample 25 \
  --num_tokens 12 \
  --token_dim 112 \
  --temperature 1.5 \
  --hidden_dim 96 \
  --num_blocks 3 \
  --lr 0.0008 \
  --epochs 50 \
  --batch_size 128
```

## 📁 项目结构

```
PhasePool-TokenMixer/
├── main.py                          # 训练入口
├── trainer.py                       # 训练器
├── dataset.py                       # 数据加载
├── utils.py                         # 工具函数
├── requirements.txt                 # 依赖列表
│
├── models/                          # 模型定义
│   ├── __init__.py
│   ├── phasepool_tokenmixer.py     # PhasePool-TokenMixer 模型
│   └── base_model.py               # 基础模型类
│
├── experiments/                     # 实验管理
│   ├── __init__.py
│   ├── experiment_manager.py       # 实验管理器
│   ├── config_loader.py            # 配置加载器
│   └── configs/                    # 配置文件
│       ├── fd001_best.json         # FD001 最佳配置
│       ├── fd002_best.json         # FD002 最佳配置
│       ├── fd003_best.json         # FD003 最佳配置
│       └── fd004_best.json         # FD004 最佳配置
│
└── CMAPSSData/                      # C-MAPSS 数据集
    ├── train_FD001.txt
    ├── test_FD001.txt
    ├── RUL_FD001.txt
    └── ...
```

---

## 🏗️ 模型架构

### PhasePool-TokenMixer

本框架实现了 PhasePool-TokenMixer 模型：

1. **PhasePoolTokenMixerRUL**：主模型，结合 PhasePool 序列压缩和 TokenMixer 时空混合

### 核心组件

#### PhasePool 模块
- 自适应序列压缩，将长序列压缩为固定数量的 Token
- 基于注意力机制的软聚类
- 温度参数控制注意力分布的锐度

#### TokenMixer 模块
- 时间混合（Temporal Mixing）：在时间维度上混合信息
- 特征混合（Feature Mixing）：在特征维度上混合信息
- 残差连接和层归一化保证训练稳定性

### 评估指标

- **RMSE**：均方根误差
- **Score**：非对称评分函数（惩罚晚预测）
  ```
  早预测（d < 0）: exp(-d/13) - 1
  晚预测（d ≥ 0）: exp(d/10) - 1
  d = 预测RUL - 真实RUL
  ```

---

## 📝 训练输出

训练过程中会自动创建实验目录并保存：

```
experiments/runs/<exp_name>_<timestamp>/
├── checkpoints/              # 最佳模型检查点
│   └── best_model_epoch{epoch:03d}_score{score:.0f}.pth
├── logs/                     # 训练日志
│   └── training.log
├── results/                  # 测试结果
│   ├── test_predictions.csv
│   └── test_predictions.png
└── config/                   # 配置备份
    └── config.json
```

---

## 📖 C-MAPSS 数据集

C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) 是 NASA 提供的涡扇发动机退化模拟数据集。

### 数据集特点

| 数据集 | 训练引擎 | 测试引擎 | 工况数 | 故障模式 |
|--------|----------|----------|--------|----------|
| FD001  | 100      | 100      | 1      | 1        |
| FD002  | 260      | 259      | 6      | 1        |
| FD003  | 100      | 100      | 1      | 2        |
| FD004  | 249      | 248      | 6      | 2        |

### 传感器信号

数据集包含 21 个传感器信号和 3 个操作设置，用于监测发动机的健康状态。
