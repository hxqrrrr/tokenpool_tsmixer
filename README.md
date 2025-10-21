# TokenPool-TSMixer RUL Prediction

航空发动机剩余使用寿命（RUL）预测框架 - 基于 TokenPool 自适应序列压缩和 TSMixer 时空混合

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## 📋 目录

- [快速开始](#快速开始)
- [常用命令](#常用命令)
- [检查点管理](#检查点管理)
- [项目结构](#项目结构)
- [实验结果](#实验结果)

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**依赖包**：`torch` `numpy` `pandas` `matplotlib` `scikit-learn`

### 2. 准备数据

将 C-MAPSS 数据集放在 `CMAPSSData/` 目录下

---

## 💻 常用命令

### 训练模型

**单次实验**

```bash
python main.py --config experiments/configs/single_experiment.json
```

**批量实验**

```bash
# 温度参数扫描
python main.py --batch experiments/configs/batch_temperature_sweep.json

# 窗口长度扫描
python main.py --batch experiments/configs/batch_window_sweep.json
```

### 命令行参数

```bash
# 查看所有参数
python main.py --help

# 常用参数
--dataset FD001/FD002/FD003/FD004    # 数据集
--temperature 1.6                     # 温度
--num_tokens 16                       # Token数量
--window_sample 50                    # 窗口长度
--lr 0.0008                           # 学习率
--epochs 50                           # 训练轮数
--batch_size 128                      # 批次大小
```

---

## 📊 检查点管理

训练时自动保存每个最优模型到 `experiments/runs/<exp_name>/checkpoints/`

### 批量推理

```bash
# 完整推理（加载模型实际测试）
python inference.py experiments/runs/<exp_name>

# 保存结果到CSV
python inference.py experiments/runs/<exp_name> --save-csv

# 快速模式（仅读取保存的指标）
python inference.py experiments/runs/<exp_name> --fast
```

### 检查点文件

**命名格式**：`best_model_epoch{epoch:03d}_score{score:.0f}.pth`

**包含信息**：
- 模型权重 (`model_state_dict`)
- 性能指标 (`test_rmse`, `test_score`, `val_loss`)
- 训练参数 (`lr`, `batch_size`, `weight_decay`, `epochs`)
- 数据参数 (`dataset_name`, `window_sample`, `patch_size`, `max_rul`)
- 模型参数 (`hidden_dim`, `num_tokens`, `temperature`, `dropout`)

---

## 📁 项目结构

```
rebuild/
├── main.py                          # 训练入口
├── inference.py                     # 批量推理
├── trainer.py                       # 训练器
├── dataset.py                       # 数据加载
├── config.py                        # 配置管理
├── requirements.txt                 # 依赖列表
│
├── models/                          # 模型
│   ├── tokenpool_tsmixer.py
│   ├── tsmixer.py
│   └── base_model.py
│
├── experiments/                     # 实验管理
│   ├── experiment_manager.py
│   ├── config_loader.py
│   ├── configs/                     # 配置文件
│   └── runs/                        # 实验结果
│       └── <exp_name>_<timestamp>/
│           ├── checkpoints/         # 模型检查点 ⭐
│           ├── logs/                # 训练日志
│           ├── results/             # 测试结果
│           └── config/              # 配置备份
│
└── CMAPSSData/                      # 数据集
```

---

## 📈 实验结果

### C-MAPSS 基准测试

| 数据集 | 工况 | 故障 | RMSE ↓ | Score ↓ | 参数量 |
|--------|------|------|--------|---------|--------|
| FD001  | 1    | 1    | 10.90  | 177.11  | 180K   |
| FD002  | 6    | 1    | 12.62  | 594.33  | 194K   |
| FD003  | 1    | 2    | 11.06  | 176.02  | 280K   |
| FD004  | 6    | 2    | 13.26  | 760.47  | 194K   |

**Score 计算**（非对称惩罚）：
```
早预测（d < 0）: exp(-d/13) - 1
晚预测（d ≥ 0）: exp(d/10) - 1
d = 预测RUL - 真实RUL
```

### 最优配置

**FD001** (单工况)
```json
{
  "window_sample": 30,
  "num_tokens": 10,
  "temperature": 1.5,
  "hidden_dim": 64,
  "num_blocks": 4
}
```

**FD002** (多工况)
```json
{
  "window_sample": 50,
  "num_tokens": 16,
  "token_dim": 64,
  "temperature": 1.6,
  "hidden_dim": 96,
  "num_blocks": 3,
  "dropout": 0.15
}
```

---

## 📚 相关文档

- [检查点详细指南](CHECKPOINT_GUIDE.md)
- [实验框架说明](experiments/README.md)
