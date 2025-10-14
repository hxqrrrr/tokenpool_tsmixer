# TokenPool-TSMixer 实验框架

这是一个专为 TokenPool-TSMixer RUL 预测模型设计的实验管理框架，支持JSON配置、批量实验、自动记录和可视化。

---

## 📁 目录结构

```
experiments/
├── README.md                    # 本文档
├── __init__.py                  # 模块初始化
├── config_loader.py             # JSON配置加载器
├── experiment_manager.py        # 实验管理器
├── visualize.py                 # 可视化工具
├── configs/                     # 实验配置文件
│   ├── single_experiment.json   # 单实验配置示例
│   ├── batch_temperature_sweep.json  # 温度扫描批量实验
│   └── batch_window_sweep.json       # 窗口扫描批量实验
└── runs/                        # 实验运行记录（自动生成）
    └── <experiment_name>_<timestamp>/
        ├── checkpoints/         # 模型检查点
        ├── logs/                # 训练日志
        ├── results/             # 预测结果和可视化
        ├── config/              # 实验配置备份
        └── experiment_summary.json  # 实验总结
```

---

## 🚀 快速开始

### 1. 运行单个实验

```bash
# 使用配置文件
python main.py --config experiments/configs/single_experiment.json

# 快速实验（使用默认配置）
python main.py --quick --dataset FD001

# 快速实验并覆盖参数
python main.py --quick --dataset FD002 --temperature 1.8 --lr 0.0005
```

### 2. 批量实验

```bash
# 运行温度扫描实验（4个实验）
python main.py --batch experiments/configs/batch_temperature_sweep.json

# 运行窗口扫描实验
python main.py --batch experiments/configs/batch_window_sweep.json
```

### 3. 可视化结果

```bash
# 查看训练历史
python experiments/visualize.py experiments/runs/temp_1.6_20241014_123456 --plot history

# 查看预测结果
python experiments/visualize.py experiments/runs/temp_1.6_20241014_123456 --plot predictions

# 查看注意力权重
python experiments/visualize.py experiments/runs/temp_1.6_20241014_123456 --plot attention

# 生成所有图表
python experiments/visualize.py experiments/runs/temp_1.6_20241014_123456 --plot all
```

---

## 📝 配置文件格式

### 单实验配置

```json
{
  "experiment_name": "baseline_fd001",
  "dataset_name": "FD001",
  "description": "Baseline experiment",
  
  "data_params": {
    "window_sample": 30,
    "patch_size": 5,
    "time_denpen_len": 6,
    "max_rul": 125
  },
  
  "model_params": {
    "num_tokens": 10,
    "token_dim": 128,
    "num_heads": 4,
    "temperature": 1.5,
    "attn_dropout": 0.1,
    "use_pos_encoding": true,
    "hidden_dim": 64,
    "num_blocks": 4,
    "dropout": 0.1
  },
  
  "training_params": {
    "epochs": 50,
    "batch_size": 128,
    "lr": 0.001,
    "weight_decay": 1e-5,
    "lr_scheduler": true,
    "patience": 10
  },
  
  "output": {
    "save_checkpoint": true,
    "save_predictions": true,
    "save_attention_weights": false
  }
}
```

### 批量实验配置

```json
{
  "description": "Batch experiments for parameter sweep",
  
  "global_settings": {
    "dataset_name": "FD002",
    "data_params": { ... },
    "training_params": { ... }
  },
  
  "experiments": [
    {
      "experiment_name": "exp001",
      "description": "Test config 1",
      "model_params": { "temperature": 1.4 }
    },
    {
      "experiment_name": "exp002",
      "description": "Test config 2",
      "model_params": { "temperature": 1.6 }
    }
  ]
}
```

**说明**：
- `global_settings` 中的参数会应用到所有实验
- 每个实验的参数会覆盖 `global_settings`

---

## ⚙️ 可调参数说明

### 数据参数 (`data_params`)

| 参数 | 说明 | 默认值 | 范围 |
|------|------|--------|------|
| `window_sample` | 时间窗口长度 | 30/50 | 20-100 |
| `patch_size` | Patch大小 | 5 | 5-10 |
| `time_denpen_len` | Patch数量 | 6/10 | 4-20 |
| `max_rul` | RUL截断值 | 125 | 100-150 |

**注意**：`window_sample = patch_size × time_denpen_len`

### 模型参数 (`model_params`)

| 参数 | 说明 | 默认值 | 范围 | 重要性 |
|------|------|--------|------|--------|
| `num_tokens` | 压缩后token数量 | 10/16 | 8-24 | ⭐⭐⭐⭐ |
| `token_dim` | Token特征维度 | 64/128 | 32-256 | ⭐⭐⭐⭐⭐ |
| `num_heads` | 多头注意力头数 | 4 | 2-8 | ⭐⭐⭐ |
| `temperature` | 注意力温度参数 | 1.5-1.8 | 1.0-3.0 | ⭐⭐⭐⭐⭐ |
| `attn_dropout` | 注意力dropout率 | 0.1-0.15 | 0.0-0.3 | ⭐⭐⭐ |
| `use_pos_encoding` | 是否使用位置编码 | true | true/false | ⭐⭐⭐ |
| `hidden_dim` | TSMixer隐藏层维度 | 64/96 | 32-256 | ⭐⭐⭐⭐ |
| `num_blocks` | TSMixer层数 | 3-4 | 2-8 | ⭐⭐⭐ |
| `dropout` | 主dropout率 | 0.1-0.15 | 0.0-0.3 | ⭐⭐⭐ |

**关键参数**：
- **temperature**：最敏感参数！0.1变化可导致20分性能差异
- **token_dim**：不建议低于64，会严重影响性能
- **num_tokens**：平衡压缩比和信息保留

### 训练参数 (`training_params`)

| 参数 | 说明 | 默认值 | 范围 |
|------|------|--------|------|
| `epochs` | 最大训练轮数 | 50 | 30-100 |
| `batch_size` | 批次大小 | 128 | 64-256 |
| `lr` | 初始学习率 | 0.0008-0.001 | 1e-4 to 1e-3 |
| `weight_decay` | L2正则化系数 | 1e-5 to 2e-5 | 0-1e-4 |
| `lr_scheduler` | 是否使用学习率调度 | true | true/false |
| `patience` | Early stopping耐心值 | 10 | 5-20 |

---

## 📊 实验结果管理

### 查看所有实验

```python
from experiments import list_experiments
import pandas as pd

# 列出所有实验
df = list_experiments()
print(df)

# 输出示例：
#   experiment_name  status  start_time  duration  rmse    score  
# 0  temp_1.6        completed  2024-10-14  15m 23s  12.62  618.23
# 1  temp_1.8        completed  2024-10-14  16m 45s  13.05  629.87
```

### 加载实验总结

```python
from experiments import load_experiment_summary

summary = load_experiment_summary('experiments/runs/temp_1.6_20241014_123456')
print(f"RMSE: {summary['best_metrics']['rmse']:.2f}")
print(f"Score: {summary['best_metrics']['score']:.2f}")
```

### 对比多个实验

```python
from experiments.visualize import compare_experiments

exp_dirs = [
    'experiments/runs/temp_1.4_20241014_123456',
    'experiments/runs/temp_1.6_20241014_134567',
    'experiments/runs/temp_1.8_20241014_145678'
]

# 对比RMSE
compare_experiments(exp_dirs, metric='rmse', save_path='comparison_rmse.png')

# 对比Score
compare_experiments(exp_dirs, metric='score', save_path='comparison_score.png')
```

---

## 🎯 推荐实验流程

### 1. 基线实验

首先在所有数据集上运行基线配置：

```bash
python main.py --quick --dataset FD001
python main.py --quick --dataset FD002
python main.py --quick --dataset FD003
python main.py --quick --dataset FD004
```

### 2. 参数扫描

针对目标数据集进行关键参数扫描：

**Temperature扫描**（最重要）：
```bash
python main.py --batch experiments/configs/batch_temperature_sweep.json
```

**窗口长度扫描**：
```bash
python main.py --batch experiments/configs/batch_window_sweep.json
```

### 3. 最优组合

根据扫描结果，创建最优组合配置并验证：

```json
{
  "experiment_name": "optimal_fd002",
  "model_params": {
    "temperature": 1.6,      // 从温度扫描得出
    "num_tokens": 16,         // 从token扫描得出
    "token_dim": 64,
    "hidden_dim": 96,
    "num_blocks": 3
  }
}
```

### 4. 多轮验证

用最优配置运行多次（不同随机种子）：

```bash
python main.py --config optimal_config.json --seed 42
python main.py --config optimal_config.json --seed 123
python main.py --config optimal_config.json --seed 456
```

---

## 📈 性能基准

### C-MAPSS 预期性能

| 数据集 | 工况 | 故障模式 | 目标RMSE | 目标Score |
|--------|------|----------|----------|-----------|
| FD001  | 1    | 1        | < 11     | < 180     |
| FD002  | 6    | 1        | < 13     | < 620     |
| FD003  | 1    | 2        | < 11.5   | < 180     |
| FD004  | 6    | 2        | < 13.5   | < 770     |

### 关键超参数建议

| 数据集 | Temperature | Num Tokens | Token Dim | Hidden Dim |
|--------|-------------|------------|-----------|------------|
| FD001  | 1.5         | 10         | 128       | 64         |
| FD002  | **1.6**     | 16         | 64        | 96         |
| FD003  | 1.6         | 12         | 160       | 96         |
| FD004  | **1.8**     | 16         | 64        | 64-96      |

**规律**：工况越复杂，temperature越大（注意力越平滑）

---

## 🛠️ 高级功能

### 1. 自定义配置模板生成

```python
from experiments.config_loader import create_config_template, create_batch_config_template

# 生成单实验模板
create_config_template('my_config.json')

# 生成批量实验模板
create_batch_config_template('my_batch.json')
```

### 2. 编程式运行实验

```python
from main import run_single_experiment

config = {
    'experiment_name': 'custom_exp',
    'dataset_name': 'FD002',
    'data_params': { ... },
    'model_params': { ... },
    'training_params': { ... }
}

results = run_single_experiment(config)
print(f"RMSE: {results['rmse']:.2f}")
print(f"Score: {results['score']:.2f}")
```

### 3. 实验管理器高级用法

```python
from experiments import ExperimentManager

# 创建实验管理器
exp_manager = ExperimentManager('my_experiment')

# 保存配置
exp_manager.save_config(config)

# 开始实验
exp_manager.start_experiment()

# 记录epoch
exp_manager.log_epoch(epoch=1, metrics={
    'train_loss': 0.5,
    'val_loss': 0.6,
    'test_rmse': 12.5,
    'test_score': 650.0
})

# 保存检查点
exp_manager.save_checkpoint(model, optimizer, epoch=10, metrics={})

# 保存预测
exp_manager.save_predictions(predictions, targets, dataset_split='test')

# 结束实验
exp_manager.end_experiment(status='completed')
```

---

## 📌 注意事项

1. **实验命名**：使用描述性名称，如 `temp_1.6_fd002` 而不是 `exp001`
2. **配置备份**：每次实验的配置会自动备份到 `runs/<exp>/config/`
3. **磁盘空间**：检查点文件可能较大，定期清理旧实验
4. **随机种子**：使用 `--seed` 参数确保实验可复现
5. **Early Stopping**：默认patience=10，根据数据集调整

---

## 🤝 贡献

如需添加新功能或改进，请提交Pull Request。

---

**最后更新**：2024年10月

