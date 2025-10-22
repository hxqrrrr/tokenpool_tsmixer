"""
Checkpoint详细预测分析工具
对指定checkpoint在测试集上进行详细预测分析，输出CSV表格和可视化图表

用法: 
    python predict_detail.py <checkpoint_path> [--output-dir OUTPUT_DIR] [--dpi DPI] [--figsize WIDTH,HEIGHT]

示例:
    python predict_detail.py experiments/runs/baseline_fd002_20251022_125017/checkpoints/best_model_epoch020_score554.pth
    python predict_detail.py checkpoint.pth --output-dir ./analysis --dpi 150
"""
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# 导入必要的模块
from dataset import CMAPSSDataset
from models.tokenpool_tsmixer import TokenPoolTSMixerRUL
from models.tsmixer import TSMixerRUL
from utils import scoring_function


def create_model_from_checkpoint(checkpoint: dict):
    """根据检查点参数创建模型"""
    model_params = checkpoint['model_params']
    data_params = checkpoint['data_params']
    model_name = model_params.get('model_name', 'TokenPoolTSMixerRUL')
    
    if model_name == 'TokenPoolTSMixerRUL':
        model = TokenPoolTSMixerRUL(
            patch_size=data_params.get('patch_size', 5),
            time_denpen_len=data_params.get('time_denpen_len', 6),
            num_sensor=data_params.get('num_sensor', 14),
            num_tokens=model_params.get('num_tokens', 10),
            token_dim=model_params.get('token_dim', 128),
            num_heads=model_params.get('num_heads', 4),
            temperature=model_params.get('temperature', 1.5),
            attn_dropout=model_params.get('attn_dropout', 0.1),
            use_pos_encoding=model_params.get('use_pos_encoding', True),
            hidden_dim=model_params.get('hidden_dim', 64),
            num_blocks=model_params.get('num_blocks', 4),
            dropout=model_params.get('dropout', 0.1)
        )
    elif model_name == 'TSMixerRUL':
        model = TSMixerRUL(
            patch_size=data_params.get('patch_size', 5),
            time_denpen_len=data_params.get('time_denpen_len', 6),
            num_sensor=data_params.get('num_sensor', 14),
            hidden_dim=model_params.get('hidden_dim', 64),
            num_blocks=model_params.get('num_blocks', 4),
            dropout=model_params.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"未知的模型类型: {model_name}")
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def create_dataset_from_checkpoint(checkpoint: dict):
    """根据检查点参数创建数据集"""
    data_params = checkpoint['data_params']
    
    dataset = CMAPSSDataset(
        data_root='./CMAPSSData',
        dataset_name=data_params.get('dataset_name', 'FD001'),
        max_rul=data_params.get('max_rul', 125),
        seq_len=data_params.get('patch_size', 5),
        time_denpen_len=data_params.get('time_denpen_len', 6),
        window_sample=data_params.get('window_sample', 30)
    )
    
    return dataset


def predict_on_test(checkpoint_path: Path, device: torch.device):
    """
    在测试集上进行预测
    
    Returns:
        tuple: (predictions_rul, actual_rul, rmse, score, checkpoint, dataset)
    """
    print(f"\n{'='*80}")
    print(f"加载Checkpoint: {checkpoint_path.name}")
    print(f"{'='*80}\n")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 创建模型
    print("[1/4] 创建模型...")
    model = create_model_from_checkpoint(checkpoint)
    model = model.to(device)
    print(f"  模型: {checkpoint['model_params'].get('model_name')}")
    
    # 创建数据集
    print("[2/4] 加载数据集...")
    dataset = create_dataset_from_checkpoint(checkpoint)
    print(f"  数据集: {dataset.dataset_name}")
    
    # 获取测试数据
    print("[3/4] 获取测试数据...")
    test_data = dataset.get_test_data()
    test_x = torch.FloatTensor(test_data['x']).to(device)
    test_y = torch.FloatTensor(test_data['y']).to(device)
    print(f"  测试样本数: {len(test_x)}")
    
    # 推理
    print("[4/4] 执行推理...")
    with torch.no_grad():
        predictions = model(test_x)
    
    # 反归一化到实际RUL值
    predictions_rul = (predictions * dataset.max_rul).cpu().numpy()
    actual_rul = (test_y * dataset.max_rul).cpu().numpy()
    
    # 计算性能指标
    mse = nn.functional.mse_loss(predictions, test_y)
    rmse = torch.sqrt(mse) * dataset.max_rul
    score = scoring_function(predictions, test_y, dataset.max_rul)
    
    print(f"  RMSE: {rmse.item():.4f}")
    print(f"  Score: {score.item():.2f}")
    print()
    
    return predictions_rul, actual_rul, rmse.item(), score.item(), checkpoint, dataset


def save_predictions_csv(predictions_rul, actual_rul, output_path: Path):
    """保存预测结果到CSV"""
    print(f"[保存CSV] {output_path}")
    
    # 计算误差
    errors = predictions_rul - actual_rul
    abs_errors = np.abs(errors)
    
    # 创建DataFrame
    results_df = pd.DataFrame({
        'Sample_Index': range(len(predictions_rul)),
        'Predicted_RUL': predictions_rul.flatten(),
        'Actual_RUL': actual_rul.flatten(),
        'Error': errors.flatten(),
        'Absolute_Error': abs_errors.flatten()
    })
    
    # 保存
    results_df.to_csv(output_path, index=False)
    print(f"  已保存 {len(results_df)} 条记录\n")
    
    return results_df


def plot_predictions(predictions_rul, actual_rul, checkpoint_name: str, 
                    rmse: float, score: float, output_path: Path,
                    figsize=(12, 4), dpi=300):
    """生成预测对比可视化图表"""
    print(f"[生成图表] {output_path}")
    
    sample_indices = np.arange(len(predictions_rul))
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 实际RUL - 蓝色圆点虚线
    ax.plot(sample_indices, actual_rul.flatten(), 'o--', 
            color='blue', label='Actual RUL', 
            markersize=5, linewidth=1, alpha=0.8)
    
    # 预测RUL - 红色三角实线
    ax.plot(sample_indices, predictions_rul.flatten(), '^-', 
            color='red', label='Predicted RUL',
            markersize=6, linewidth=1, alpha=0.8)
    
    # 样式设置
    ax.set_xlabel('Samples', fontsize=12)
    ax.set_ylabel('RUL', fontsize=12)
    ax.set_title(f'Prediction vs Actual - {checkpoint_name}\n'
                f'RMSE: {rmse:.4f}, Score: {score:.2f}', 
                fontsize=13)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-2, len(sample_indices) + 2)
    
    # 保存
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  图表尺寸: {figsize}, DPI: {dpi}\n")


def save_summary(checkpoint, dataset, predictions_rul, actual_rul, 
                rmse: float, score: float, output_path: Path):
    """生成统计摘要"""
    print(f"[生成摘要] {output_path}")
    
    errors = predictions_rul - actual_rul
    abs_errors = np.abs(errors)
    
    # 计算统计指标
    mae = np.mean(abs_errors)
    max_error = np.max(abs_errors)
    min_error = np.min(abs_errors)
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    median_error = np.median(errors)
    
    # 生成摘要内容
    summary_lines = [
        f"Checkpoint: {checkpoint['model_params'].get('model_name')}",
        f"Dataset: {dataset.dataset_name}",
        f"Checkpoint File: {output_path.parent.parent.name}",
        f"Total Samples: {len(predictions_rul)}",
        "",
        "Performance Metrics:",
        f"  RMSE: {rmse:.4f}",
        f"  Score: {score:.2f}",
        f"  MAE: {mae:.4f}",
        f"  Max Error: {max_error:.4f}",
        f"  Min Error: {min_error:.4f}",
        "",
        "Error Statistics:",
        f"  Mean Error: {mean_error:.4f}",
        f"  Std Error: {std_error:.4f}",
        f"  Median Error: {median_error:.4f}",
        "",
        "Model Configuration:",
    ]
    
    # 添加模型参数
    model_params = checkpoint['model_params']
    for key, value in model_params.items():
        summary_lines.append(f"  {key}: {value}")
    
    summary_lines.append("")
    summary_lines.append("Data Configuration:")
    
    # 添加数据参数
    data_params = checkpoint['data_params']
    for key, value in data_params.items():
        summary_lines.append(f"  {key}: {value}")
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"  摘要已保存\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Checkpoint详细预测分析工具 - 分析checkpoint在测试集上的预测表现',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用
  python predict_detail.py experiments/runs/baseline_fd002_20251022_125017/checkpoints/best_model_epoch020_score554.pth
  
  # 指定输出目录
  python predict_detail.py checkpoint.pth --output-dir ./analysis
  
  # 自定义图表大小和DPI
  python predict_detail.py checkpoint.pth --figsize 16,6 --dpi 150
  
  # 只预测前100个样本
  python predict_detail.py checkpoint.pth --num-samples 100
        """
    )
    
    parser.add_argument('checkpoint_path', type=str,
                       help='Checkpoint文件路径 (.pth)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录 (默认: checkpoint目录下的predictions_detail/)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='图表DPI (默认: 300)')
    parser.add_argument('--figsize', type=str, default='12,4',
                       help='图表大小 "宽,高" (默认: 12,4)')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='限制预测的样本数量 (默认: 全部样本)')
    
    args = parser.parse_args()
    
    # 解析checkpoint路径
    checkpoint_path = Path(args.checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"[错误] Checkpoint文件不存在: {checkpoint_path}")
        sys.exit(1)
    
    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_path.parent.parent / 'predictions_detail'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 解析图表大小
    try:
        figsize = tuple(map(float, args.figsize.split(',')))
        if len(figsize) != 2:
            raise ValueError
    except:
        print(f"[警告] 无效的figsize格式，使用默认值 (12, 4)")
        figsize = (12, 4)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    try:
        # 1. 执行预测
        predictions_rul, actual_rul, rmse, score, checkpoint, dataset = predict_on_test(
            checkpoint_path, device
        )
        
        # 限制样本数量（如果指定）
        if args.num_samples is not None and args.num_samples > 0:
            num_samples = min(args.num_samples, len(predictions_rul))
            predictions_rul = predictions_rul[:num_samples]
            actual_rul = actual_rul[:num_samples]
            print(f"{'='*80}")
            print(f"限制样本数量: {num_samples} / {len(dataset.get_test_data()['x'])}")
            print(f"{'='*80}\n")
        
        print(f"{'='*80}")
        print("输出结果")
        print(f"{'='*80}\n")
        print(f"输出目录: {output_dir}\n")
        
        # 2. 保存CSV
        csv_path = output_dir / 'predictions_detail.csv'
        results_df = save_predictions_csv(predictions_rul, actual_rul, csv_path)
        
        # 3. 生成可视化图表
        fig_path = output_dir / 'prediction_comparison.png'
        plot_predictions(
            predictions_rul, actual_rul, 
            checkpoint_path.name, rmse, score,
            fig_path, figsize=figsize, dpi=args.dpi
        )
        
        # 4. 生成统计摘要
        summary_path = output_dir / 'summary.txt'
        save_summary(checkpoint, dataset, predictions_rul, actual_rul, 
                    rmse, score, summary_path)
        
        print(f"{'='*80}")
        print("✓ 分析完成！")
        print(f"{'='*80}\n")
        print("生成的文件:")
        print(f"  1. CSV表格: {csv_path}")
        print(f"  2. 可视化图表: {fig_path}")
        print(f"  3. 统计摘要: {summary_path}")
        print()
        
    except Exception as e:
        print(f"\n[错误] 分析过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

