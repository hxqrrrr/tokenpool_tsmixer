"""
批量检查点推理脚本 - 对实验中所有检查点进行实际推理测试
用法: python inference.py <experiment_dir> [--save-csv] [--fast]
"""
import sys
import torch
import torch.nn as nn
import pandas as pd
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


def inference_checkpoint(checkpoint_path: Path, device: torch.device):
    """对单个检查点进行推理"""
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 创建模型
    model = create_model_from_checkpoint(checkpoint)
    model = model.to(device)
    
    # 创建数据集
    dataset = create_dataset_from_checkpoint(checkpoint)
    
    # 获取测试数据
    test_data = dataset.get_test_data()
    test_x = torch.FloatTensor(test_data['x']).to(device)
    test_y = torch.FloatTensor(test_data['y']).to(device)
    
    # 推理
    with torch.no_grad():
        predictions = model(test_x)
    
    # 计算RMSE
    mse = nn.functional.mse_loss(predictions, test_y)
    rmse = torch.sqrt(mse) * dataset.max_rul
    
    # 计算Score
    score = scoring_function(predictions, test_y, dataset.max_rul)
    
    return rmse.item(), score.item()


def batch_inference(experiment_dir: Path, save_csv: bool = False, fast_mode: bool = False):
    """批量推理实验中的所有检查点"""
    checkpoint_dir = experiment_dir / 'checkpoints'
    
    if not checkpoint_dir.exists():
        print(f"[错误] 检查点目录不存在: {checkpoint_dir}")
        return None, None
    
    # 获取所有检查点文件
    checkpoint_files = sorted(checkpoint_dir.glob('*.pth'))
    
    if not checkpoint_files:
        print(f"[错误] 在 {checkpoint_dir} 中没有找到检查点文件")
        return None, None
    
    print(f"\n{'='*100}")
    print(f"[批量推理] 实验: {experiment_dir.name}")
    print(f"{'='*100}\n")
    print(f"找到 {len(checkpoint_files)} 个检查点")
    print(f"检查点目录: {checkpoint_dir}")
    
    if fast_mode:
        print(f"模式: 快速模式（仅读取保存的指标）\n")
    else:
        print(f"模式: 完整推理模式（实际加载模型并测试）\n")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not fast_mode:
        print(f"使用设备: {device}\n")
    
    # 分析所有检查点
    results = []
    for i, ckpt_path in enumerate(checkpoint_files, 1):
        try:
            print(f"[{i}/{len(checkpoint_files)}] 处理: {ckpt_path.name}")
            
            # 加载检查点
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            
            # 提取保存的信息
            metrics = checkpoint.get('metrics', {})
            model_params = checkpoint.get('model_params', {})
            data_params = checkpoint.get('data_params', {})
            training_params = checkpoint.get('training_params', {})
            
            saved_rmse = metrics.get('test_rmse', 0.0)
            saved_score = metrics.get('test_score', 0.0)
            
            # 如果不是快速模式，进行实际推理
            if not fast_mode:
                inferred_rmse, inferred_score = inference_checkpoint(ckpt_path, device)
                
                # 计算差异
                rmse_diff = abs(inferred_rmse - saved_rmse)
                score_diff = abs(inferred_score - saved_score)
                
                print(f"  保存: RMSE={saved_rmse:.4f}, Score={saved_score:.2f}")
                print(f"  推理: RMSE={inferred_rmse:.4f}, Score={inferred_score:.2f}")
                if rmse_diff > 0.01 or score_diff > 0.1:
                    print(f"  [注意] 差异: RMSE差={rmse_diff:.4f}, Score差={score_diff:.2f}")
            else:
                inferred_rmse = saved_rmse
                inferred_score = saved_score
                print(f"  RMSE={saved_rmse:.4f}, Score={saved_score:.2f}")
            
            result = {
                '检查点文件': ckpt_path.name,
                'Epoch': checkpoint.get('epoch', 'N/A'),
                'RMSE(推理)': inferred_rmse,
                'Score(推理)': inferred_score,
                'RMSE(保存)': saved_rmse,
                'Score(保存)': saved_score,
                'Val Loss': metrics.get('val_loss', 0.0),
                '保存时间': checkpoint.get('timestamp', 'N/A'),
                '模型名称': model_params.get('model_name', 'N/A'),
                '数据集': data_params.get('dataset_name', 'N/A'),
                'Temperature': model_params.get('temperature', 'N/A'),
                'Num Tokens': model_params.get('num_tokens', 'N/A'),
                'Hidden Dim': model_params.get('hidden_dim', 'N/A'),
                'Num Blocks': model_params.get('num_blocks', 'N/A'),
                'Learning Rate': training_params.get('lr', 'N/A'),
                'Batch Size': training_params.get('batch_size', 'N/A'),
            }
            results.append(result)
            print()
            
        except Exception as e:
            print(f"[警告] 处理 {ckpt_path.name} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results:
        print("[错误] 没有成功处理的检查点")
        return None, None
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 按推理Score排序（越小越好）
    df_sorted = df.sort_values('Score(推理)')
    
    # 找到最优模型
    best_idx = df_sorted['Score(推理)'].idxmin()
    best_result = df_sorted.loc[best_idx]
    
    # 显示结果表格
    print(f"{'='*100}")
    print("[所有检查点性能对比] 按推理Score排序")
    print(f"{'='*100}\n")
    
    # 创建显示用的DataFrame
    display_df = df_sorted[[
        '检查点文件', 'Epoch', 
        'RMSE(推理)', 'Score(推理)',
        'RMSE(保存)', 'Score(保存)',
        'Val Loss'
    ]].copy()
    
    # 格式化数值
    display_df['RMSE(推理)'] = display_df['RMSE(推理)'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
    display_df['Score(推理)'] = display_df['Score(推理)'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
    display_df['RMSE(保存)'] = display_df['RMSE(保存)'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
    display_df['Score(保存)'] = display_df['Score(保存)'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
    display_df['Val Loss'] = display_df['Val Loss'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
    
    # 设置pandas显示选项
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    
    print(display_df.to_string(index=False))
    
    # 显示最优结果
    print(f"\n{'='*100}")
    print("[最优模型] 详细信息")
    print(f"{'='*100}\n")
    print(f"检查点文件: {best_result['检查点文件']}")
    print(f"Epoch: {best_result['Epoch']}")
    print(f"保存时间: {best_result['保存时间']}")
    
    print(f"\n性能指标:")
    print(f"  RMSE (推理): {best_result['RMSE(推理)']:.4f}")
    print(f"  Score (推理): {best_result['Score(推理)']:.2f}")
    print(f"  RMSE (保存): {best_result['RMSE(保存)']:.4f}")
    print(f"  Score (保存): {best_result['Score(保存)']:.2f}")
    print(f"  Val Loss: {best_result['Val Loss']:.4f}")
    
    # 检查差异
    rmse_diff = abs(best_result['RMSE(推理)'] - best_result['RMSE(保存)'])
    score_diff = abs(best_result['Score(推理)'] - best_result['Score(保存)'])
    if not fast_mode:
        print(f"\n一致性检查:")
        print(f"  RMSE 差异: {rmse_diff:.4f}")
        print(f"  Score 差异: {score_diff:.2f}")
        if rmse_diff < 0.01 and score_diff < 0.1:
            print(f"  状态: [通过] 推理结果与保存结果一致")
        else:
            print(f"  状态: [警告] 推理结果与保存结果存在差异")
    
    print(f"\n模型配置:")
    print(f"  模型名称: {best_result['模型名称']}")
    print(f"  数据集: {best_result['数据集']}")
    if best_result['Temperature'] != 'N/A':
        print(f"  Temperature: {best_result['Temperature']}")
    if best_result['Num Tokens'] != 'N/A':
        print(f"  Num Tokens: {best_result['Num Tokens']}")
    if best_result['Hidden Dim'] != 'N/A':
        print(f"  Hidden Dim: {best_result['Hidden Dim']}")
    if best_result['Num Blocks'] != 'N/A':
        print(f"  Num Blocks: {best_result['Num Blocks']}")
    
    print(f"\n训练配置:")
    if best_result['Learning Rate'] != 'N/A':
        print(f"  Learning Rate: {best_result['Learning Rate']}")
    if best_result['Batch Size'] != 'N/A':
        print(f"  Batch Size: {best_result['Batch Size']}")
    
    # 统计信息
    print(f"\n{'='*100}")
    print("[统计摘要]")
    print(f"{'='*100}\n")
    print(f"总检查点数: {len(results)}")
    print(f"\nScore 指标 (推理):")
    print(f"  最优: {df_sorted['Score(推理)'].min():.2f}")
    print(f"  最差: {df_sorted['Score(推理)'].max():.2f}")
    print(f"  平均: {df_sorted['Score(推理)'].mean():.2f}")
    print(f"  标准差: {df_sorted['Score(推理)'].std():.2f}")
    print(f"\nRMSE 指标 (推理):")
    print(f"  最优: {df_sorted['RMSE(推理)'].min():.4f}")
    print(f"  最差: {df_sorted['RMSE(推理)'].max():.4f}")
    print(f"  平均: {df_sorted['RMSE(推理)'].mean():.4f}")
    print(f"  标准差: {df_sorted['RMSE(推理)'].std():.4f}")
    
    # 显示改进趋势
    print(f"\n{'='*100}")
    print("[训练改进趋势] 按Epoch顺序")
    print(f"{'='*100}\n")
    df_by_epoch = df.sort_values('Epoch')
    for idx, row in df_by_epoch.iterrows():
        marker = "★" if row['Score(推理)'] == best_result['Score(推理)'] else " "
        print(f"{marker} Epoch {row['Epoch']:3d} | Score: {row['Score(推理)']:7.2f} | RMSE: {row['RMSE(推理)']:.4f}")
    
    # 保存到CSV
    if save_csv:
        csv_path = experiment_dir / 'inference_results.csv'
        df_sorted.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n[保存] 推理结果已保存到: {csv_path}")
    
    print(f"\n{'='*100}\n")
    
    return df_sorted, best_result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='批量检查点推理脚本 - 对实验中所有检查点进行实际推理测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整推理模式（默认）- 实际加载模型并测试
  python inference.py experiments/runs/baseline_fd001_20251021_155712
  
  # 推理并保存结果到CSV
  python inference.py experiments/runs/baseline_fd001_20251021_155712 --save-csv
  
  # 快速模式 - 只读取保存的指标，不进行实际推理
  python inference.py experiments/runs/temp_1.6_20251021 --fast
        """
    )
    
    parser.add_argument('experiment_dir', type=str,
                       help='实验目录路径')
    parser.add_argument('--save-csv', action='store_true',
                       help='保存推理结果到CSV文件')
    parser.add_argument('--fast', action='store_true',
                       help='快速模式：只读取保存的指标，不进行实际推理')
    
    args = parser.parse_args()
    
    # 转换为Path对象
    experiment_dir = Path(args.experiment_dir)
    
    if not experiment_dir.exists():
        print(f"[错误] 实验目录不存在: {experiment_dir}")
        sys.exit(1)
    
    # 执行批量推理
    batch_inference(experiment_dir, save_csv=args.save_csv, fast_mode=args.fast)


if __name__ == '__main__':
    main()
