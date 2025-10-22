"""
单引擎完整生命周期预测分析工具
选择一个引擎，展示其整个运行周期的RUL预测轨迹

用法: 
    python predict_engine_lifecycle.py <checkpoint_path> <engine_id> [--dataset train/test] [--output-dir OUTPUT_DIR]

示例:
    # 分析训练集中的引擎#56
    python predict_engine_lifecycle.py checkpoint.pth 56 --dataset train
    
    # 分析测试集中的引擎#10
    python predict_engine_lifecycle.py checkpoint.pth 10 --dataset test
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


def get_engine_data(dataset, engine_id: int, split: str = 'train'):
    """
    获取指定引擎的完整数据
    
    Args:
        dataset: CMAPSSDataset实例
        engine_id: 引擎ID（从1开始）
        split: 'train' 或 'test'
    
    Returns:
        engine_df: 该引擎的数据（包含归一化传感器数据和元信息）
    """
    # 获取原始数据（包含id和cycle）
    if split == 'train':
        df_raw = dataset.train_df
        df_normalized = dataset.train_normalized
    else:
        df_raw = dataset.test_df
        df_normalized = dataset.test_normalized
    
    # 筛选指定引擎的原始数据
    engine_raw = df_raw[df_raw['id'] == engine_id].copy()
    
    if len(engine_raw) == 0:
        available_engines = sorted(df_raw['id'].unique())
        raise ValueError(
            f"引擎 #{engine_id} 不存在于{split}集中！\n"
            f"可用的引擎ID: {available_engines}"
        )
    
    # 获取对应的归一化传感器数据（使用相同的索引）
    engine_normalized = df_normalized.loc[engine_raw.index]
    
    # 合并：保留id, cycle, setting列和归一化的传感器数据
    engine_df = pd.concat([
        engine_raw[['id', 'cycle', 'setting1', 'setting2', 'setting3']],
        engine_normalized
    ], axis=1)
    
    # 计算真实RUL
    max_cycle = engine_df['cycle'].max()
    engine_df['actual_RUL'] = max_cycle - engine_df['cycle']
    engine_df['actual_RUL'] = engine_df['actual_RUL'].clip(upper=dataset.max_rul)
    
    return engine_df


def predict_engine_lifecycle(model, dataset, engine_df, device):
    """
    对引擎的整个生命周期进行逐步预测
    
    Returns:
        cycles: 时间步序列
        predicted_rul: 预测RUL序列
        actual_rul: 真实RUL序列
    """
    print(f"\n[预测引擎生命周期]")
    print(f"  引擎ID: {engine_df['id'].iloc[0]}")
    print(f"  总循环数: {len(engine_df)}")
    print(f"  窗口大小: {dataset.window_sample}")
    
    # 获取传感器列（s1-s21的归一化数据）
    sensor_cols = [col for col in engine_df.columns 
                   if col.startswith('s') and col[1:].isdigit()]
    
    window_size = dataset.window_sample
    time_denpen_len = dataset.time_denpen_len
    patch_size = dataset.seq_len
    
    cycles = []
    predicted_rul_list = []
    actual_rul_list = []
    
    # 从第window_size个循环开始预测（需要足够的历史数据）
    for i in range(window_size - 1, len(engine_df)):
        # 获取窗口数据
        window_data = engine_df.iloc[i - window_size + 1:i + 1][sensor_cols].values
        
        # 重塑为模型输入格式 [1, time_denpen_len, patch_size, num_sensor]
        # 需要将window_data reshape成 (time_denpen_len, patch_size, num_sensor)
        if len(window_data) == window_size:
            try:
                # Reshape: (window_size, num_sensor) -> (time_denpen_len, patch_size, num_sensor)
                x = window_data.reshape(time_denpen_len, patch_size, len(sensor_cols))
                x = torch.FloatTensor(x).unsqueeze(0).to(device)  # Add batch dimension
                
                # 预测
                with torch.no_grad():
                    pred = model(x)
                    pred_rul = (pred.item() * dataset.max_rul)
                
                # 记录
                cycle = engine_df.iloc[i]['cycle']
                actual_rul = engine_df.iloc[i]['actual_RUL']
                
                cycles.append(cycle)
                predicted_rul_list.append(pred_rul)
                actual_rul_list.append(actual_rul)
                
            except Exception as e:
                print(f"  [警告] 循环 {i} 预测失败: {str(e)}")
                continue
    
    print(f"  成功预测: {len(cycles)} 个时间点\n")
    
    return np.array(cycles), np.array(predicted_rul_list), np.array(actual_rul_list)


def plot_lifecycle_prediction(cycles, predicted_rul, actual_rul, 
                              engine_id: int, dataset_name: str,
                              checkpoint_name: str, output_path: Path,
                              figsize=(12, 4), dpi=300):
    """生成引擎生命周期预测对比图"""
    print(f"[生成图表] {output_path}")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 真实RUL - 蓝色虚线
    ax.plot(cycles, actual_rul, '--', 
            color='blue', label='Actual RUL', 
            linewidth=2, alpha=0.8)
    
    # 预测RUL - 红色实线带标记
    ax.plot(cycles, predicted_rul, 'o-', 
            color='red', label='Predicted RUL',
            markersize=4, linewidth=1.5, alpha=0.8)
    
    # 计算RMSE
    rmse = np.sqrt(np.mean((predicted_rul - actual_rul) ** 2))
    mae = np.mean(np.abs(predicted_rul - actual_rul))
    
    # 样式设置
    ax.set_xlabel('Cycle', fontsize=12)
    ax.set_ylabel('RUL', fontsize=12)
    ax.set_title(f'Prediction on {dataset_name} Engine #{engine_id}\n'
                f'RMSE: {rmse:.2f}, MAE: {mae:.2f}', 
                fontsize=13)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(cycles[0] - 5, cycles[-1] + 5)
    ax.set_ylim(-5, max(max(actual_rul), max(predicted_rul)) + 10)
    
    # 保存
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  图表尺寸: {figsize}, DPI: {dpi}\n")
    
    return rmse, mae


def save_lifecycle_csv(cycles, predicted_rul, actual_rul, output_path: Path):
    """保存生命周期预测结果到CSV"""
    print(f"[保存CSV] {output_path}")
    
    errors = predicted_rul - actual_rul
    abs_errors = np.abs(errors)
    
    results_df = pd.DataFrame({
        'Cycle': cycles,
        'Predicted_RUL': predicted_rul,
        'Actual_RUL': actual_rul,
        'Error': errors,
        'Absolute_Error': abs_errors
    })
    
    results_df.to_csv(output_path, index=False)
    print(f"  已保存 {len(results_df)} 条记录\n")
    
    return results_df


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='单引擎完整生命周期预测分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析训练集中的引擎#56
  python predict_engine_lifecycle.py checkpoint.pth 56 --dataset train
  
  # 分析测试集中的引擎#10
  python predict_engine_lifecycle.py checkpoint.pth 10 --dataset test
  
  # 自定义输出目录
  python predict_engine_lifecycle.py checkpoint.pth 56 --dataset train --output-dir ./analysis
        """
    )
    
    parser.add_argument('checkpoint_path', type=str,
                       help='Checkpoint文件路径 (.pth)')
    parser.add_argument('engine_id', type=int,
                       help='引擎ID')
    parser.add_argument('--dataset', type=str, default='train', choices=['train', 'test'],
                       help='数据集选择: train 或 test (默认: train)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录 (默认: checkpoint目录下的engine_lifecycle/)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='图表DPI (默认: 300)')
    parser.add_argument('--figsize', type=str, default='12,4',
                       help='图表大小 "宽,高" (默认: 12,4)')
    
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
        output_dir = checkpoint_path.parent.parent / 'engine_lifecycle' / f'engine_{args.engine_id}_{args.dataset}'
    
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
        print(f"\n{'='*80}")
        print(f"加载Checkpoint: {checkpoint_path.name}")
        print(f"{'='*80}\n")
        
        # 加载checkpoint和创建模型
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = create_model_from_checkpoint(checkpoint)
        model = model.to(device)
        
        dataset_name = checkpoint['data_params'].get('dataset_name', 'FD001')
        print(f"[1/5] 模型加载完成")
        print(f"  模型: {checkpoint['model_params'].get('model_name')}")
        print(f"  数据集: {dataset_name}")
        
        # 创建数据集
        print(f"\n[2/5] 加载数据集...")
        dataset = create_dataset_from_checkpoint(checkpoint)
        
        # 获取指定引擎数据
        print(f"\n[3/5] 获取引擎数据...")
        engine_df = get_engine_data(dataset, args.engine_id, args.dataset)
        
        # 预测引擎生命周期
        print(f"\n[4/5] 执行生命周期预测...")
        cycles, predicted_rul, actual_rul = predict_engine_lifecycle(
            model, dataset, engine_df, device
        )
        
        # 输出结果
        print(f"{'='*80}")
        print("输出结果")
        print(f"{'='*80}\n")
        print(f"输出目录: {output_dir}\n")
        
        # 保存CSV
        csv_path = output_dir / f'engine_{args.engine_id}_lifecycle.csv'
        save_lifecycle_csv(cycles, predicted_rul, actual_rul, csv_path)
        
        # 生成可视化图表
        fig_path = output_dir / f'engine_{args.engine_id}_lifecycle.png'
        rmse, mae = plot_lifecycle_prediction(
            cycles, predicted_rul, actual_rul,
            args.engine_id, dataset_name,
            checkpoint_path.name, fig_path,
            figsize=figsize, dpi=args.dpi
        )
        
        print(f"{'='*80}")
        print("✓ 分析完成！")
        print(f"{'='*80}\n")
        print("生成的文件:")
        print(f"  1. CSV表格: {csv_path}")
        print(f"  2. 可视化图表: {fig_path}")
        print(f"\n性能指标:")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print()
        
    except Exception as e:
        print(f"\n[错误] 分析过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

