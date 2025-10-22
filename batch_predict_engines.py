"""
批量预测所有引擎的生命周期
只保存RMSE低于阈值的引擎结果

用法: 
    python batch_predict_engines.py <checkpoint_path> [--rmse-threshold THRESHOLD] [--dataset train/test]

示例:
    # 预测所有训练引擎，只保存RMSE<=13的
    python batch_predict_engines.py checkpoint.pth --rmse-threshold 13
    
    # 预测测试集引擎
    python batch_predict_engines.py checkpoint.pth --dataset test --rmse-threshold 15
"""
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm

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
    """获取指定引擎的完整数据"""
    if split == 'train':
        df_raw = dataset.train_df
        df_normalized = dataset.train_normalized
    else:
        df_raw = dataset.test_df
        df_normalized = dataset.test_normalized
    
    engine_raw = df_raw[df_raw['id'] == engine_id].copy()
    
    if len(engine_raw) == 0:
        return None
    
    engine_normalized = df_normalized.loc[engine_raw.index]
    
    engine_df = pd.concat([
        engine_raw[['id', 'cycle', 'setting1', 'setting2', 'setting3']],
        engine_normalized
    ], axis=1)
    
    max_cycle = engine_df['cycle'].max()
    engine_df['actual_RUL'] = max_cycle - engine_df['cycle']
    engine_df['actual_RUL'] = engine_df['actual_RUL'].clip(upper=dataset.max_rul)
    
    return engine_df


def predict_engine_lifecycle(model, dataset, engine_df, device):
    """对引擎的整个生命周期进行逐步预测"""
    sensor_cols = [col for col in engine_df.columns 
                   if col.startswith('s') and col[1:].isdigit()]
    
    window_size = dataset.window_sample
    time_denpen_len = dataset.time_denpen_len
    patch_size = dataset.seq_len
    
    cycles = []
    predicted_rul_list = []
    actual_rul_list = []
    
    for i in range(window_size - 1, len(engine_df)):
        window_data = engine_df.iloc[i - window_size + 1:i + 1][sensor_cols].values
        
        if len(window_data) == window_size:
            try:
                x = window_data.reshape(time_denpen_len, patch_size, len(sensor_cols))
                x = torch.FloatTensor(x).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred = model(x)
                    pred_rul = (pred.item() * dataset.max_rul)
                
                cycle = engine_df.iloc[i]['cycle']
                actual_rul = engine_df.iloc[i]['actual_RUL']
                
                cycles.append(cycle)
                predicted_rul_list.append(pred_rul)
                actual_rul_list.append(actual_rul)
                
            except:
                continue
    
    return np.array(cycles), np.array(predicted_rul_list), np.array(actual_rul_list)


def plot_lifecycle_prediction(cycles, predicted_rul, actual_rul, 
                              engine_id: int, dataset_name: str,
                              rmse: float, mae: float, output_path: Path,
                              figsize=(12, 4), dpi=300):
    """生成引擎生命周期预测对比图"""
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(cycles, actual_rul.flatten(), '--', 
            color='blue', label='Actual RUL', 
            linewidth=2, alpha=0.8)
    
    ax.plot(cycles, predicted_rul.flatten(), 'o-', 
            color='red', label='Predicted RUL',
            markersize=4, linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Cycle', fontsize=12)
    ax.set_ylabel('RUL', fontsize=12)
    ax.set_title(f'Prediction on {dataset_name} Engine #{engine_id}\n'
                f'RMSE: {rmse:.2f}, MAE: {mae:.2f}', 
                fontsize=13)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(cycles[0] - 5, cycles[-1] + 5)
    ax.set_ylim(-5, max(max(actual_rul), max(predicted_rul)) + 10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def save_lifecycle_csv(cycles, predicted_rul, actual_rul, output_path: Path):
    """保存生命周期预测结果到CSV"""
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
    return results_df


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='批量预测所有引擎的生命周期',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 预测所有训练引擎，只保存RMSE<=13的
  python batch_predict_engines.py checkpoint.pth --rmse-threshold 13
  
  # 预测测试集引擎
  python batch_predict_engines.py checkpoint.pth --dataset test --rmse-threshold 15
        """
    )
    
    parser.add_argument('checkpoint_path', type=str,
                       help='Checkpoint文件路径 (.pth)')
    parser.add_argument('--dataset', type=str, default='train', choices=['train', 'test'],
                       help='数据集选择: train 或 test (默认: train)')
    parser.add_argument('--rmse-threshold', type=float, default=13.0,
                       help='RMSE阈值，只保存低于此值的结果 (默认: 13.0)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录 (默认: checkpoint目录下的batch_engines/)')
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
        output_dir = checkpoint_path.parent.parent / 'batch_engines' / args.dataset
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 解析图表大小
    try:
        figsize = tuple(map(float, args.figsize.split(',')))
    except:
        figsize = (12, 4)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print(f"批量引擎生命周期预测")
    print(f"{'='*80}\n")
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"数据集: {args.dataset}")
    print(f"RMSE阈值: {args.rmse_threshold} (只保存低于此值的结果)")
    print(f"输出目录: {output_dir}")
    print(f"设备: {device}\n")
    
    try:
        # 加载checkpoint和创建模型
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = create_model_from_checkpoint(checkpoint)
        model = model.to(device)
        
        dataset_name = checkpoint['data_params'].get('dataset_name', 'FD001')
        
        # 创建数据集
        dataset = create_dataset_from_checkpoint(checkpoint)
        
        # 获取所有引擎ID
        if args.dataset == 'train':
            all_engine_ids = sorted(dataset.train_df['id'].unique())
        else:
            all_engine_ids = sorted(dataset.test_df['id'].unique())
        
        print(f"总引擎数: {len(all_engine_ids)}")
        print(f"引擎ID范围: {all_engine_ids[0]} - {all_engine_ids[-1]}\n")
        
        # 统计信息
        results_summary = []
        saved_count = 0
        skipped_count = 0
        
        # 批量预测
        print(f"开始批量预测...\n")
        
        for engine_id in tqdm(all_engine_ids, desc="预测进度"):
            try:
                # 获取引擎数据
                engine_df = get_engine_data(dataset, engine_id, args.dataset)
                
                if engine_df is None or len(engine_df) < dataset.window_sample:
                    skipped_count += 1
                    continue
                
                # 预测生命周期
                cycles, predicted_rul, actual_rul = predict_engine_lifecycle(
                    model, dataset, engine_df, device
                )
                
                if len(cycles) == 0:
                    skipped_count += 1
                    continue
                
                # 计算RMSE和MAE
                rmse = np.sqrt(np.mean((predicted_rul - actual_rul) ** 2))
                mae = np.mean(np.abs(predicted_rul - actual_rul))
                
                # 记录统计信息
                results_summary.append({
                    'Engine_ID': engine_id,
                    'RMSE': rmse,
                    'MAE': mae,
                    'Num_Predictions': len(cycles),
                    'Max_Cycle': cycles[-1],
                    'Saved': rmse <= args.rmse_threshold
                })
                
                # 只保存RMSE低于阈值的结果
                if rmse <= args.rmse_threshold:
                    engine_dir = output_dir / f'engine_{engine_id}'
                    engine_dir.mkdir(exist_ok=True)
                    
                    # 保存CSV
                    csv_path = engine_dir / f'engine_{engine_id}_lifecycle.csv'
                    save_lifecycle_csv(cycles, predicted_rul, actual_rul, csv_path)
                    
                    # 保存图表
                    fig_path = engine_dir / f'engine_{engine_id}_lifecycle.png'
                    plot_lifecycle_prediction(
                        cycles, predicted_rul, actual_rul,
                        engine_id, dataset_name,
                        rmse, mae, fig_path,
                        figsize=figsize, dpi=args.dpi
                    )
                    
                    saved_count += 1
                else:
                    skipped_count += 1
                
            except Exception as e:
                print(f"\n[警告] 引擎 {engine_id} 处理失败: {str(e)}")
                skipped_count += 1
                continue
        
        # 保存统计摘要
        summary_df = pd.DataFrame(results_summary)
        summary_path = output_dir / 'batch_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        
        # 显示结果
        print(f"\n{'='*80}")
        print("批量预测完成！")
        print(f"{'='*80}\n")
        print(f"总处理引擎数: {len(all_engine_ids)}")
        print(f"成功预测: {len(results_summary)}")
        print(f"保存结果: {saved_count} (RMSE <= {args.rmse_threshold})")
        print(f"跳过/失败: {skipped_count}")
        
        if len(results_summary) > 0:
            print(f"\nRMSE统计:")
            print(f"  最小: {summary_df['RMSE'].min():.2f}")
            print(f"  最大: {summary_df['RMSE'].max():.2f}")
            print(f"  平均: {summary_df['RMSE'].mean():.2f}")
            print(f"  中位数: {summary_df['RMSE'].median():.2f}")
            
            saved_df = summary_df[summary_df['Saved'] == True]
            if len(saved_df) > 0:
                print(f"\n保存结果的RMSE统计:")
                print(f"  平均: {saved_df['RMSE'].mean():.2f}")
                print(f"  范围: {saved_df['RMSE'].min():.2f} - {saved_df['RMSE'].max():.2f}")
        
        print(f"\n输出文件:")
        print(f"  统计摘要: {summary_path}")
        print(f"  结果目录: {output_dir}")
        print()
        
    except Exception as e:
        print(f"\n[错误] 批量预测出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

