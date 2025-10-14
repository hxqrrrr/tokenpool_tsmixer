"""
Main entry point for RUL prediction experiments
"""
import torch
import numpy as np
import random
import os
import argparse

from dataset import CMAPSSDataset
from models import (TSMixerRUL, TSMixerSGARUL, STGNNRUL, TokenPoolTSMixerRUL,
                    MultiScaleTokenPoolTSMixerRUL)
from trainers import RULTrainer
from configs import get_config
from utils import setup_logger, log_config, log_dataset_info, log_model_info


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_single_model(dataset_name='FD001', model_name='TSMixerSGA', **kwargs):
    """
    Train a single model on a dataset
    
    Args:
        dataset_name: Dataset name (FD001-FD004)
        model_name: Model name (TSMixer, TSMixerSGA, STGNN)
        **kwargs: Additional configuration parameters
        
    Returns:
        dict: Training results
    """
    # Get config for specific model and dataset
    config = get_config(model_name, dataset_name, **kwargs)
    
    # Set random seed
    set_seed(config.seed)
    
    print("\n" + "="*80)
    print(f"Training {model_name} on CMAPSS {dataset_name}")
    print("="*80)
    print(config)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = CMAPSSDataset(
        data_root=config.data_root,
        dataset_name=dataset_name,
        max_rul=config.max_rul,
        seq_len=config.patch_size,
        time_denpen_len=config.time_denpen_len,
        window_sample=config.window_sample
    )
    print(f"Dataset loaded: {len(dataset.train_x)} train samples, {len(dataset.test_x)} test samples")
    
    # Create model
    print(f"\nCreating {model_name} model...")
    if model_name == 'TSMixer':
        model = TSMixerRUL(
            patch_size=config.patch_size,
            time_denpen_len=config.time_denpen_len,
            num_sensor=config.num_sensor,
            hidden_dim=config.hidden_dim,
            num_blocks=config.num_blocks,
            dropout=config.dropout
        )
    elif model_name == 'TSMixerSGA':
        model = TSMixerSGARUL(
            patch_size=config.patch_size,
            time_denpen_len=config.time_denpen_len,
            num_sensor=config.num_sensor,
            hidden_dim=config.hidden_dim,
            num_blocks=config.num_blocks,
            dropout=config.dropout,
            sga_time_rr=getattr(config, 'sga_time_rr', 4),      # 保持原容量
            sga_feat_rr=getattr(config, 'sga_feat_rr', 4),      # 保持原容量
            sga_dropout=getattr(config, 'sga_dropout', 0.15),   # 适度增加正则
            sga_fuse=getattr(config, 'sga_fuse', 'add'),  # Add融合更适合FD002
            head_pool=getattr(config, 'head_pool', 'none'),  # 对标TSMixer，使用flatten
            use_mlp_head=getattr(config, 'use_mlp_head', True)
        )
    elif model_name == 'STGNN':
        model = STGNNRUL(
            patch_size=config.patch_size,
            time_denpen_len=config.time_denpen_len,
            conv_kernel=getattr(config, 'stgnn_conv_kernel', 2),
            conv_out=getattr(config, 'stgnn_conv_out', 7),
            num_windows=getattr(config, 'stgnn_num_windows', 14),
            lstmout_dim=getattr(config, 'stgnn_lstmout_dim', 32),
            lstmhidden_dim=getattr(config, 'stgnn_lstmhidden_dim', 8),
            hidden_dim=getattr(config, 'stgnn_hidden_dim', 8),
            moving_window=getattr(config, 'stgnn_moving_window', [2, 2]),
            stride=getattr(config, 'stgnn_stride', [1, 2]),
            pool_choice=getattr(config, 'stgnn_pool_choice', 'mean'),
            decay=getattr(config, 'stgnn_decay', 0.7)
        )
    elif model_name == 'TokenPoolTSMixer':
        model = TokenPoolTSMixerRUL(
            patch_size=config.patch_size,
            time_denpen_len=config.time_denpen_len,
            num_sensor=config.num_sensor,
            # TokenPool 参数
            num_tokens=getattr(config, 'num_tokens', 10),
            token_dim=getattr(config, 'token_dim', 128),
            num_heads=getattr(config, 'num_heads', 4),
            temperature=getattr(config, 'temperature', 1.5),
            attn_dropout=getattr(config, 'attn_dropout', 0.1),
            use_pos_encoding=getattr(config, 'use_pos_encoding', True),
            # TSMixer 参数
            hidden_dim=config.hidden_dim,
            num_blocks=config.num_blocks,
            dropout=config.dropout
        )
    elif model_name == 'MultiScaleTokenPool':
        model = MultiScaleTokenPoolTSMixerRUL(
            patch_size=config.patch_size,
            time_denpen_len=config.time_denpen_len,
            num_sensor=config.num_sensor,
            # 多尺度参数
            short_patch_size=getattr(config, 'short_patch_size', 5),
            long_patch_size=getattr(config, 'long_patch_size', 9),
            num_short_tokens=getattr(config, 'num_short_tokens', 8),
            num_long_tokens=getattr(config, 'num_long_tokens', 4),
            # TokenPool 参数
            token_dim=getattr(config, 'token_dim', 128),
            num_heads=getattr(config, 'num_heads', 4),
            short_temperature=getattr(config, 'short_temperature', 1.3),
            long_temperature=getattr(config, 'long_temperature', 1.8),
            attn_dropout=getattr(config, 'attn_dropout', 0.1),
            use_pos_encoding=getattr(config, 'use_pos_encoding', True),
            # 融合参数
            fusion_mode=getattr(config, 'fusion_mode', 'concat'),
            # TSMixer 参数
            hidden_dim=config.hidden_dim,
            num_blocks=config.num_blocks,
            dropout=config.dropout
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Setup logger
    logger = setup_logger(
        log_dir=getattr(config, 'log_dir', './logs'),
        model_name=model_name,
        dataset_name=dataset_name
    )
    
    # Log configuration
    log_config(logger, config)
    
    # Log dataset info
    log_dataset_info(logger, dataset)
    
    # Log model info
    log_model_info(logger, model)
    
    # Create trainer
    trainer = RULTrainer(model, dataset, config, logger=logger)
    
    # Train
    results = trainer.train()
    
    # Save results
    os.makedirs(config.results_dir, exist_ok=True)
    results_path = os.path.join(
        config.results_dir, 
        f"{model_name}_{dataset_name}_results.npy"
    )
    np.save(results_path, {
        'rmse': results['rmse'],
        'score': results['score'],
        'history': results['history']
    })
    print(f"\nResults saved to {results_path}")
    
    # Log final results
    logger.info(f"结果已保存到: {results_path}")
    
    return results


def train_all_datasets(model_name='TSMixerSGA', **kwargs):
    """
    Train a model on all CMAPSS datasets
    
    Args:
        model_name: Model name (TSMixer, TSMixerSGA)
        **kwargs: Additional configuration parameters
        
    Returns:
        dict: Results for all datasets
    """
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    all_results = {}
    
    for dataset_name in datasets:
        try:
            results = train_single_model(dataset_name, model_name, **kwargs)
            all_results[dataset_name] = results
        except Exception as e:
            print(f"\nError training on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF ALL RESULTS")
    print("="*80)
    print(f"{'Dataset':<10} {'RMSE':<10} {'Score':<10}")
    print("-"*80)
    for dataset_name, results in all_results.items():
        print(f"{dataset_name:<10} {results['rmse']:<10.4f} {results['score']:<10.2f}")
    print("="*80)
    
    return all_results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train RUL prediction models')
    parser.add_argument('--dataset', type=str, default='FD004',
                       choices=['FD001', 'FD002', 'FD003', 'FD004', 'all'],
                       help='Dataset name or "all" for all datasets')
    parser.add_argument('--model', type=str, default='TokenPoolTSMixer',
                       choices=['TSMixer', 'TSMixerSGA', 'STGNN', 'TokenPoolTSMixer', 'MultiScaleTokenPool'],
                       help='Model name')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')   
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension')
    parser.add_argument('--num_blocks', type=int, default=4,
                       help='Number of model blocks')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Prepare kwargs for config
    # 只传递用户明确指定的参数，避免默认值覆盖模型特定配置
    kwargs = {}
    if args.epochs != 50:  # 如果用户指定了非默认值
        kwargs['epochs'] = args.epochs
    if args.batch_size != 128:  # 如果用户指定了非默认值
        kwargs['batch_size'] = args.batch_size
    if args.lr != 1e-3:  # 如果用户指定了非默认值
        kwargs['lr'] = args.lr
    if args.hidden_dim != 64:  # 如果用户指定了非默认值
        kwargs['hidden_dim'] = args.hidden_dim
    if args.num_blocks != 4:  # 如果用户指定了非默认值
        kwargs['num_blocks'] = args.num_blocks
    if args.seed != 42:  # 如果用户指定了非默认值
        kwargs['seed'] = args.seed
    
    # Train
    if args.dataset == 'all':
        train_all_datasets(args.model, **kwargs)
    else:
        train_single_model(args.dataset, args.model, **kwargs)


if __name__ == '__main__':
    main()

