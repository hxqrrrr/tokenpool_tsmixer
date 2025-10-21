"""
Main entry point for TokenPool-TSMixer RUL Prediction Experiments
Supports single and batch experiments with JSON configuration
"""
import argparse
import sys
import torch
import random
import numpy as np
from pathlib import Path

from dataset import CMAPSSDataset
from models.tokenpool_tsmixer import TokenPoolTSMixerRUL
from trainer import RULTrainer
from experiments import ExperimentManager, load_experiment_config, load_batch_configs


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def merge_config(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries
    
    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    
    return result


def run_single_experiment(config: dict):
    """
    Run a single experiment
    
    Args:
        config: Experiment configuration dictionary
    """
    # Set random seed
    set_seed(config.get('seed', 42))
    
    # Create experiment manager
    exp_manager = ExperimentManager(
        experiment_name=config['experiment_name'],
        base_dir='./experiments'
    )
    
    # Save configuration
    exp_manager.save_config(config)
    
    # Start experiment
    exp_manager.start_experiment()
    
    try:
        # Extract parameters
        dataset_name = config['dataset_name']
        data_params = config.get('data_params', {})
        model_params = config.get('model_params', {})
        training_params = config.get('training_params', {})
        output_config = config.get('output', {})
        
        # Create dataset
        print(f"\n[DATA] Loading {dataset_name} dataset...")
        dataset = CMAPSSDataset(
            data_root='./CMAPSSData',
            dataset_name=dataset_name,
            max_rul=data_params.get('max_rul', 125),
            seq_len=data_params.get('patch_size', 5),
            time_denpen_len=data_params.get('time_denpen_len', 6),
            window_sample=data_params.get('window_sample', 30)
        )
        
        # Get data to check sizes
        train_data = dataset.get_train_data()
        val_data = dataset.get_val_data()
        test_data = dataset.get_test_data()
        
        print(f"[DATA] Dataset loaded:")
        print(f"  - Training samples: {len(train_data['x'])}")
        print(f"  - Validation samples: {len(val_data['x'])}")
        print(f"  - Test samples: {len(test_data['x'])}")
        print(f"  - Window size: {data_params.get('window_sample', 30)}")
        print(f"  - Sequence shape: [{data_params.get('time_denpen_len', 6)}, {data_params.get('patch_size', 5)}, 14]")
        
        # Create model
        print(f"\n[MODEL] Creating TokenPool-TSMixer...")
        model = TokenPoolTSMixerRUL(
            patch_size=data_params.get('patch_size', 5),
            time_denpen_len=data_params.get('time_denpen_len', 6),
            num_sensor=14,
            # TokenPool parameters
            num_tokens=model_params.get('num_tokens', 10),
            token_dim=model_params.get('token_dim', 128),
            num_heads=model_params.get('num_heads', 4),
            temperature=model_params.get('temperature', 1.5),
            attn_dropout=model_params.get('attn_dropout', 0.1),
            use_pos_encoding=model_params.get('use_pos_encoding', True),
            # TSMixer parameters
            hidden_dim=model_params.get('hidden_dim', 64),
            num_blocks=model_params.get('num_blocks', 4),
            dropout=model_params.get('dropout', 0.1)
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"[MODEL] Model created:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - TokenPool: {data_params.get('window_sample', 30)} steps → {model_params.get('num_tokens', 10)} tokens")
        print(f"  - Compression ratio: {data_params.get('window_sample', 30) / model_params.get('num_tokens', 10):.1f}×")
        
        # Create training configuration
        train_config = {
            'epochs': training_params.get('epochs', 50),
            'batch_size': training_params.get('batch_size', 128),
            'lr': training_params.get('lr', 0.001),
            'weight_decay': training_params.get('weight_decay', 1e-5),
            'lr_scheduler': training_params.get('lr_scheduler', True),
            'patience': training_params.get('patience', 10),
            'lr_decay_factor': training_params.get('lr_decay_factor', 0.5),
            'lr_patience': training_params.get('lr_patience', 5),
            'log_dir': str(exp_manager.log_dir),
            'best_model_metric': 'score'  # Use scoring function for best model
        }
        
        # Create trainer
        print(f"\n[TRAIN] Training configuration:")
        print(f"  - Epochs: {train_config['epochs']}")
        print(f"  - Batch size: {train_config['batch_size']}")
        print(f"  - Learning rate: {train_config['lr']}")
        print(f"  - Weight decay: {train_config['weight_decay']}")
        
        trainer = RULTrainer(
            model=model,
            dataset=dataset,
            config=train_config,
            experiment_manager=exp_manager
        )
        
        # Train model
        print(f"\n{'='*80}")
        print(f"[TRAIN] Starting training...")
        print(f"{'='*80}\n")
        
        results = trainer.train()
        
        # Extract test results from training
        test_results = {
            'rmse': results['rmse'],
            'score': results['score']
        }
        
        # Log test results
        exp_manager.log_test_results(test_results)
        
        # Save predictions if requested
        if output_config.get('save_predictions', True):
            predictions, targets = trainer.predict(split='test')
            exp_manager.save_predictions(predictions, targets, dataset_split='test')
        
        # Save attention weights if requested
        if output_config.get('save_attention_weights', False):
            attention_weights = trainer.get_attention_weights(split='test')
            if attention_weights is not None:
                exp_manager.save_attention_weights(attention_weights)
        
        # End experiment
        exp_manager.end_experiment(status='completed')
        
        return test_results
        
    except Exception as e:
        print(f"\n[ERROR] Experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        exp_manager.end_experiment(status='failed')
        raise


def run_batch_experiments(batch_config_path: str):
    """
    Run batch experiments from JSON configuration
    
    Args:
        batch_config_path: Path to batch configuration JSON file
    """
    print(f"\n{'='*80}")
    print(f"[BATCH] Loading batch configuration: {batch_config_path}")
    print(f"{'='*80}\n")
    
    # Load batch configuration
    experiments = load_batch_configs(batch_config_path)
    
    print(f"[BATCH] Found {len(experiments)} experiments to run\n")
    
    # Run each experiment
    all_results = []
    
    for i, exp_config in enumerate(experiments, 1):
        print(f"\n{'#'*80}")
        print(f"# BATCH EXPERIMENT {i}/{len(experiments)}: {exp_config['experiment_name']}")
        print(f"{'#'*80}\n")
        
        try:
            results = run_single_experiment(exp_config)
            all_results.append({
                'experiment_name': exp_config['experiment_name'],
                'status': 'completed',
                **results
            })
        except Exception as e:
            print(f"\n[ERROR] Experiment {exp_config['experiment_name']} failed: {str(e)}")
            all_results.append({
                'experiment_name': exp_config['experiment_name'],
                'status': 'failed',
                'error': str(e)
            })
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"[BATCH] Batch experiments completed")
    print(f"{'='*80}\n")
    
    print(f"{'='*80}")
    print(f"SUMMARY OF ALL EXPERIMENTS")
    print(f"{'='*80}")
    
    for result in all_results:
        print(f"\n{result['experiment_name']}:")
        if result['status'] == 'completed':
            print(f"  Status: ✓ Completed")
            print(f"  RMSE: {result.get('rmse', 'N/A'):.4f}")
            print(f"  Score: {result.get('score', 'N/A'):.2f}")
        else:
            print(f"  Status: ✗ Failed")
            print(f"  Error: {result.get('error', 'Unknown')}")
    
    print(f"\n{'='*80}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='TokenPool-TSMixer RUL Prediction Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single experiment from JSON config
  python main.py --config experiments/configs/single_experiment.json
  
  # Run batch experiments
  python main.py --batch experiments/configs/batch_temperature_sweep.json
  
  # Quick single experiment with default FD001 settings
  python main.py --quick --dataset FD001
  
  # Override specific parameters
  python main.py --config experiments/configs/single_experiment.json --temperature 1.8 --lr 0.0005
        """
    )
    
    parser.add_argument('--config', type=str, 
                       help='Path to single experiment JSON configuration file')
    parser.add_argument('--batch', type=str,
                       help='Path to batch experiment JSON configuration file')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick experiment with default settings')
    parser.add_argument('--dataset', type=str, default='FD001',
                       choices=['FD001', 'FD002', 'FD003', 'FD004'],
                       help='Dataset to use (for quick mode)')
    
    # Parameter overrides
    parser.add_argument('--temperature', type=float,
                       help='Override temperature parameter')
    parser.add_argument('--num_tokens', type=int,
                       help='Override number of tokens')
    parser.add_argument('--token_dim', type=int,
                       help='Override token dimension')
    parser.add_argument('--window_sample', type=int,
                       help='Override window sample size')
    parser.add_argument('--lr', type=float,
                       help='Override learning rate')
    parser.add_argument('--epochs', type=int,
                       help='Override number of epochs')
    parser.add_argument('--batch_size', type=int,
                       help='Override batch size')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Determine mode
    if args.batch:
        # Batch experiment mode
        run_batch_experiments(args.batch)
    
    elif args.config:
        # Single experiment from config file
        config = load_experiment_config(args.config)
        
        # Apply overrides
        if args.temperature is not None:
            config.setdefault('model_params', {})['temperature'] = args.temperature
        if args.num_tokens is not None:
            config.setdefault('model_params', {})['num_tokens'] = args.num_tokens
        if args.token_dim is not None:
            config.setdefault('model_params', {})['token_dim'] = args.token_dim
        if args.window_sample is not None:
            config.setdefault('data_params', {})['window_sample'] = args.window_sample
        if args.lr is not None:
            config.setdefault('training_params', {})['lr'] = args.lr
        if args.epochs is not None:
            config.setdefault('training_params', {})['epochs'] = args.epochs
        if args.batch_size is not None:
            config.setdefault('training_params', {})['batch_size'] = args.batch_size
        if args.seed is not None:
            config['seed'] = args.seed
        
        run_single_experiment(config)
    
    elif args.quick:
        # Quick experiment mode with default settings
        from experiments.config_loader import create_config_template
        import tempfile
        import os
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_config_path = f.name
        
        create_config_template(temp_config_path)
        config = load_experiment_config(temp_config_path)
        
        # Update with dataset
        config['dataset_name'] = args.dataset
        config['experiment_name'] = f'quick_{args.dataset.lower()}'
        
        # Apply dataset-specific defaults
        if args.dataset == 'FD001':
            config['data_params'] = {
                'window_sample': 30,
                'patch_size': 5,
                'time_denpen_len': 6,
                'max_rul': 125
            }
            config['model_params'].update({
                'num_tokens': 10,
                'temperature': 1.5
            })
        elif args.dataset in ['FD002', 'FD003', 'FD004']:
            config['data_params'] = {
                'window_sample': 50,
                'patch_size': 5,
                'time_denpen_len': 10,
                'max_rul': 125
            }
            config['model_params'].update({
                'num_tokens': 16,
                'token_dim': 64,
                'temperature': 1.6 if args.dataset == 'FD002' else 1.8
            })
        
        # Apply overrides
        if args.temperature is not None:
            config['model_params']['temperature'] = args.temperature
        if args.num_tokens is not None:
            config['model_params']['num_tokens'] = args.num_tokens
        if args.token_dim is not None:
            config['model_params']['token_dim'] = args.token_dim
        if args.window_sample is not None:
            config['data_params']['window_sample'] = args.window_sample
        if args.lr is not None:
            config['training_params']['lr'] = args.lr
        if args.epochs is not None:
            config['training_params']['epochs'] = args.epochs
        if args.batch_size is not None:
            config['training_params']['batch_size'] = args.batch_size
        config['seed'] = args.seed
        
        run_single_experiment(config)
        
        # Clean up temp file
        os.unlink(temp_config_path)
    
    else:
        parser.print_help()
        print("\n[ERROR] Please specify --config, --batch, or --quick mode")
        sys.exit(1)


if __name__ == '__main__':
    main()
