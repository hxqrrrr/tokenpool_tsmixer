"""
Main entry point for TokenPool-TSMixer RUL Prediction Experiments
Supports single and batch experiments with JSON configuration
"""
import argparse
import sys
import subprocess
import json
import time
from datetime import datetime
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
        print(f"  - TokenPool: {data_params.get('window_sample', 30)} steps -> {model_params.get('num_tokens', 10)} tokens")
        print(f"  - Compression ratio: {data_params.get('window_sample', 30) / model_params.get('num_tokens', 10):.1f}x")
        
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
    Run batch experiments from JSON configuration using subprocess
    Supports both 'base_config' and 'global_settings' formats
    
    Args:
        batch_config_path: Path to batch configuration JSON file
    """
    print(f"\n{'='*80}")
    print(f"[BATCH] Loading batch configuration: {batch_config_path}")
    print(f"{'='*80}\n")
    
    # Load batch configuration
    experiments = load_batch_configs(batch_config_path)
    
    print(f"[BATCH] Found {len(experiments)} experiments to run")
    print(f"[BATCH] Each experiment runs in an isolated subprocess")
    print(f"{'='*80}\n")
    
    # Run each experiment
    all_results = []
    
    for i, exp_config in enumerate(experiments, 1):
        exp_name = exp_config.get('experiment_name', f'exp_{i}')
        exp_desc = exp_config.get('description', '')
        
        print(f"\n{'#'*80}")
        print(f"# EXPERIMENT {i}/{len(experiments)}: {exp_name}")
        if exp_desc:
            print(f"# {exp_desc}")
        print(f"{'#'*80}\n")
        
        # Create temporary configuration file with timestamp to avoid conflicts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        temp_config_path = f"temp_{exp_name}_{timestamp}.json"
        
        try:
            # Save experiment config to temp file
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                json.dump(exp_config, f, indent=2, ensure_ascii=False)
            
            # Run experiment in subprocess
            start_time = time.time()
            result = subprocess.run(
                ['python', 'main.py', '--config', temp_config_path],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'  # Handle encoding errors gracefully
            )
            duration = time.time() - start_time
            
            # Parse results from output
            if result.returncode == 0:
                print(f"\n[SUCCESS] Experiment completed successfully!")
                print(f"   Duration: {duration:.1f}s")
                
                # Extract metrics from stdout
                output_lines = result.stdout.split('\n') if result.stdout else []
                rmse, score = None, None
                
                for line in output_lines:
                    if 'Test RMSE:' in line or 'RMSE:' in line:
                        try:
                            # Try to extract numeric value
                            parts = line.split('RMSE:')
                            if len(parts) > 1:
                                rmse = float(parts[1].split()[0].strip(','))
                        except (ValueError, IndexError):
                            pass
                    
                    if 'Test Score:' in line or 'Score:' in line:
                        try:
                            parts = line.split('Score:')
                            if len(parts) > 1:
                                score = float(parts[1].split()[0].strip(','))
                        except (ValueError, IndexError):
                            pass
                
                all_results.append({
                    'experiment_name': exp_name,
                    'status': 'success',
                    'rmse': rmse,
                    'score': score,
                    'duration': duration
                })
                
                if rmse is not None and score is not None:
                    print(f"   RMSE: {rmse:.4f}")
                    print(f"   Score: {score:.2f}")
                else:
                    print(f"   (Could not extract metrics from output)")
            
            else:
                print(f"\n[FAILED] Experiment failed!")
                stderr_text = result.stderr if result.stderr else "No error message available"
                
                # Try to extract the actual error from the end of stderr (skip warnings)
                error_lines = stderr_text.split('\n') if stderr_text else []
                # Look for actual error messages (typically at the end)
                actual_error_lines = []
                for line in reversed(error_lines):
                    if line.strip():
                        actual_error_lines.insert(0, line)
                        if len(actual_error_lines) >= 10:  # Get last 10 non-empty lines
                            break
                
                error_summary = '\n'.join(actual_error_lines[-10:]) if actual_error_lines else stderr_text[:500]
                
                print(f"   Error output (last lines):")
                for line in error_summary.split('\n')[:5]:  # Show first 5 lines of error
                    print(f"   {line}")
                if len(error_summary.split('\n')) > 5:
                    print(f"   ... (see full error in results JSON)")
                
                all_results.append({
                    'experiment_name': exp_name,
                    'status': 'failed',
                    'error': error_summary[:1000] if error_summary else "Unknown error",
                    'duration': duration,
                    'return_code': result.returncode
                })
        
        except Exception as e:
            print(f"\n[ERROR] Experiment exception: {str(e)}")
            all_results.append({
                'experiment_name': exp_name,
                'status': 'error',
                'error': str(e)
            })
        
        finally:
            # Clean up temporary config file
            try:
                Path(temp_config_path).unlink(missing_ok=True)
            except Exception:
                pass
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"[BATCH] All experiments completed")
    print(f"{'='*80}\n")
    
    # Results table
    print(f"{'='*80}")
    print(f"EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Experiment':<35} {'Status':<10} {'RMSE':<12} {'Score':<12}")
    print(f"{'-'*80}")
    
    successful = [r for r in all_results if r['status'] == 'success']
    
    for result in all_results:
        status_icon = '[OK]' if result['status'] == 'success' else '[FAIL]'
        rmse_str = f"{result['rmse']:.4f}" if result.get('rmse') is not None else 'N/A'
        score_str = f"{result['score']:.2f}" if result.get('score') is not None else 'N/A'
        print(f"{result['experiment_name']:<35} {status_icon:<10} {rmse_str:<12} {score_str:<12}")
    
    print(f"{'-'*80}")
    print(f"Success rate: {len(successful)}/{len(all_results)}")
    
    # Find best results
    if successful:
        print(f"\n{'='*80}")
        print(f"BEST RESULTS")
        print(f"{'='*80}")
        
        valid_rmse = [r for r in successful if r.get('rmse') is not None]
        valid_score = [r for r in successful if r.get('score') is not None]
        
        if valid_rmse:
            best_rmse = min(valid_rmse, key=lambda x: x['rmse'])
            print(f"Best RMSE: {best_rmse['rmse']:.4f} ({best_rmse['experiment_name']})")
        
        if valid_score:
            best_score = min(valid_score, key=lambda x: x['score'])
            print(f"Best Score: {best_score['score']:.2f} ({best_score['experiment_name']})")
    
    # Save results to JSON file
    results_filename = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump({
            'batch_config': batch_config_path,
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(all_results),
            'successful': len(successful),
            'results': all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {results_filename}")
    print(f"{'='*80}\n")


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
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (override config file seed)')
    
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
