"""
Experiment Manager for TokenPool-TSMixer RUL Prediction
Handles experiment tracking, logging, and result management
"""
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import numpy as np
import pandas as pd


class ExperimentManager:
    """
    Manager for tracking and logging experiments
    """
    
    def __init__(self, experiment_name: str, base_dir: str = './experiments'):
        """
        Initialize experiment manager
        
        Args:
            experiment_name: Name of the experiment
            base_dir: Base directory for experiments
        """
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        
        # Create experiment directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = self.base_dir / 'runs' / f"{experiment_name}_{timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.log_dir = self.exp_dir / 'logs'
        self.result_dir = self.exp_dir / 'results'
        self.config_dir = self.exp_dir / 'config'
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        
        for dir_path in [self.log_dir, self.result_dir, self.config_dir, self.checkpoint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking
        self.start_time = None
        self.end_time = None
        self.metrics_history = []
        self.config = {}
        self.best_metrics = {}
        
        print(f"[INFO] Experiment directory: {self.exp_dir}")
    
    def save_config(self, config: Dict[str, Any]):
        """Save experiment configuration"""
        self.config = config
        config_path = self.config_dir / 'config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Config saved: {config_path}")
    
    def start_experiment(self):
        """Mark experiment start time"""
        self.start_time = time.time()
        print(f"\n{'='*80}")
        print(f"[START] Experiment: {self.experiment_name}")
        print(f"[START] Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """
        Log metrics for an epoch
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics (e.g., {'train_loss': 0.5, 'val_loss': 0.6})
        """
        metrics['epoch'] = epoch
        metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.metrics_history.append(metrics)
        
        # Print metrics
        metric_str = ' | '.join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                  for k, v in metrics.items()])
        print(f"[EPOCH {epoch:3d}] {metric_str}")
    
    def log_test_results(self, results: Dict[str, Any]):
        """
        Log test results
        
        Args:
            results: Dictionary containing test metrics
        """
        self.best_metrics = results
        
        # Save to JSON
        results_path = self.result_dir / 'test_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"[TEST RESULTS]")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print(f"{'='*80}\n")
    
    def save_checkpoint(self, model, epoch: int, metrics: Dict[str, float], 
                       training_params: Dict[str, Any], data_params: Dict[str, Any],
                       model_params: Dict[str, Any]):
        """
        Save model checkpoint with complete configuration
        
        Args:
            model: PyTorch model
            epoch: Current epoch
            metrics: Current metrics (test_rmse, test_score, val_loss)
            training_params: Training parameters (lr, batch_size, etc.)
            data_params: Data parameters (window_sample, patch_size, etc.)
            model_params: Model parameters (hidden_dim, num_blocks, etc.)
        """
        # Create checkpoint filename with epoch and score
        score = metrics.get('test_score', 0)
        checkpoint_name = f"best_model_epoch{epoch:03d}_score{score:.0f}.pth"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Build comprehensive checkpoint
        checkpoint = {
            'epoch': epoch,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_state_dict': model.state_dict(),
            
            # Performance metrics
            'metrics': metrics,
            
            # Training parameters
            'training_params': training_params,
            
            # Data parameters
            'data_params': data_params,
            
            # Model parameters
            'model_params': model_params,
            
            # Full config for reference
            'full_config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"[SAVE] Checkpoint saved: {checkpoint_path}")
        print(f"       Epoch: {epoch} | Score: {score:.2f} | RMSE: {metrics.get('test_rmse', 0):.4f}")
    
    def save_predictions(self, predictions: np.ndarray, targets: np.ndarray, 
                        dataset_split: str = 'test'):
        """
        Save predictions and targets
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            dataset_split: Dataset split name (train/val/test)
        """
        pred_path = self.result_dir / f'{dataset_split}_predictions.npz'
        np.savez(pred_path, predictions=predictions, targets=targets)
        
        # Also save as CSV for easy viewing
        df = pd.DataFrame({
            'predictions': predictions.flatten(),
            'targets': targets.flatten(),
            'error': predictions.flatten() - targets.flatten()
        })
        csv_path = self.result_dir / f'{dataset_split}_predictions.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"[SAVE] Predictions saved: {pred_path}")
    
    def save_attention_weights(self, attention_weights: np.ndarray, 
                               filename: str = 'attention_weights.npz'):
        """
        Save attention weights for visualization
        
        Args:
            attention_weights: Attention weight arrays
            filename: Output filename
        """
        attn_path = self.result_dir / filename
        np.savez(attn_path, attention=attention_weights)
        print(f"[SAVE] Attention weights saved: {attn_path}")
    
    def save_training_history(self):
        """Save training history to CSV"""
        if not self.metrics_history:
            return
        
        df = pd.DataFrame(self.metrics_history)
        history_path = self.log_dir / 'training_history.csv'
        df.to_csv(history_path, index=False)
        print(f"[SAVE] Training history saved: {history_path}")
    
    def end_experiment(self, status: str = 'completed'):
        """
        Mark experiment end and generate summary
        
        Args:
            status: Experiment status ('completed', 'failed', 'interrupted')
        """
        self.end_time = time.time()
        duration = self.end_time - self.start_time if self.start_time else 0
        
        # Save training history
        self.save_training_history()
        
        # Create experiment summary
        summary = {
            'experiment_name': self.experiment_name,
            'status': status,
            'start_time': datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S') if self.start_time else None,
            'end_time': datetime.fromtimestamp(self.end_time).strftime('%Y-%m-%d %H:%M:%S') if self.end_time else None,
            'duration_seconds': duration,
            'duration_formatted': f"{duration//3600:.0f}h {(duration%3600)//60:.0f}m {duration%60:.0f}s",
            'total_epochs': len(self.metrics_history),
            'best_metrics': self.best_metrics,
            'config': self.config
        }
        
        summary_path = self.exp_dir / 'experiment_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"[END] Experiment: {self.experiment_name}")
        print(f"[END] Status: {status}")
        print(f"[END] Duration: {summary['duration_formatted']}")
        print(f"[END] Summary saved: {summary_path}")
        print(f"{'='*80}\n")
    
    def get_checkpoint_path(self, checkpoint_name: str = 'best_model.pth') -> Path:
        """Get path to checkpoint file"""
        return self.checkpoint_dir / checkpoint_name
    
    def get_result_path(self, filename: str) -> Path:
        """Get path to result file"""
        return self.result_dir / filename


def load_experiment_summary(exp_dir: str) -> Dict[str, Any]:
    """
    Load experiment summary from directory
    
    Args:
        exp_dir: Experiment directory path
        
    Returns:
        Dictionary containing experiment summary
    """
    summary_path = Path(exp_dir) / 'experiment_summary.json'
    
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    return summary


def list_experiments(base_dir: str = './experiments/runs') -> pd.DataFrame:
    """
    List all experiments with their summaries
    
    Args:
        base_dir: Base directory containing experiment runs
        
    Returns:
        DataFrame containing experiment information
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"No experiments found in {base_dir}")
        return pd.DataFrame()
    
    experiments = []
    
    for exp_dir in base_path.iterdir():
        if exp_dir.is_dir():
            summary_path = exp_dir / 'experiment_summary.json'
            if summary_path.exists():
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                    experiments.append({
                        'experiment_name': summary.get('experiment_name', exp_dir.name),
                        'status': summary.get('status', 'unknown'),
                        'start_time': summary.get('start_time', ''),
                        'duration': summary.get('duration_formatted', ''),
                        'rmse': summary.get('best_metrics', {}).get('rmse', None),
                        'score': summary.get('best_metrics', {}).get('score', None),
                        'directory': str(exp_dir)
                    })
    
    if not experiments:
        return pd.DataFrame()
    
    df = pd.DataFrame(experiments)
    df = df.sort_values('start_time', ascending=False)
    
    return df

