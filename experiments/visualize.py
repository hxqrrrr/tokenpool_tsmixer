"""
Visualization tools for experiment results
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


def plot_training_history(exp_dir, save_path=None):
    """
    Plot training history from experiment directory
    
    Args:
        exp_dir: Path to experiment directory
        save_path: Optional path to save the plot
    """
    exp_path = Path(exp_dir)
    history_path = exp_path / 'logs' / 'training_history.csv'
    
    if not history_path.exists():
        print(f"Training history not found: {history_path}")
        return
    
    # Load history
    df = pd.read_csv(history_path)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot train/val loss
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o', markersize=3)
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s', markersize=3)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot test RMSE
    axes[0, 1].plot(df['epoch'], df['test_rmse'], label='Test RMSE', color='green', marker='^', markersize=3)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('Test RMSE over Training')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot test score
    axes[1, 0].plot(df['epoch'], df['test_score'], label='Test Score', color='red', marker='v', markersize=3)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Test Score over Training (lower is better)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot learning rate
    axes[1, 1].plot(df['epoch'], df['learning_rate'], label='Learning Rate', color='orange', marker='d', markersize=3)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    else:
        plt.show()


def plot_predictions(exp_dir, split='test', save_path=None):
    """
    Plot predictions vs targets
    
    Args:
        exp_dir: Path to experiment directory
        split: Dataset split ('train', 'val', 'test')
        save_path: Optional path to save the plot
    """
    exp_path = Path(exp_dir)
    pred_path = exp_path / 'results' / f'{split}_predictions.csv'
    
    if not pred_path.exists():
        print(f"Predictions not found: {pred_path}")
        return
    
    # Load predictions
    df = pd.read_csv(pred_path)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Scatter plot
    axes[0].scatter(df['targets'], df['predictions'], alpha=0.5, s=20)
    min_val = min(df['targets'].min(), df['predictions'].min())
    max_val = max(df['targets'].max(), df['predictions'].max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    axes[0].set_xlabel('True RUL')
    axes[0].set_ylabel('Predicted RUL')
    axes[0].set_title(f'Predictions vs True RUL ({split.upper()})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error distribution
    errors = df['error']
    axes[1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='r', linestyle='--', label='Zero Error')
    axes[1].axvline(errors.mean(), color='g', linestyle='--', label=f'Mean Error: {errors.mean():.2f}')
    axes[1].set_xlabel('Prediction Error')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Error Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    else:
        plt.show()


def plot_attention_weights(exp_dir, num_samples=5, save_path=None):
    """
    Plot attention weights heatmap
    
    Args:
        exp_dir: Path to experiment directory
        num_samples: Number of samples to plot
        save_path: Optional path to save the plot
    """
    exp_path = Path(exp_dir)
    attn_path = exp_path / 'results' / 'attention_weights.npz'
    
    if not attn_path.exists():
        print(f"Attention weights not found: {attn_path}")
        return
    
    # Load attention weights
    data = np.load(attn_path)
    attn = data['attention']  # [batch, heads, num_tokens, seq_len]
    
    # Average over heads
    attn_avg = attn.mean(axis=1)  # [batch, num_tokens, seq_len]
    
    # Select samples
    num_samples = min(num_samples, len(attn_avg))
    
    # Create subplots
    fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 5))
    
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        sns.heatmap(attn_avg[i], ax=axes[i], cmap='YlOrRd', cbar=True, 
                   xticklabels=5, yticklabels=2)
        axes[i].set_xlabel('Time Steps')
        axes[i].set_ylabel('Tokens')
        axes[i].set_title(f'Sample {i+1}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    else:
        plt.show()


def compare_experiments(exp_dirs, metric='rmse', save_path=None):
    """
    Compare multiple experiments
    
    Args:
        exp_dirs: List of experiment directory paths
        metric: Metric to compare ('rmse' or 'score')
        save_path: Optional path to save the plot
    """
    results = []
    
    for exp_dir in exp_dirs:
        exp_path = Path(exp_dir)
        summary_path = exp_path / 'experiment_summary.json'
        
        if not summary_path.exists():
            print(f"Summary not found: {summary_path}")
            continue
        
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        exp_name = summary.get('experiment_name', exp_path.name)
        best_metrics = summary.get('best_metrics', {})
        
        results.append({
            'experiment': exp_name,
            'rmse': best_metrics.get('rmse', np.nan),
            'score': best_metrics.get('score', np.nan)
        })
    
    if not results:
        print("No valid experiment results found")
        return
    
    df = pd.DataFrame(results)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(max(10, len(results)*1.5), 6))
    
    x = np.arange(len(df))
    width = 0.35
    
    if metric == 'rmse':
        bars = ax.bar(x, df['rmse'], width, label='RMSE', color='skyblue', edgecolor='black')
        ax.set_ylabel('RMSE (lower is better)')
        ax.set_title('RMSE Comparison Across Experiments')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom')
    else:
        bars = ax.bar(x, df['score'], width, label='Score', color='lightcoral', edgecolor='black')
        ax.set_ylabel('Score (lower is better)')
        ax.set_title('Score Comparison Across Experiments')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}',
                       ha='center', va='bottom')
    
    ax.set_xlabel('Experiment')
    ax.set_xticks(x)
    ax.set_xticklabels(df['experiment'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize experiment results')
    parser.add_argument('exp_dir', type=str, help='Experiment directory path')
    parser.add_argument('--plot', type=str, default='history',
                       choices=['history', 'predictions', 'attention', 'all'],
                       help='Type of plot to generate')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split for predictions plot')
    parser.add_argument('--save', type=str, help='Path to save the plot')
    
    args = parser.parse_args()
    
    if args.plot == 'history' or args.plot == 'all':
        save_path = args.save if args.plot != 'all' else f"{args.exp_dir}/training_history.png"
        plot_training_history(args.exp_dir, save_path)
    
    if args.plot == 'predictions' or args.plot == 'all':
        save_path = args.save if args.plot != 'all' else f"{args.exp_dir}/predictions_{args.split}.png"
        plot_predictions(args.exp_dir, args.split, save_path)
    
    if args.plot == 'attention' or args.plot == 'all':
        save_path = args.save if args.plot != 'all' else f"{args.exp_dir}/attention_weights.png"
        plot_attention_weights(args.exp_dir, save_path=save_path)

