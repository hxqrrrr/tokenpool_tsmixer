"""
Utility script to load and inspect saved checkpoints
Usage: python load_checkpoint.py <checkpoint_path>
"""
import sys
import torch
from pathlib import Path


def load_and_inspect_checkpoint(checkpoint_path):
    """
    Load and display checkpoint information
    
    Args:
        checkpoint_path: Path to checkpoint file
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"Loading checkpoint: {checkpoint_path.name}")
    print(f"{'='*80}\n")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Display basic information
    print(f"[BASIC INFO]")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Timestamp: {checkpoint.get('timestamp', 'N/A')}")
    
    # Display metrics
    print(f"\n[PERFORMANCE METRICS]")
    metrics = checkpoint.get('metrics', {})
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Display training parameters
    print(f"\n[TRAINING PARAMETERS]")
    training_params = checkpoint.get('training_params', {})
    for key, value in training_params.items():
        print(f"  {key}: {value}")
    
    # Display data parameters
    print(f"\n[DATA PARAMETERS]")
    data_params = checkpoint.get('data_params', {})
    for key, value in data_params.items():
        print(f"  {key}: {value}")
    
    # Display model parameters
    print(f"\n[MODEL PARAMETERS]")
    model_params = checkpoint.get('model_params', {})
    for key, value in model_params.items():
        print(f"  {key}: {value}")
    
    # Display model state info
    print(f"\n[MODEL STATE]")
    model_state = checkpoint.get('model_state_dict', {})
    if model_state:
        total_params = sum(p.numel() for p in model_state.values())
        print(f"  Number of parameter tensors: {len(model_state)}")
        print(f"  Total parameters: {total_params:,}")
        print(f"\n  First 5 parameter names:")
        for i, key in enumerate(list(model_state.keys())[:5]):
            print(f"    - {key}: {model_state[key].shape}")
    
    # Display full config if available
    if 'full_config' in checkpoint:
        print(f"\n[FULL CONFIG AVAILABLE]")
        print(f"  Full configuration is stored in checkpoint")
    
    print(f"\n{'='*80}")
    print(f"Checkpoint inspection complete!")
    print(f"{'='*80}\n")


def list_checkpoints(experiment_dir):
    """
    List all checkpoints in an experiment directory
    
    Args:
        experiment_dir: Path to experiment directory
    """
    experiment_dir = Path(experiment_dir)
    checkpoint_dir = experiment_dir / 'checkpoints'
    
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return
    
    checkpoints = sorted(checkpoint_dir.glob('*.pth'))
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"Checkpoints in: {experiment_dir.name}")
    print(f"{'='*80}\n")
    
    for i, ckpt_path in enumerate(checkpoints, 1):
        # Load checkpoint to get info
        ckpt = torch.load(ckpt_path, map_location='cpu')
        epoch = ckpt.get('epoch', 'N/A')
        metrics = ckpt.get('metrics', {})
        score = metrics.get('test_score', 'N/A')
        rmse = metrics.get('test_rmse', 'N/A')
        
        print(f"{i}. {ckpt_path.name}")
        print(f"   Epoch: {epoch} | Score: {score:.2f if isinstance(score, float) else score} | "
              f"RMSE: {rmse:.4f if isinstance(rmse, float) else rmse}")
    
    print(f"\n{'='*80}\n")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Inspect a checkpoint:")
        print("    python load_checkpoint.py <checkpoint_path>")
        print("  List checkpoints in experiment:")
        print("    python load_checkpoint.py --list <experiment_dir>")
        print("\nExamples:")
        print("  python load_checkpoint.py experiments/runs/temp_1.6_20251021/checkpoints/best_model_epoch042_score618.pth")
        print("  python load_checkpoint.py --list experiments/runs/temp_1.6_20251021")
        sys.exit(1)
    
    if sys.argv[1] == '--list':
        if len(sys.argv) < 3:
            print("Error: Please provide experiment directory path")
            sys.exit(1)
        list_checkpoints(sys.argv[2])
    else:
        load_and_inspect_checkpoint(sys.argv[1])


if __name__ == '__main__':
    main()

