"""
Configuration loader for experiments
Supports JSON-based batch experiment configuration
"""
import json
import os
from typing import List, Dict, Any
from pathlib import Path


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """
    Load a single experiment configuration from JSON file
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        Dictionary containing experiment configuration
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Validate required fields
    required_fields = ['experiment_name', 'dataset_name', 'model_params']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    
    return config


def deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries recursively
    
    Args:
        base: Base dictionary
        override: Dictionary with values to override/merge
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge(result[key], value)
        else:
            # Override value
            result[key] = value
    return result


def load_batch_configs(batch_config_path: str) -> List[Dict[str, Any]]:
    """
    Load batch experiment configurations from JSON file
    Supports both 'base_config' (deep merge) and 'global_settings' (shallow merge) formats
    
    Args:
        batch_config_path: Path to batch JSON configuration file
        
    Returns:
        List of experiment configurations
    """
    batch_config_path = Path(batch_config_path)
    
    if not batch_config_path.exists():
        raise FileNotFoundError(f"Batch configuration file not found: {batch_config_path}")
    
    with open(batch_config_path, 'r', encoding='utf-8') as f:
        batch_config = json.load(f)
    
    if 'experiments' not in batch_config:
        raise ValueError("Batch config must contain 'experiments' field")
    
    experiments = batch_config['experiments']
    
    # Support two configuration formats
    if 'base_config' in batch_config:
        # Format 1: base_config with deep merge
        base_config = batch_config['base_config']
        merged_experiments = []
        for exp in experiments:
            # Deep merge: recursively merge nested dictionaries
            merged_exp = deep_merge(base_config, exp)
            merged_experiments.append(merged_exp)
        return merged_experiments
    
    elif 'global_settings' in batch_config:
        # Format 2: global_settings with shallow merge (backward compatible)
        global_settings = batch_config['global_settings']
        for exp in experiments:
            # Shallow merge: only add missing top-level keys
            for key, value in global_settings.items():
                if key not in exp:
                    exp[key] = value
        return experiments
    
    else:
        # No base config or global settings, return experiments as-is
        return experiments


def save_experiment_config(config: Dict[str, Any], output_path: str):
    """
    Save experiment configuration to JSON file
    
    Args:
        config: Experiment configuration dictionary
        output_path: Output JSON file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def create_config_template(output_path: str = 'experiment_template.json'):
    """
    Create a template experiment configuration file
    
    Args:
        output_path: Output path for template file
    """
    template = {
        "experiment_name": "phasepool_baseline",
        "dataset_name": "FD001",
        "description": "Baseline experiment for PhasePool-TokenMixer",
        
        "data_params": {
            "window_sample": 30,
            "patch_size": 5,
            "time_denpen_len": 6,
            "max_rul": 125
        },
        
        "model_params": {
            "num_tokens": 10,
            "token_dim": 128,
            "num_heads": 4,
            "temperature": 1.5,
            "attn_dropout": 0.1,
            "use_pos_encoding": True,
            "hidden_dim": 64,
            "num_blocks": 4,
            "dropout": 0.1
        },
        
        "training_params": {
            "epochs": 50,
            "batch_size": 128,
            "lr": 0.001,
            "weight_decay": 1e-5,
            "lr_scheduler": True,
            "patience": 10,
            "lr_decay_factor": 0.5,
            "lr_patience": 5
        },
        
        "output": {
            "save_checkpoint": True,
            "save_predictions": True,
            "save_attention_weights": False
        }
    }
    
    save_experiment_config(template, output_path)
    print(f"Template created: {output_path}")


def create_batch_config_template(output_path: str = 'batch_experiments.json'):
    """
    Create a template batch experiment configuration file
    
    Args:
        output_path: Output path for batch template file
    """
    batch_template = {
        "description": "Batch experiments for parameter sweep",
        
        "global_settings": {
            "dataset_name": "FD002",
            "data_params": {
                "window_sample": 50,
                "patch_size": 5,
                "time_denpen_len": 10,
                "max_rul": 125
            },
            "training_params": {
                "epochs": 50,
                "batch_size": 128,
                "lr": 0.0008,
                "weight_decay": 2e-5,
                "lr_scheduler": True,
                "patience": 10
            }
        },
        
        "experiments": [
            {
                "experiment_name": "exp001_temperature_1.4",
                "description": "Test temperature=1.4",
                "model_params": {
                    "num_tokens": 16,
                    "token_dim": 64,
                    "num_heads": 4,
                    "temperature": 1.4,
                    "attn_dropout": 0.15,
                    "use_pos_encoding": True,
                    "hidden_dim": 96,
                    "num_blocks": 3,
                    "dropout": 0.15
                }
            },
            {
                "experiment_name": "exp002_temperature_1.6",
                "description": "Test temperature=1.6",
                "model_params": {
                    "num_tokens": 16,
                    "token_dim": 64,
                    "temperature": 1.6,
                    "hidden_dim": 96,
                    "num_blocks": 3
                }
            },
            {
                "experiment_name": "exp003_temperature_1.8",
                "description": "Test temperature=1.8",
                "model_params": {
                    "num_tokens": 16,
                    "token_dim": 64,
                    "temperature": 1.8,
                    "hidden_dim": 96,
                    "num_blocks": 3
                }
            }
        ]
    }
    
    save_experiment_config(batch_template, output_path)
    print(f"Batch template created: {output_path}")


if __name__ == '__main__':
    # Create template files
    create_config_template('experiments/configs/experiment_template.json')
    create_batch_config_template('experiments/configs/batch_experiments_template.json')

