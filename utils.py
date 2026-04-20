"""
Utility functions for RUL prediction
包含评估指标和日志工具
"""
import torch
import numpy as np
import random
import logging
import os
import sys
from datetime import datetime


# ==================== Evaluation Metrics ====================

def scoring_function(predicted, real, max_rul):
    """
    Asymmetric scoring function for RUL prediction
    Penalizes late predictions more heavily than early predictions
    
    Args:
        predicted: Predicted RUL values (normalized)
        real: True RUL values (normalized)
        max_rul: Maximum RUL value for denormalization
        
    Returns:
        Mean score (N-CMAPSS standard)
    """
    score = 0
    num = predicted.size(0)
    
    for i in range(num):
        pred_denorm = predicted[i] * max_rul
        real_denorm = real[i] * max_rul
        
        if real_denorm > pred_denorm:
            # Late prediction (more dangerous)
            score = score + (torch.exp((real_denorm - pred_denorm) / 13) - 1)
        else:
            # Early prediction (less dangerous)
            score = score + (torch.exp((pred_denorm - real_denorm) / 10) - 1)
    
    # ✅ 修复：返回 Mean Score 而非 Sum Score
    return score / num


def calculate_rmse(predicted, real, max_rul):
    """
    Calculate Root Mean Square Error
    
    Args:
        predicted: Predicted RUL values (normalized)
        real: True RUL values (normalized)
        max_rul: Maximum RUL value for denormalization
        
    Returns:
        RMSE value
    """
    mse = torch.mean((predicted - real) ** 2)
    rmse = torch.sqrt(mse) * max_rul
    return rmse


def phm_score(predicted, real):
    """
    PHM08 Challenge scoring function (for numpy arrays)
    
    Args:
        predicted: Predicted RUL values (denormalized)
        real: True RUL values (denormalized)
        
    Returns:
        Mean score (N-CMAPSS standard)
    """
    predicted = np.array(predicted).flatten()
    real = np.array(real).flatten()
    
    diff = real - predicted
    score = 0
    
    for d in diff:
        if d < 0:
            # Late prediction (more dangerous)
            score += np.exp(-d / 13) - 1
        else:
            # Early prediction (less dangerous)
            score += np.exp(d / 10) - 1
    
    # ✅ 修复：返回 Mean Score 而非 Sum Score
    return score / len(diff)


# ==================== Random Seed ====================

def set_seed(seed=42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================== Logging Utilities ====================

def setup_logger(log_dir='./logs', model_name='model', dataset_name='dataset'):
    """
    设置日志系统
    
    Args:
        log_dir: 日志保存目录
        model_name: 模型名称
        dataset_name: 数据集名称
        
    Returns:
        logger: 配置好的logger对象
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名（带时间戳）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{model_name}_{dataset_name}_{timestamp}.log')
    
    # 创建logger
    logger = logging.getLogger(f'{model_name}_{dataset_name}')
    logger.setLevel(logging.INFO)
    
    # 清除已有的handlers（避免重复）
    logger.handlers.clear()
    
    # 文件handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"日志已初始化，保存到: {log_file}")
    logger.info("="*80)
    
    return logger


def log_config(logger, config):
    """记录配置信息"""
    logger.info("配置参数:")
    logger.info("-"*80)
    for key, value in config.__dict__.items():
        logger.info(f"  {key}: {value}")
    logger.info("-"*80)


def log_dataset_info(logger, dataset):
    """记录数据集信息"""
    logger.info("数据集信息:")
    logger.info("-"*80)
    logger.info(f"  训练样本: {len(dataset.train_x)}")
    logger.info(f"  验证样本: {len(dataset.val_indices)}")
    logger.info(f"  测试样本: {len(dataset.test_x)}")
    logger.info(f"  数据形状: {dataset.train_x.shape}")
    logger.info("-"*80)


def log_model_info(logger, model):
    """记录模型信息"""
    logger.info("模型信息:")
    logger.info("-"*80)
    logger.info(f"  模型名称: {model.get_model_name()}")
    logger.info(f"  参数量: {model.count_parameters():,}")
    logger.info("-"*80)


def log_epoch(logger, epoch, train_loss, val_loss, test_rmse, test_score):
    """记录每个epoch的训练信息"""
    logger.info(
        f"Epoch {epoch:3d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Test RMSE: {test_rmse:.4f} | "
        f"Test Score: {test_score:.2f}"
    )


def log_training_summary(logger, history):
    """记录训练总结"""
    logger.info("="*80)
    logger.info("训练总结:")
    logger.info("-"*80)
    
    # 最佳验证损失
    logger.info(f"最佳验证损失:")
    logger.info(f"  Epoch {history['best_epoch_val']} | "
                f"Val Loss: {history['best_val_loss']:.4f}")
    
    # 最佳RMSE
    logger.info(f"最佳测试RMSE:")
    logger.info(f"  Epoch {history['best_epoch_rmse']} | "
                f"RMSE: {history['best_test_rmse']:.4f}")
    
    # 最佳Score
    logger.info(f"最佳测试Score:")
    logger.info(f"  Epoch {history['best_epoch_score']} | "
                f"Score: {history['best_test_score']:.2f}")
    
    logger.info("-"*80)
    logger.info("="*80)


def log_final_results(logger, rmse, score):
    """记录最终结果"""
    logger.info("最终测试结果:")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  Score: {score:.2f}")
    logger.info("="*80)


# 导出所有公共函数
__all__ = [
    'scoring_function',
    'calculate_rmse',
    'phm_score',
    'set_seed',
    'setup_logger',
    'log_config',
    'log_dataset_info',
    'log_model_info',
    'log_epoch',
    'log_training_summary',
    'log_final_results'
]

