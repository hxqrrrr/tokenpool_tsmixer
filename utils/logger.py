"""
训练日志工具
"""
import logging
import os
import sys
from datetime import datetime


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

