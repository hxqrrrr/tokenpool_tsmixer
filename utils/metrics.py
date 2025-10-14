"""
Evaluation metrics for RUL prediction
"""
import torch
import numpy as np


def scoring_function(predicted, real, max_rul):
    """
    Asymmetric scoring function for RUL prediction
    Penalizes late predictions more heavily than early predictions
    
    Args:
        predicted: Predicted RUL values (normalized)
        real: True RUL values (normalized)
        max_rul: Maximum RUL value for denormalization
        
    Returns:
        Total score
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
    
    return score


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

