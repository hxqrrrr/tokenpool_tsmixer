"""
Base model class for RUL prediction
All models should inherit from this class
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseRULModel(nn.Module, ABC):
    """
    Base class for all RUL prediction models
    """
    
    def __init__(self):
        super(BaseRULModel, self).__init__()
        
    @abstractmethod
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor, shape depends on model
            
        Returns:
            Predicted RUL values
        """
        pass
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_name(self):
        """Get model name"""
        return self.__class__.__name__
    
    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_name': self.get_model_name(),
        }, path)
    
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint

