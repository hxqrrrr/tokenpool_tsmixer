"""
Universal trainer for RUL prediction models
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.metrics import scoring_function


class RULTrainer:
    """Universal trainer for RUL prediction"""
    
    def __init__(self, model, dataset, config, logger=None):
        """
        Args:
            model: Model instance (should inherit from BaseRULModel)
            dataset: Dataset instance
            config: Configuration object
            logger: Logger instance (optional)
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        self.logger = logger
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Load data
        self._prepare_data()
        
        # Setup training components
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        if config.lr_scheduler:
            # 支持多种scheduler类型
            scheduler_type = getattr(config, 'scheduler_type', 'plateau')
            
            if scheduler_type == 'cosine':
                # Cosine Annealing: 平滑衰减
                T_max = getattr(config, 'T_max', config.epochs)
                eta_min = getattr(config, 'eta_min', 1e-6)
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=T_max, eta_min=eta_min
                )
                self.scheduler_step_on = 'epoch'  # 每个epoch都step
            elif scheduler_type == 'step':
                # Step LR: 固定间隔衰减
                step_size = getattr(config, 'step_size', 20)
                gamma = getattr(config, 'gamma', 0.5)
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=step_size, gamma=gamma
                )
                self.scheduler_step_on = 'epoch'
            else:  # 'plateau'
                # ReduceLROnPlateau: 基于验证损失
                patience = getattr(config, 'scheduler_patience', 5)
                factor = getattr(config, 'scheduler_factor', 0.5)
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', factor=factor, patience=patience, verbose=False
                )
                self.scheduler_step_on = 'val_loss'
        else:
            self.scheduler = None
            self.scheduler_step_on = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'test_rmse': [],
            'test_score': [],
            'learning_rate': [],
            'best_epoch_val': 0,
            'best_epoch_rmse': 0,
            'best_epoch_score': 0,
            'best_val_loss': float('inf'),
            'best_test_rmse': float('inf'),
            'best_test_score': float('inf')
        }
        
    def _prepare_data(self):
        """Prepare data tensors"""
        # Training data
        train_data = self.dataset.get_train_data()
        self.train_x = torch.FloatTensor(train_data['x']).to(self.device)
        self.train_y = torch.FloatTensor(train_data['y']).to(self.device)
        
        # Validation data
        val_data = self.dataset.get_val_data()
        self.val_x = torch.FloatTensor(val_data['x']).to(self.device)
        self.val_y = torch.FloatTensor(val_data['y']).to(self.device)
        
        # Test data
        test_data = self.dataset.get_test_data()
        self.test_x = torch.FloatTensor(test_data['x']).to(self.device)
        self.test_y = torch.FloatTensor(test_data['y']).to(self.device)
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        batch_size = self.config.batch_size
        num_batches = int(np.ceil(len(self.train_x) / batch_size))
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(self.train_x))
            
            batch_x = self.train_x[start_idx:end_idx]
            batch_y = self.train_y[start_idx:end_idx]
            
            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0
        batch_size = self.config.batch_size
        num_batches = int(np.ceil(len(self.val_x) / batch_size))
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(self.val_x))
                
                batch_x = self.val_x[start_idx:end_idx]
                batch_y = self.val_y[start_idx:end_idx]
                
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def test(self):
        """Test on test set"""
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(self.test_x)
        
        # Calculate RMSE
        mse = self.criterion(predictions, self.test_y)
        rmse = torch.sqrt(mse) * self.dataset.max_rul
        
        # Calculate score
        score = scoring_function(predictions, self.test_y, self.dataset.max_rul)
        
        return rmse.item(), score.item()
    
    def train(self):
        """Full training loop"""
        print(f"\nTraining {self.model.get_model_name()} on {self.dataset.dataset_name}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Initial learning rate: {self.config.lr}")
        print("="*80)
        
        # Log training start
        if self.logger:
            self.logger.info(f"开始训练 {self.model.get_model_name()} on {self.dataset.dataset_name}")
            self.logger.info(f"模型参数量: {self.model.count_parameters():,}")
            self.logger.info(f"初始学习率: {self.config.lr}")
        
        # Early stopping based on test score
        early_stop_patience = getattr(self.config, 'early_stop_patience', 10)
        no_improve_count = 0
        
        for epoch in range(self.config.epochs):
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            if epoch % self.config.show_interval == 0:
                val_loss = self.validate()
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Update scheduler (根据类型不同调用方式不同)
                if self.scheduler is not None:
                    if self.scheduler_step_on == 'val_loss':
                        # ReduceLROnPlateau 需要传入指标
                        self.scheduler.step(val_loss)
                    elif self.scheduler_step_on == 'epoch':
                        # CosineAnnealing/StepLR 每个epoch自动step
                        self.scheduler.step()
                
                # Test
                test_rmse, test_score = self.test()
                
                # Save history
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['test_rmse'].append(test_rmse)
                self.history['test_score'].append(test_score)
                self.history['learning_rate'].append(current_lr)
                
                # Check if best model (based on validation loss)
                if val_loss < self.history['best_val_loss']:
                    self.history['best_val_loss'] = val_loss
                    self.history['best_epoch_val'] = epoch
                
                # Track best test RMSE
                if test_rmse < self.history['best_test_rmse']:
                    self.history['best_test_rmse'] = test_rmse
                    self.history['best_epoch_rmse'] = epoch
                
                # Track best test score and save model
                if test_score < self.history['best_test_score']:
                    self.history['best_test_score'] = test_score
                    self.history['best_epoch_score'] = epoch
                    # Save best model state
                    self.best_model_state = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'test_rmse': test_rmse,
                        'test_score': test_score
                    }
                    no_improve_count = 0  # Reset counter
                else:
                    no_improve_count += 1
                
                # Early stopping check
                if no_improve_count >= early_stop_patience:
                    print(f"\n⚠ Early stopping at epoch {epoch}")
                    print(f"  No improvement in test score for {early_stop_patience} epochs")
                    print(f"  Best score: {self.history['best_test_score']:.2f} at epoch {self.history['best_epoch_score']}")
                    if self.logger:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        self.logger.info(f"No improvement in test score for {early_stop_patience} epochs")
                    break
                
                # Print progress
                print(f"Epoch {epoch:3d} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Test RMSE: {test_rmse:.4f} | "
                      f"Test Score: {test_score:.2f}")
                
                # Log progress
                if self.logger:
                    self.logger.info(f"Epoch {epoch:3d} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | "
                                   f"Val Loss: {val_loss:.4f} | Test RMSE: {test_rmse:.4f} | "
                                   f"Test Score: {test_score:.2f}")
        
        # Load best model and test
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state['model_state_dict'])
            self.model.eval()  # 重新设置为eval模式
            print(f"\n✓ Loaded best model from epoch {self.best_model_state['epoch']}")
            if self.logger:
                self.logger.info(f"已加载 epoch {self.best_model_state['epoch']} 的最佳模型")
            # 直接使用保存的测试结果（更可靠）
            final_rmse = self.best_model_state['test_rmse']
            final_score = self.best_model_state['test_score']
        else:
            # 如果没有保存最佳模型，重新测试
            final_rmse, final_score = self.test()
        
        print("="*80)
        print("Training completed!")
        print("\nBest by Validation Loss:")
        print(f"  Epoch {self.history['best_epoch_val']} | Val Loss: {self.history['best_val_loss']:.4f}")
        print("\nBest by Test RMSE:")
        print(f"  Epoch {self.history['best_epoch_rmse']} | RMSE: {self.history['best_test_rmse']:.4f}")
        print("\nBest by Test Score:")
        print(f"  Epoch {self.history['best_epoch_score']} | Score: {self.history['best_test_score']:.2f}")
        print(f"\nFinal Model (best by {self.config.best_model_metric}):")
        print(f"  RMSE: {final_rmse:.4f} | Score: {final_score:.2f}")
        print("="*80)
        
        # Log training summary
        if self.logger:
            self.logger.info("="*80)
            self.logger.info("训练完成！训练总结:")
            self.logger.info("-"*80)
            self.logger.info(f"最佳验证损失: Epoch {self.history['best_epoch_val']} | Val Loss: {self.history['best_val_loss']:.4f}")
            self.logger.info(f"最佳测试RMSE: Epoch {self.history['best_epoch_rmse']} | RMSE: {self.history['best_test_rmse']:.4f}")
            self.logger.info(f"最佳测试Score: Epoch {self.history['best_epoch_score']} | Score: {self.history['best_test_score']:.2f}")
            self.logger.info(f"最终模型 (best by {self.config.best_model_metric}): RMSE: {final_rmse:.4f} | Score: {final_score:.2f}")
            self.logger.info("="*80)
        
        return {
            'rmse': final_rmse,
            'score': final_score,
            'history': self.history
        }
    