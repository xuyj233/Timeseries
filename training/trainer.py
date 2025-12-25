"""
训练器类
支持预训练和微调
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math


class CosineAnnealingLRWithMin:
    """
    带最小学习率的Cosine Annealing调度器
    实现从base_lr到eta_min的cosine衰减
    """
    def __init__(self, optimizer, T_max, eta_min=0, base_lr=None, last_epoch=-1):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = base_lr if base_lr is not None else optimizer.param_groups[0]['lr']
        self.last_epoch = last_epoch
        self.step_count = 0
    
    def step(self):
        """更新学习率"""
        self.step_count += 1
        if self.step_count <= self.T_max:
            # Cosine annealing: eta = eta_min + (base_lr - eta_min) * (1 + cos(π * step / T_max)) / 2
            lr = self.eta_min + (self.base_lr - self.eta_min) * \
                 (1 + math.cos(math.pi * self.step_count / self.T_max)) / 2
        else:
            lr = self.eta_min
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_last_lr(self):
        """获取当前学习率"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class Trainer:
    """Timer模型训练器"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config,
        device,
        output_dir="outputs"
    ):
        """
        Args:
            model: Timer模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 训练配置
            device: 设备
            output_dir: 输出目录
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 优化器 - 使用AdamW（论文设置）
        base_lr = config.get('learning_rate', 5e-5)  # 论文默认：5e-5
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=base_lr,
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # 学习率调度器 - 使用Cosine Annealing（论文设置）
        num_epochs = config.get('num_epochs', 10)
        num_training_steps = num_epochs * len(train_loader)
        
        # 论文：decay steps proportional to 10 epochs
        # 如果当前epochs不是10，按比例调整
        base_epochs = 10
        if num_epochs != base_epochs:
            # 按比例调整，但至少使用当前epochs
            decay_steps = max(num_training_steps, int(num_training_steps * base_epochs / num_epochs))
        else:
            decay_steps = num_training_steps
        
        # 最终学习率（论文：2e-6）
        min_lr = config.get('min_learning_rate', 2e-6)
        
        # 选择调度器类型
        scheduler_type = config.get('scheduler_type', 'cosine')  # 'cosine' or 'linear'
        
        if scheduler_type == 'cosine':
            # Cosine Annealing调度器
            # 使用CosineAnnealingLR，但需要手动实现从base_lr到min_lr的衰减
            # 创建一个自定义的cosine scheduler
            self.scheduler = CosineAnnealingLRWithMin(
                self.optimizer,
                T_max=decay_steps,
                eta_min=min_lr,
                base_lr=base_lr
            )
        else:
            # Linear schedule with warmup（向后兼容）
            num_warmup_steps = int(num_training_steps * config.get('warmup_ratio', 0.1))
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.val_metrics_history = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        epoch_train_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.get('num_epochs', 10)} [Train]")
        
        for history, target in pbar:
            history = history.to(self.device)
            target = target.to(self.device)
            
            # 拼接输入和目标用于teacher forcing
            # 对于预训练，我们使用完整的序列
            full_sequence = torch.cat([history, target], dim=1)
            
            # 前向传播
            outputs = self.model(
                input_ids=full_sequence,
                labels=full_sequence,  # 自回归预测
                return_dict=True
            )
            
            loss = outputs.loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            epoch_train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        return epoch_train_loss / len(self.train_loader)
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for history, target in tqdm(self.val_loader, desc="Validating"):
                history = history.to(self.device)
                target = target.to(self.device)
                
                # 使用历史数据生成预测
                full_sequence = torch.cat([history, target], dim=1)
                outputs = self.model(
                    input_ids=full_sequence,
                    labels=full_sequence,
                    return_dict=True
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # 获取预测（使用最后一个时间步的预测）
                predictions = outputs.logits
                if predictions is not None:
                    # 提取预测部分
                    if predictions.shape[1] > history.shape[1]:
                        pred_part = predictions[:, history.shape[1]:, :]
                        # 如果是多token输出，取第一个
                        if len(pred_part.shape) == 3 and pred_part.shape[2] > 1:
                            pred_part = pred_part[:, :, 0]
                        all_predictions.append(pred_part.cpu().numpy())
                        all_targets.append(target.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        
        # 计算指标
        if all_predictions:
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            
            # 确保形状匹配
            if all_predictions.shape != all_targets.shape:
                min_len = min(all_predictions.shape[1], all_targets.shape[1])
                all_predictions = all_predictions[:, :min_len]
                all_targets = all_targets[:, :min_len]
            
            mse = np.mean((all_targets - all_predictions) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(all_targets - all_predictions))
            
            # 方向准确率
            first_step_pred = all_predictions[:, 0] if all_predictions.shape[1] > 0 else all_predictions.flatten()
            first_step_true = all_targets[:, 0] if all_targets.shape[1] > 0 else all_targets.flatten()
            direction_acc = np.mean(np.sign(first_step_pred) == np.sign(first_step_true))
        else:
            mse = rmse = mae = direction_acc = 0.0
        
        metrics = {
            'loss': avg_loss,
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'direction_accuracy': float(direction_acc)
        }
        
        return metrics
    
    def train(self):
        """完整训练流程"""
        num_epochs = self.config.get('num_epochs', 10)
        
        print(f"\n{'='*60}")
        print("Starting Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {self.config.get('batch_size', 4)}")
        print(f"Learning rate: {self.config.get('learning_rate', 1e-4)}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"{'='*60}\n")
        
        best_model_path = os.path.join(self.output_dir, "best_model")
        
        for epoch in range(num_epochs):
            # 训练
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # 验证
            val_metrics = self.validate()
            self.val_losses.append(val_metrics['loss'])
            self.val_metrics_history.append(val_metrics)
            
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_metrics['loss']:.6f}")
            print(f"  Val RMSE: {val_metrics['rmse']:.6f}")
            print(f"  Val MAE: {val_metrics['mae']:.6f}")
            print(f"  Val Direction Acc: {val_metrics['direction_accuracy']:.2%}")
            
            # 保存最佳模型
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                print(f"  [OK] New best model! Saving...")
                os.makedirs(best_model_path, exist_ok=True)
                # 保存模型状态和配置
                torch.save(self.model.state_dict(), os.path.join(best_model_path, "model.pt"))
                self.model.config.save_pretrained(best_model_path)
                torch.save(self.optimizer.state_dict(), os.path.join(best_model_path, "optimizer.pt"))
            
            # 保存训练历史
            self.save_history()
        
        # 保存最终模型
        final_model_path = os.path.join(self.output_dir, "final_model")
        os.makedirs(final_model_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(final_model_path, "model.pt"))
        self.model.config.save_pretrained(final_model_path)
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        print(f"\n{'='*60}")
        print("Training Completed!")
        print(f"{'='*60}")
        print(f"Best model saved to: {best_model_path}")
        print(f"Final model saved to: {final_model_path}")
        print(f"{'='*60}\n")
    
    def save_history(self):
        """保存训练历史"""
        history_data = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics_history
        }
        with open(os.path.join(self.output_dir, "training_history.json"), "w") as f:
            json.dump(history_data, f, indent=2)
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(self.train_losses) + 1)
        
        # 损失曲线
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE 曲线
        val_rmse = [m['rmse'] for m in self.val_metrics_history]
        axes[0, 1].plot(epochs, val_rmse, 'g-', label='Val RMSE', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('Validation RMSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # MAE 曲线
        val_mae = [m['mae'] for m in self.val_metrics_history]
        axes[1, 0].plot(epochs, val_mae, 'orange', label='Val MAE', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].set_title('Validation MAE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 方向准确率曲线
        val_dir_acc = [m['direction_accuracy'] for m in self.val_metrics_history]
        axes[1, 1].plot(epochs, val_dir_acc, 'purple', label='Val Direction Acc', linewidth=2)
        axes[1, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random (50%)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Direction Accuracy')
        axes[1, 1].set_title('Validation Direction Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "training_curves.png"), dpi=150, bbox_inches='tight')
        print(f"[OK] Training curves saved to {self.output_dir}/training_curves.png")

