"""
Fine-tuning trainer
Supports fine-tuning from pretrained models or HuggingFace models with DeepSpeed acceleration
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
from pathlib import Path

# Add project root to path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from models.timer_model import TimerForPrediction
from models.timer_config import TimerConfig
from utils.model_utils import load_pretrained_model

# DeepSpeed support (optional)
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    deepspeed = None


class FineTuneTrainer:
    """Fine-tuning trainer with DeepSpeed support"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config,
        device,
        output_dir="outputs",
        deepspeed_config=None
    ):
        """
        Args:
            model: Timer model (can be pretrained model or HuggingFace model)
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device
            output_dir: Output directory
            deepspeed_config: DeepSpeed configuration file path (optional)
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.use_deepspeed = deepspeed_config is not None and DEEPSPEED_AVAILABLE
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.use_deepspeed:
            # Initialize DeepSpeed
            print("[INFO] Initializing DeepSpeed...")
            # Load DeepSpeed config
            with open(deepspeed_config, 'r') as f:
                ds_config = json.load(f)
            
            # Update config with training parameters
            base_lr = config.get('learning_rate', 1e-4)
            if 'optimizer' in ds_config and 'params' in ds_config['optimizer']:
                if ds_config['optimizer']['params'].get('lr') == 'auto':
                    ds_config['optimizer']['params']['lr'] = base_lr
                if ds_config['optimizer']['params'].get('weight_decay') == 'auto':
                    ds_config['optimizer']['params']['weight_decay'] = config.get('weight_decay', 0.01)
            
            model_engine, optimizer, _, scheduler = deepspeed.initialize(
                model=model,
                optimizer=optim.AdamW(
                    model.parameters(),
                    lr=base_lr,
                    weight_decay=config.get('weight_decay', 0.01)
                ),
                config=ds_config
            )
            self.model = model_engine
            self.optimizer = optimizer
            self.scheduler = scheduler
            print("[OK] DeepSpeed initialized successfully!")
        else:
            # Standard PyTorch training
            self.model = model.to(device)
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.get('learning_rate', 1e-4),
                weight_decay=config.get('weight_decay', 0.01)
            )
            
            # Learning rate scheduler
            num_epochs = config.get('num_epochs', 10)
            num_training_steps = num_epochs * len(train_loader)
            num_warmup_steps = int(num_training_steps * config.get('warmup_ratio', 0.1))
            
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_metrics_history = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        epoch_train_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.get('num_epochs', 10)} [Train]")
        
        for history, target in pbar:
            if not self.use_deepspeed:
                history = history.to(self.device)
                target = target.to(self.device)
            
            # Concatenate input and target for teacher forcing
            full_sequence = torch.cat([history, target], dim=1)
            
            if self.use_deepspeed:
                # DeepSpeed forward pass
                outputs = self.model(
                    input_ids=full_sequence,
                    labels=full_sequence,
                    return_dict=True
                )
                loss = outputs.loss
                # DeepSpeed backward and step
                self.model.backward(loss)
                self.model.step()
            else:
                # Standard PyTorch forward pass
                outputs = self.model(
                    input_ids=full_sequence,
                    labels=full_sequence,
                    return_dict=True
                )
                loss = outputs.loss
                
                # Standard backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
            
            epoch_train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        return epoch_train_loss / len(self.train_loader)
    
    def validate(self):
        """Validation"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for history, target in tqdm(self.val_loader, desc="Validating"):
                if not self.use_deepspeed:
                    history = history.to(self.device)
                    target = target.to(self.device)
                
                full_sequence = torch.cat([history, target], dim=1)
                
                if self.use_deepspeed:
                    outputs = self.model.module(
                        input_ids=full_sequence,
                        labels=full_sequence,
                        return_dict=True
                    )
                else:
                    outputs = self.model(
                        input_ids=full_sequence,
                        labels=full_sequence,
                        return_dict=True
                    )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                predictions = outputs.logits
                if predictions is not None:
                    if predictions.shape[1] > history.shape[1]:
                        pred_part = predictions[:, history.shape[1]:, :]
                        if len(pred_part.shape) == 3 and pred_part.shape[2] > 1:
                            pred_part = pred_part[:, :, 0]
                        all_predictions.append(pred_part.cpu().numpy())
                        all_targets.append(target.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate metrics
        if all_predictions:
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            
            if all_predictions.shape != all_targets.shape:
                min_len = min(all_predictions.shape[1], all_targets.shape[1])
                all_predictions = all_predictions[:, :min_len]
                all_targets = all_targets[:, :min_len]
            
            mse = np.mean((all_targets - all_predictions) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(all_targets - all_predictions))
            
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
        """Complete training workflow"""
        num_epochs = self.config.get('num_epochs', 10)
        
        print(f"\n{'='*60}")
        print("Starting Fine-tuning")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"DeepSpeed: {'Enabled' if self.use_deepspeed else 'Disabled'}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {self.config.get('batch_size', 4)}")
        print(f"Learning rate: {self.config.get('learning_rate', 1e-4)}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"{'='*60}\n")
        
        best_model_path = os.path.join(self.output_dir, "best_model")
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validation
            val_metrics = self.validate()
            self.val_losses.append(val_metrics['loss'])
            self.val_metrics_history.append(val_metrics)
            
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_metrics['loss']:.6f}")
            print(f"  Val RMSE: {val_metrics['rmse']:.6f}")
            print(f"  Val MAE: {val_metrics['mae']:.6f}")
            print(f"  Val Direction Acc: {val_metrics['direction_accuracy']:.2%}")
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                print(f"  [OK] New best model! Saving...")
                os.makedirs(best_model_path, exist_ok=True)
                
                # Save model
                if self.use_deepspeed:
                    # DeepSpeed model saving
                    self.model.save_checkpoint(best_model_path)
                    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                    if isinstance(model_to_save, TimerForPrediction):
                        model_to_save.config.save_pretrained(best_model_path)
                    else:
                        model_to_save.save_pretrained(best_model_path)
                else:
                    if isinstance(self.model, TimerForPrediction):
                        torch.save(self.model.state_dict(), os.path.join(best_model_path, "model.pt"))
                        self.model.config.save_pretrained(best_model_path)
                    else:
                        # HuggingFace model
                        self.model.save_pretrained(best_model_path)
                    torch.save(self.optimizer.state_dict(), os.path.join(best_model_path, "optimizer.pt"))
            
            # Save training history
            self.save_history()
        
        # Save final model
        final_model_path = os.path.join(self.output_dir, "final_model")
        os.makedirs(final_model_path, exist_ok=True)
        if self.use_deepspeed:
            self.model.save_checkpoint(final_model_path)
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            if isinstance(model_to_save, TimerForPrediction):
                model_to_save.config.save_pretrained(final_model_path)
            else:
                model_to_save.save_pretrained(final_model_path)
        else:
            if isinstance(self.model, TimerForPrediction):
                torch.save(self.model.state_dict(), os.path.join(final_model_path, "model.pt"))
                self.model.config.save_pretrained(final_model_path)
            else:
                self.model.save_pretrained(final_model_path)
        
        # Plot training curves
        self.plot_training_curves()
        
        print(f"\n{'='*60}")
        print("Fine-tuning Completed!")
        print(f"{'='*60}")
        print(f"Best model saved to: {best_model_path}")
        print(f"Final model saved to: {final_model_path}")
        print(f"{'='*60}\n")
    
    def save_history(self):
        """Save training history"""
        history_data = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics_history
        }
        with open(os.path.join(self.output_dir, "training_history.json"), "w") as f:
            json.dump(history_data, f, indent=2)
    
    def plot_training_curves(self):
        """Plot training curves"""
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

