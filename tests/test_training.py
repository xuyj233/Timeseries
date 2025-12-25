"""
Test training modules
"""
import sys
from pathlib import Path
import torch
import numpy as np
import pytest
import tempfile
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import TimerConfig, TimerForPrediction
from data_processing.dataset import TimeSeriesDataset
from training import Trainer, FineTuneTrainer
from torch.utils.data import DataLoader


class TestTrainer:
    """Test Trainer class"""
    
    def test_trainer_creation(self):
        """Test trainer creation"""
        config = TimerConfig(
            input_token_len=96,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            output_token_lens=[96]
        )
        model = TimerForPrediction(config)
        
        # Create dummy datasets
        train_sequences = [np.random.randn(200) for _ in range(20)]
        val_sequences = [np.random.randn(200) for _ in range(10)]
        
        train_dataset = TimeSeriesDataset(train_sequences, lookback=100, pred_len=50)
        val_dataset = TimeSeriesDataset(val_sequences, lookback=100, pred_len=50)
        
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        
        device = torch.device("cpu")
        train_config = {
            'batch_size': 2,
            'num_epochs': 1,
            'learning_rate': 1e-4,
            'min_learning_rate': 1e-6,
            'weight_decay': 0.01,
            'scheduler_type': 'cosine',
            'lookback': 100,
            'pred_len': 50
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=train_config,
                device=device,
                output_dir=tmpdir
            )
            assert trainer is not None
    
    def test_trainer_train_epoch(self):
        """Test trainer train_epoch"""
        config = TimerConfig(
            input_token_len=96,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            output_token_lens=[96]
        )
        model = TimerForPrediction(config)
        
        train_sequences = [np.random.randn(200) for _ in range(10)]
        val_sequences = [np.random.randn(200) for _ in range(5)]
        
        train_dataset = TimeSeriesDataset(train_sequences, lookback=100, pred_len=50)
        val_dataset = TimeSeriesDataset(val_sequences, lookback=100, pred_len=50)
        
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        
        device = torch.device("cpu")
        train_config = {
            'batch_size': 2,
            'num_epochs': 1,
            'learning_rate': 1e-4,
            'min_learning_rate': 1e-6,
            'weight_decay': 0.01,
            'scheduler_type': 'cosine',
            'lookback': 100,
            'pred_len': 50
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=train_config,
                device=device,
                output_dir=tmpdir
            )
            
            # Test train_epoch
            train_loss = trainer.train_epoch(0)
            assert train_loss >= 0
            assert isinstance(train_loss, float)
    
    def test_trainer_validate(self):
        """Test trainer validate"""
        config = TimerConfig(
            input_token_len=96,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            output_token_lens=[96]
        )
        model = TimerForPrediction(config)
        
        train_sequences = [np.random.randn(200) for _ in range(10)]
        val_sequences = [np.random.randn(200) for _ in range(5)]
        
        train_dataset = TimeSeriesDataset(train_sequences, lookback=100, pred_len=50)
        val_dataset = TimeSeriesDataset(val_sequences, lookback=100, pred_len=50)
        
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        
        device = torch.device("cpu")
        train_config = {
            'batch_size': 2,
            'num_epochs': 1,
            'learning_rate': 1e-4,
            'min_learning_rate': 1e-6,
            'weight_decay': 0.01,
            'scheduler_type': 'cosine',
            'lookback': 100,
            'pred_len': 50
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=train_config,
                device=device,
                output_dir=tmpdir
            )
            
            # Test validate
            metrics = trainer.validate()
            assert isinstance(metrics, dict)
            assert 'loss' in metrics
            assert metrics['loss'] >= 0


class TestFineTuneTrainer:
    """Test FineTuneTrainer class"""
    
    def test_finetune_trainer_creation(self):
        """Test fine-tune trainer creation"""
        config = TimerConfig(
            input_token_len=96,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            output_token_lens=[96]
        )
        model = TimerForPrediction(config)
        
        train_sequences = [np.random.randn(200) for _ in range(10)]
        val_sequences = [np.random.randn(200) for _ in range(5)]
        
        train_dataset = TimeSeriesDataset(train_sequences, lookback=100, pred_len=50)
        val_dataset = TimeSeriesDataset(val_sequences, lookback=100, pred_len=50)
        
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        
        device = torch.device("cpu")
        train_config = {
            'batch_size': 2,
            'num_epochs': 1,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'warmup_ratio': 0.1,
            'lookback': 100,
            'pred_len': 50
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FineTuneTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=train_config,
                device=device,
                output_dir=tmpdir
            )
            assert trainer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

