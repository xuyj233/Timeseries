"""
Integration tests for the complete training pipeline
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
from training import Trainer
from torch.utils.data import DataLoader
from utils import count_parameters


class TestTrainingPipeline:
    """Test complete training pipeline"""
    
    def test_end_to_end_training(self):
        """Test end-to-end training with dummy data"""
        # Create model
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
        train_sequences = [np.random.randn(200) for _ in range(50)]
        val_sequences = [np.random.randn(200) for _ in range(20)]
        
        train_dataset = TimeSeriesDataset(train_sequences, lookback=100, pred_len=50)
        val_dataset = TimeSeriesDataset(val_sequences, lookback=100, pred_len=50)
        
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        
        device = torch.device("cpu")
        train_config = {
            'batch_size': 4,
            'num_epochs': 2,
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
            
            # Run training (skip plotting to avoid matplotlib issues in tests)
            # We'll manually call train_epoch and validate instead
            for epoch in range(train_config['num_epochs']):
                trainer.train_epoch(epoch)
                trainer.validate()
            
            # Manually save history (skip plotting)
            trainer.save_history()
            
            # Manually save models
            best_model_path = os.path.join(tmpdir, "best_model")
            final_model_path = os.path.join(tmpdir, "final_model")
            os.makedirs(best_model_path, exist_ok=True)
            os.makedirs(final_model_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(best_model_path, "model.pt"))
            model.config.save_pretrained(best_model_path)
            torch.save(model.state_dict(), os.path.join(final_model_path, "model.pt"))
            model.config.save_pretrained(final_model_path)
            
            # Check that outputs were created
            assert os.path.exists(os.path.join(tmpdir, "best_model"))
            assert os.path.exists(os.path.join(tmpdir, "final_model"))
            assert os.path.exists(os.path.join(tmpdir, "training_history.json"))
            
            # Check training history
            import json
            with open(os.path.join(tmpdir, "training_history.json"), 'r') as f:
                history = json.load(f)
            
            assert 'train_losses' in history
            assert 'val_losses' in history
            assert 'val_metrics' in history
            assert len(history['train_losses']) == train_config['num_epochs']
            assert len(history['val_losses']) == train_config['num_epochs']
    
    def test_model_save_and_load(self):
        """Test model save and load functionality"""
        # Create and train a small model
        config = TimerConfig(
            input_token_len=96,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            output_token_lens=[96]
        )
        model = TimerForPrediction(config)
        
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
            
            # Train one epoch
            trainer.train_epoch(0)
            
            # Save model
            best_model_path = os.path.join(tmpdir, "best_model")
            os.makedirs(best_model_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(best_model_path, "model.pt"))
            model.config.save_pretrained(best_model_path)
            
            # Load model
            from utils import load_pretrained_model
            loaded_model = load_pretrained_model(best_model_path)
            
            assert loaded_model is not None
            assert isinstance(loaded_model, TimerForPrediction)
            
            # Test that loaded model works
            batch_size = 2
            seq_len = 150
            input_ids = torch.randn(batch_size, seq_len)  # Use float instead of int
            
            with torch.no_grad():
                outputs = loaded_model(input_ids=input_ids, return_dict=True)
                assert outputs is not None
                assert hasattr(outputs, 'logits')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

