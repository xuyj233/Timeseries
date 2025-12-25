"""
Test model creation and forward pass
"""
import sys
from pathlib import Path
import torch
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import TimerConfig, TimerForPrediction


class TestTimerConfig:
    """Test TimerConfig class"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = TimerConfig()
        assert config.input_token_len == 96
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 8
        assert config.num_attention_heads == 8
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = TimerConfig(
            input_token_len=128,
            hidden_size=512,
            num_hidden_layers=4,
            num_attention_heads=4
        )
        assert config.input_token_len == 128
        assert config.hidden_size == 512
        assert config.num_hidden_layers == 4
        assert config.num_attention_heads == 4
    
    def test_config_save_load(self, tmp_path):
        """Test saving and loading configuration"""
        config = TimerConfig(
            input_token_len=96,
            hidden_size=256
        )
        config.save_pretrained(tmp_path)
        
        loaded_config = TimerConfig.from_pretrained(tmp_path)
        assert loaded_config.input_token_len == config.input_token_len
        assert loaded_config.hidden_size == config.hidden_size


class TestTimerModel:
    """Test TimerForPrediction model"""
    
    def test_model_creation(self):
        """Test model creation"""
        config = TimerConfig(
            input_token_len=96,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            output_token_lens=[96]
        )
        model = TimerForPrediction(config)
        assert model is not None
    
    def test_model_forward(self):
        """Test model forward pass"""
        config = TimerConfig(
            input_token_len=96,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            output_token_lens=[96]
        )
        model = TimerForPrediction(config)
        model.eval()
        
        batch_size = 2
        seq_len = 192  # lookback + pred_len
        input_ids = torch.randn(batch_size, seq_len)  # Use float instead of int
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, return_dict=True)
            assert outputs is not None
            assert hasattr(outputs, 'logits')
    
    def test_model_training_mode(self):
        """Test model in training mode"""
        config = TimerConfig(
            input_token_len=96,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            output_token_lens=[96]
        )
        model = TimerForPrediction(config)
        model.train()
        
        batch_size = 2
        seq_len = 192
        input_ids = torch.randn(batch_size, seq_len)  # Use float instead of int
        labels = input_ids.clone()
        
        outputs = model(input_ids=input_ids, labels=labels, return_dict=True)
        assert outputs is not None
        assert hasattr(outputs, 'loss')
        assert outputs.loss.item() >= 0
    
    def test_model_parameter_count(self):
        """Test model parameter count"""
        config = TimerConfig(
            input_token_len=96,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            output_token_lens=[96]
        )
        model = TimerForPrediction(config)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params == total_params  # All parameters should be trainable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

