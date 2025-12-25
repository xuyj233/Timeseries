"""
Test utility functions
"""
import sys
from pathlib import Path
import torch
import pytest
import tempfile
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import TimerConfig, TimerForPrediction
from utils import count_parameters, load_pretrained_model


class TestModelUtils:
    """Test model utility functions"""
    
    def test_count_parameters(self):
        """Test count_parameters function"""
        config = TimerConfig(
            input_token_len=96,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            output_token_lens=[96]
        )
        model = TimerForPrediction(config)
        
        stats = count_parameters(model)
        
        assert 'total' in stats
        assert 'trainable' in stats
        assert 'frozen' in stats
        assert 'trainable_ratio' in stats
        
        assert stats['total'] > 0
        assert stats['trainable'] > 0
        assert stats['frozen'] == 0  # All parameters should be trainable
        assert 0 <= stats['trainable_ratio'] <= 1
    
    def test_load_pretrained_model(self):
        """Test load_pretrained_model function"""
        config = TimerConfig(
            input_token_len=96,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            output_token_lens=[96]
        )
        model = TimerForPrediction(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model
            config.save_pretrained(tmpdir)
            torch.save(model.state_dict(), os.path.join(tmpdir, "model.pt"))
            
            # Load model
            loaded_model = load_pretrained_model(tmpdir)
            
            assert loaded_model is not None
            assert isinstance(loaded_model, TimerForPrediction)
            
            # Check that parameters match
            original_params = dict(model.named_parameters())
            loaded_params = dict(loaded_model.named_parameters())
            
            for name in original_params:
                assert torch.allclose(
                    original_params[name],
                    loaded_params[name],
                    atol=1e-6
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

