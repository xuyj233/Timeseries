"""
Model utility functions
"""
import os
import torch
import sys
from pathlib import Path

# Add project root to path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from models.timer_config import TimerConfig
from models.timer_model import TimerForPrediction


def load_pretrained_model(model_path, device=None):
    """
    Load pretrained model
    
    Args:
        model_path: Model path (directory containing config.json and model.pt)
        device: Device (if None, auto-select)
    
    Returns:
        model: Loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load configuration
    config = TimerConfig.from_pretrained(model_path)
    
    # Create model
    model = TimerForPrediction(config)
    
    # Load weights
    model_path_pt = os.path.join(model_path, "model.pt")
    if os.path.exists(model_path_pt):
        state_dict = torch.load(model_path_pt, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Model file not found: {model_path_pt}")
    
    model.to(device)
    model.eval()
    
    return model


def count_parameters(model):
    """
    Count model parameters
    
    Args:
        model: Model
    
    Returns:
        dict: Dictionary containing total parameters, trainable parameters, etc.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params,
        'trainable_ratio': trainable_params / total_params if total_params > 0 else 0
    }

