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
    加载预训练模型
    
    Args:
        model_path: 模型路径（包含config.json和model.pt的目录）
        device: 设备（如果为None，自动选择）
    
    Returns:
        model: 加载的模型
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载配置
    config = TimerConfig.from_pretrained(model_path)
    
    # 创建模型
    model = TimerForPrediction(config)
    
    # 加载权重
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

