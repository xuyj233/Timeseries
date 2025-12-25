"""
数据加载器工具
"""
import pickle
import os
from torch.utils.data import DataLoader
from .dataset import TimeSeriesDataset


def create_dataloaders(data_dir, batch_size, num_workers=0, pin_memory=True):
    """
    创建训练、验证和测试数据加载器（用于本地数据）
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        num_workers: 数据加载器工作进程数
        pin_memory: 是否使用pin_memory
    
    Returns:
        train_loader, val_loader, test_loader, data_config
    """
    # 加载数据配置
    config_path = os.path.join(data_dir, "data_config.pkl")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Data config not found: {config_path}")
    
    with open(config_path, "rb") as f:
        data_config = pickle.load(f)
    
    lookback = data_config['lookback']
    pred_len = data_config['pred_len']
    
    # 加载数据集
    train_path = os.path.join(data_dir, "train_dataset.pkl")
    val_path = os.path.join(data_dir, "val_dataset.pkl")
    test_path = os.path.join(data_dir, "test_dataset.pkl")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train dataset not found: {train_path}")
    
    with open(train_path, "rb") as f:
        train_segments = pickle.load(f)
    with open(val_path, "rb") as f:
        val_segments = pickle.load(f)
    with open(test_path, "rb") as f:
        test_segments = pickle.load(f)
    
    # 创建数据集对象
    train_dataset = TimeSeriesDataset(train_segments, lookback, pred_len)
    val_dataset = TimeSeriesDataset(val_segments, lookback, pred_len)
    test_dataset = TimeSeriesDataset(test_segments, lookback, pred_len)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader, data_config

