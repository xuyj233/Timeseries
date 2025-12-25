"""
Data loader utilities
"""
import pickle
import os
from torch.utils.data import DataLoader
from .dataset import TimeSeriesDataset


def create_dataloaders(data_dir, batch_size, num_workers=0, pin_memory=True):
    """
    Create training, validation and test data loaders (for local data)
    
    Args:
        data_dir: Data directory
        batch_size: Batch size
        num_workers: Number of data loader worker processes
        pin_memory: Whether to use pin_memory
    
    Returns:
        train_loader, val_loader, test_loader, data_config
    """
    # Load data configuration
    config_path = os.path.join(data_dir, "data_config.pkl")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Data config not found: {config_path}")
    
    with open(config_path, "rb") as f:
        data_config = pickle.load(f)
    
    lookback = data_config['lookback']
    pred_len = data_config['pred_len']
    
    # Load datasets
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
    
    # Create dataset objects
    train_dataset = TimeSeriesDataset(train_segments, lookback, pred_len)
    val_dataset = TimeSeriesDataset(val_segments, lookback, pred_len)
    test_dataset = TimeSeriesDataset(test_segments, lookback, pred_len)
    
    # Create data loaders
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

