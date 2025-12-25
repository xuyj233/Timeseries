"""
Time series dataset
"""
import torch
from torch.utils.data import Dataset
import numpy as np


class TimeSeriesDataset(Dataset):
    """Time series dataset class"""
    
    def __init__(self, sequences, lookback, pred_len):
        """
        Args:
            sequences: List of time series data, each element is an array
            lookback: Historical data length
            pred_len: Prediction length
        """
        self.sequences = sequences
        self.lookback = lookback
        self.pred_len = pred_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # 前 lookback 个点作为输入
        history = seq[:self.lookback]
        # 后 pred_len 个点作为目标
        target = seq[self.lookback:self.lookback + self.pred_len]
        
        return torch.tensor(history, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

