"""
数据准备脚本：将时间序列数据转换为 TIMER 模型训练所需的格式
"""
import os
import sys

# 设置镜像（如果需要加载模型相关配置）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path

# 添加父目录到路径，以便导入数据
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TimeSeriesDataset(Dataset):
    """时间序列数据集类"""
    def __init__(self, sequences, lookback, pred_len):
        """
        Args:
            sequences: 时间序列数据 (list of arrays)
            lookback: 历史数据长度
            pred_len: 预测长度
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
        # 后 pred_len 个点作为目标（teacher forcing 训练）
        target = seq[self.lookback:self.lookback + self.pred_len]
        
        return torch.tensor(history, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def load_data(csv_path):
    """加载原始数据并转换为收益率序列"""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 最后一列（Close价格）
    close = df[df.columns[-1]]
    print(f"Close price range: {close.min():.2f} - {close.max():.2f}")
    
    # 转收益率
    returns = close.pct_change().dropna().reset_index(drop=True)
    series = returns.astype(float).values
    
    print(f"Loaded {len(series)} return points.")
    print(f"Return statistics: mean={series.mean():.6f}, std={series.std():.6f}, "
          f"min={series.min():.6f}, max={series.max():.6f}")
    
    return series


def create_segments(series, lookback, pred_len, stride=None, max_samples=None):
    """
    将时间序列拆分为训练片段
    
    Args:
        series: 时间序列数组
        lookback: 历史数据长度
        pred_len: 预测长度
        stride: 滑动步长（None 表示非重叠）
        max_samples: 最大样本数（None 表示使用所有可能样本）
    
    Returns:
        segments: 片段列表，每个片段长度为 lookback + pred_len
    """
    if stride is None:
        stride = lookback + pred_len  # 非重叠
    
    segments = []
    segment_length = lookback + pred_len
    
    i = 0
    while i + segment_length <= len(series):
        segment = series[i:i + segment_length]
        segments.append(segment)
        i += stride
        
        if max_samples and len(segments) >= max_samples:
            break
    
    return segments


def split_train_val_test(segments, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    划分训练集、验证集和测试集
    
    Args:
        segments: 所有片段
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    
    Returns:
        train_segments, val_segments, test_segments
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    n = len(segments)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_segments = segments[:n_train]
    val_segments = segments[n_train:n_train + n_val]
    test_segments = segments[n_train + n_val:]
    
    return train_segments, val_segments, test_segments


def prepare_datasets(csv_path, lookback=512, pred_len=96, stride=None, 
                     max_samples=None, train_ratio=0.7, val_ratio=0.15, 
                     test_ratio=0.15, output_dir="data"):
    """
    准备完整的数据集
    
    Args:
        csv_path: 原始数据 CSV 路径
        lookback: 历史数据长度
        pred_len: 预测长度
        stride: 滑动步长
        max_samples: 最大样本数
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        output_dir: 输出目录
    
    Returns:
        train_dataset, val_dataset, test_dataset, scaler_info
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    series = load_data(csv_path)
    
    # 数据标准化（可选，TIMER 通常不需要，但可以尝试）
    # 这里我们保存统计信息，但不进行标准化，让模型自己学习
    scaler_info = {
        'mean': float(series.mean()),
        'std': float(series.std()),
        'min': float(series.min()),
        'max': float(series.max())
    }
    
    # 创建片段
    print(f"\nCreating segments with lookback={lookback}, pred_len={pred_len}, stride={stride}")
    segments = create_segments(series, lookback, pred_len, stride, max_samples)
    print(f"Created {len(segments)} segments")
    
    # 划分数据集
    train_segments, val_segments, test_segments = split_train_val_test(
        segments, train_ratio, val_ratio, test_ratio
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_segments)} segments")
    print(f"  Val: {len(val_segments)} segments")
    print(f"  Test: {len(test_segments)} segments")
    
    # 创建数据集对象
    train_dataset = TimeSeriesDataset(train_segments, lookback, pred_len)
    val_dataset = TimeSeriesDataset(val_segments, lookback, pred_len)
    test_dataset = TimeSeriesDataset(test_segments, lookback, pred_len)
    
    # 保存数据集
    print(f"\nSaving datasets to {output_dir}...")
    with open(os.path.join(output_dir, "train_dataset.pkl"), "wb") as f:
        pickle.dump(train_segments, f)
    with open(os.path.join(output_dir, "val_dataset.pkl"), "wb") as f:
        pickle.dump(val_segments, f)
    with open(os.path.join(output_dir, "test_dataset.pkl"), "wb") as f:
        pickle.dump(test_segments, f)
    
    # 保存统计信息
    with open(os.path.join(output_dir, "scaler_info.pkl"), "wb") as f:
        pickle.dump(scaler_info, f)
    
    # 保存配置信息
    config = {
        'lookback': lookback,
        'pred_len': pred_len,
        'stride': stride,
        'n_train': len(train_segments),
        'n_val': len(val_segments),
        'n_test': len(test_segments),
        'scaler_info': scaler_info
    }
    with open(os.path.join(output_dir, "data_config.pkl"), "wb") as f:
        pickle.dump(config, f)
    
    print("[OK] Datasets saved successfully!")
    
    return train_dataset, val_dataset, test_dataset, scaler_info


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare data for TIMER fine-tuning")
    parser.add_argument("--csv-path", type=str, 
                       default=r"data\selected_factors.csv",
                       help="Path to input CSV file")
    parser.add_argument("--lookback", type=int, default=512,
                       help="Lookback length")
    parser.add_argument("--pred-len", type=int, default=96,
                       help="Prediction length")
    parser.add_argument("--stride", type=int, default=None,
                       help="Stride for sliding window (None for non-overlapping)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                       help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                       help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                       help="Test set ratio")
    parser.add_argument("--output-dir", type=str, default="data",
                       help="Output directory")
    
    args = parser.parse_args()
    
    prepare_datasets(
        csv_path=args.csv_path,
        lookback=args.lookback,
        pred_len=args.pred_len,
        stride=args.stride,
        max_samples=args.max_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        output_dir=args.output_dir
    )

