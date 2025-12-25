"""
Data preparation script: Convert time series data to format required for TIMER model training
"""
import os
import sys

# Set mirror (if needed for loading model-related configurations)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path

# Add parent directory to path for importing data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TimeSeriesDataset(Dataset):
    """Time series dataset class"""
    def __init__(self, sequences, lookback, pred_len):
        """
        Args:
            sequences: Time series data (list of arrays)
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
        # First lookback points as input
        history = seq[:self.lookback]
        # Last pred_len points as target (teacher forcing training)
        target = seq[self.lookback:self.lookback + self.pred_len]
        
        return torch.tensor(history, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def load_data(csv_path):
    """Load raw data and convert to return series"""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Last column (Close price)
    close = df[df.columns[-1]]
    print(f"Close price range: {close.min():.2f} - {close.max():.2f}")
    
    # Convert to returns
    returns = close.pct_change().dropna().reset_index(drop=True)
    series = returns.astype(float).values
    
    print(f"Loaded {len(series)} return points.")
    print(f"Return statistics: mean={series.mean():.6f}, std={series.std():.6f}, "
          f"min={series.min():.6f}, max={series.max():.6f}")
    
    return series


def create_segments(series, lookback, pred_len, stride=None, max_samples=None):
    """
    Split time series into training segments
    
    Args:
        series: Time series array
        lookback: Historical data length
        pred_len: Prediction length
        stride: Sliding step size (None means non-overlapping)
        max_samples: Maximum number of samples (None means use all possible samples)
    
    Returns:
        segments: List of segments, each segment length is lookback + pred_len
    """
    if stride is None:
        stride = lookback + pred_len  # Non-overlapping
    
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
    Split into training, validation and test sets
    
    Args:
        segments: All segments
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
    
    Returns:
        train_segments, val_segments, test_segments
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
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
    Prepare complete dataset
    
    Args:
        csv_path: Raw data CSV path
        lookback: Historical data length
        pred_len: Prediction length
        stride: Sliding step size
        max_samples: Maximum number of samples
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        output_dir: Output directory
    
    Returns:
        train_dataset, val_dataset, test_dataset, scaler_info
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    series = load_data(csv_path)
    
    # Data standardization (optional, TIMER usually doesn't need it, but can try)
    # Here we save statistics but don't standardize, let the model learn by itself
    scaler_info = {
        'mean': float(series.mean()),
        'std': float(series.std()),
        'min': float(series.min()),
        'max': float(series.max())
    }
    
    # Create segments
    print(f"\nCreating segments with lookback={lookback}, pred_len={pred_len}, stride={stride}")
    segments = create_segments(series, lookback, pred_len, stride, max_samples)
    print(f"Created {len(segments)} segments")
    
    # Split dataset
    train_segments, val_segments, test_segments = split_train_val_test(
        segments, train_ratio, val_ratio, test_ratio
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_segments)} segments")
    print(f"  Val: {len(val_segments)} segments")
    print(f"  Test: {len(test_segments)} segments")
    
    # Create dataset objects
    train_dataset = TimeSeriesDataset(train_segments, lookback, pred_len)
    val_dataset = TimeSeriesDataset(val_segments, lookback, pred_len)
    test_dataset = TimeSeriesDataset(test_segments, lookback, pred_len)
    
    # Save datasets
    print(f"\nSaving datasets to {output_dir}...")
    with open(os.path.join(output_dir, "train_dataset.pkl"), "wb") as f:
        pickle.dump(train_segments, f)
    with open(os.path.join(output_dir, "val_dataset.pkl"), "wb") as f:
        pickle.dump(val_segments, f)
    with open(os.path.join(output_dir, "test_dataset.pkl"), "wb") as f:
        pickle.dump(test_segments, f)
    
    # Save statistics
    with open(os.path.join(output_dir, "scaler_info.pkl"), "wb") as f:
        pickle.dump(scaler_info, f)
    
    # Save configuration information
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

