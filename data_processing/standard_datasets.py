"""
Standard time series dataset support
Supports ETTH1, ECL, TRAFFIC, WEATHER, PEMS03, PEMS04 and other datasets
"""
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from typing import Tuple, Optional, List
import pickle


class StandardTimeSeriesDataset(Dataset):
    """
    Standard time series dataset
    Supports multiple standard time series forecasting datasets
    """
    
    # Dataset information
    DATASET_INFO = {
        'ETTH1': {
            'url': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv',
            'target_cols': None,  # Use all columns
            'date_col': 'date'
        },
        'ETTH2': {
            'url': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv',
            'target_cols': None,
            'date_col': 'date'
        },
        'ETTM1': {
            'url': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv',
            'target_cols': None,
            'date_col': 'date'
        },
        'ETTM2': {
            'url': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv',
            'target_cols': None,
            'date_col': 'date'
        },
        'ECL': {
            'url': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ECL/ECL.csv',
            'target_cols': None,
            'date_col': 'date'
        },
        'TRAFFIC': {
            'url': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/Traffic/Traffic.csv',
            'target_cols': None,
            'date_col': 'date'
        },
        'WEATHER': {
            'url': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/Weather/Weather.csv',
            'target_cols': None,
            'date_col': 'date'
        },
        'PEMS03': {
            'url': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/PEMS/PEMS03.csv',
            'target_cols': None,
            'date_col': None
        },
        'PEMS04': {
            'url': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/PEMS/PEMS04.csv',
            'target_cols': None,
            'date_col': None
        },
        'PEMS07': {
            'url': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/PEMS/PEMS07.csv',
            'target_cols': None,
            'date_col': None
        },
        'PEMS08': {
            'url': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/PEMS/PEMS08.csv',
            'target_cols': None,
            'date_col': None
        },
    }
    
    def __init__(
        self,
        data: np.ndarray,
        lookback: int = 672,
        pred_len: int = 96,
        flag: str = 'train',
        scale: bool = True,
        train_mean: Optional[float] = None,
        train_std: Optional[float] = None
    ):
        """
        Args:
            data: 时间序列数据 (n_samples, n_features)
            lookback: 历史数据长度
            pred_len: 预测长度
            flag: 'train', 'val', or 'test'
            scale: 是否归一化
            train_mean: 训练集均值（用于归一化）
            train_std: 训练集标准差（用于归一化）
        """
        self.lookback = lookback
        self.pred_len = pred_len
        self.flag = flag
        self.scale = scale
        
        # 归一化
        if scale:
            if train_mean is None or train_std is None:
                # 训练集：使用自身统计量
                self.mean = np.mean(data, axis=0, keepdims=True)
                self.std = np.std(data, axis=0, keepdims=True)
                self.std[self.std < 1e-8] = 1.0  # 避免除零
            else:
                # 验证/测试集：使用训练集统计量
                self.mean = train_mean
                self.std = train_std
            
            self.data = (data - self.mean) / self.std
        else:
            self.data = data
            self.mean = None
            self.std = None
        
        # 创建滑动窗口样本
        self.sequences = []
        total_len = lookback + pred_len
        
        for i in range(len(data) - total_len + 1):
            seq = self.data[i:i + total_len]
            self.sequences.append(seq)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # First lookback points as input
        history = seq[:self.lookback]
        # Last pred_len points as target
        target = seq[self.lookback:self.lookback + self.pred_len]
        
        return torch.tensor(history, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
    
    def inverse_transform(self, data):
        """Denormalize"""
        if self.scale and self.mean is not None and self.std is not None:
            return data * self.std + self.mean
        return data


def download_dataset(dataset_name: str, data_dir: str = "data/standard_datasets") -> str:
    """
    Download standard dataset
    
    Args:
        dataset_name: Dataset name
        data_dir: Data save directory
    
    Returns:
        csv_path: CSV file path
    """
    import urllib.request
    
    os.makedirs(data_dir, exist_ok=True)
    
    if dataset_name not in StandardTimeSeriesDataset.DATASET_INFO:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(StandardTimeSeriesDataset.DATASET_INFO.keys())}")
    
    csv_path = os.path.join(data_dir, f"{dataset_name}.csv")
    
    # If file already exists, return directly
    if os.path.exists(csv_path):
        print(f"Dataset {dataset_name} already exists at {csv_path}")
        return csv_path
    
    # Download dataset
    url = StandardTimeSeriesDataset.DATASET_INFO[dataset_name]['url']
    print(f"Downloading {dataset_name} from {url}...")
    
    try:
        urllib.request.urlretrieve(url, csv_path)
        print(f"Downloaded {dataset_name} to {csv_path}")
    except Exception as e:
        print(f"Error downloading {dataset_name}: {str(e)}")
        raise
    
    return csv_path


def load_standard_dataset(
    dataset_name: str,
    lookback: int = 672,
    pred_len: int = 96,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    data_dir: str = "data/standard_datasets",
    download: bool = True
) -> Tuple[StandardTimeSeriesDataset, StandardTimeSeriesDataset, StandardTimeSeriesDataset, dict]:
    """
    Load standard time series dataset
    
    Args:
        dataset_name: Dataset name (ETTH1, ECL, TRAFFIC, WEATHER, PEMS03, PEMS04, etc.)
        lookback: Historical data length
        pred_len: Prediction length
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        data_dir: Data directory
        download: Whether to download if dataset doesn't exist
    
    Returns:
        train_dataset, val_dataset, test_dataset, data_config
    """
    # Download dataset (if needed)
    if download:
        csv_path = download_dataset(dataset_name, data_dir)
    else:
        csv_path = os.path.join(data_dir, f"{dataset_name}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found: {csv_path}. Set download=True to download.")
    
    # Read CSV file
    print(f"Loading {dataset_name} from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Get date column (if exists)
    info = StandardTimeSeriesDataset.DATASET_INFO.get(dataset_name, {})
    date_col = info.get('date_col', None)
    
    # Remove date column (if exists)
    if date_col and date_col in df.columns:
        df = df.drop(columns=[date_col])
    
    # Convert to numpy array
    data = df.values.astype(np.float32)
    
    print(f"Dataset shape: {data.shape}")
    print(f"Features: {data.shape[1] if len(data.shape) > 1 else 1}")
    
    # Split dataset
    n_samples = len(data)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    # Create datasets (training set used to calculate normalization statistics)
    train_dataset = StandardTimeSeriesDataset(
        train_data,
        lookback=lookback,
        pred_len=pred_len,
        flag='train',
        scale=True
    )
    
    # Validation and test sets use training set statistics
    val_dataset = StandardTimeSeriesDataset(
        val_data,
        lookback=lookback,
        pred_len=pred_len,
        flag='val',
        scale=True,
        train_mean=train_dataset.mean,
        train_std=train_dataset.std
    )
    
    test_dataset = StandardTimeSeriesDataset(
        test_data,
        lookback=lookback,
        pred_len=pred_len,
        flag='test',
        scale=True,
        train_mean=train_dataset.mean,
        train_std=train_dataset.std
    )
    
    # Data configuration
    data_config = {
        'dataset_name': dataset_name,
        'lookback': lookback,
        'pred_len': pred_len,
        'n_features': data.shape[1] if len(data.shape) > 1 else 1,
        'n_train': len(train_dataset),
        'n_val': len(val_dataset),
        'n_test': len(test_dataset),
        'train_mean': train_dataset.mean,
        'train_std': train_dataset.std,
        'scale': True
    }
    
    return train_dataset, val_dataset, test_dataset, data_config


def prepare_multiple_datasets(
    dataset_names: List[str],
    lookback: int = 672,
    pred_len: int = 96,
    data_dir: str = "data/standard_datasets",
    output_dir: str = "data/standard_datasets_combined",
    download: bool = True
) -> Tuple[StandardTimeSeriesDataset, StandardTimeSeriesDataset, StandardTimeSeriesDataset, dict]:
    """
    准备多个数据集并合并
    
    Args:
        dataset_names: 数据集名称列表
        lookback: 历史数据长度
        pred_len: 预测长度
        data_dir: 数据目录
        output_dir: 输出目录
        download: 是否下载数据集
    
    Returns:
        train_dataset, val_dataset, test_dataset, data_config
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_train_datasets = []
    all_val_datasets = []
    all_test_datasets = []
    all_configs = []
    
    for dataset_name in dataset_names:
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name}")
        print(f"{'='*60}")
        
        train_ds, val_ds, test_ds, config = load_standard_dataset(
            dataset_name=dataset_name,
            lookback=lookback,
            pred_len=pred_len,
            data_dir=data_dir,
            download=download
        )
        
        all_train_datasets.append(train_ds)
        all_val_datasets.append(val_ds)
        all_test_datasets.append(test_ds)
        all_configs.append(config)
    
    # Merge sequences from all datasets
    from torch.utils.data import ConcatDataset
    
    combined_train = ConcatDataset(all_train_datasets)
    combined_val = ConcatDataset(all_val_datasets)
    combined_test = ConcatDataset(all_test_datasets)
    
    # Merge configuration
    combined_config = {
        'datasets': dataset_names,
        'lookback': lookback,
        'pred_len': pred_len,
        'n_train': len(combined_train),
        'n_val': len(combined_val),
        'n_test': len(combined_test),
        'individual_configs': all_configs
    }
    
    # Save configuration
    config_path = os.path.join(output_dir, "data_config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(combined_config, f)
    
    print(f"\n{'='*60}")
    print(f"Combined Dataset Summary")
    print(f"{'='*60}")
    print(f"Datasets: {', '.join(dataset_names)}")
    print(f"Total train samples: {len(combined_train)}")
    print(f"Total val samples: {len(combined_val)}")
    print(f"Total test samples: {len(combined_test)}")
    print(f"Lookback: {lookback}, Prediction length: {pred_len}")
    print(f"[OK] Config saved to {config_path}")
    
    return combined_train, combined_val, combined_test, combined_config

