"""
UTSD数据集支持
从HuggingFace下载和加载UTSD数据集
"""
import os
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
import torch


def download_utsd_dataset(
    dataset_name="thuml/UTSD",
    subset=None,
    cache_dir=None,
    use_mirror=True
):
    """
    下载UTSD数据集
    
    Args:
        dataset_name: 数据集名称
        subset: 子集名称（如'UTSD-1G', 'UTSD-2G', 'UTSD-4G', 'UTSD-12G'）
        cache_dir: 缓存目录
        use_mirror: 是否使用镜像
    
    Returns:
        dataset: HuggingFace数据集对象
    """
    if use_mirror:
        # 设置镜像端点
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    print(f"Downloading UTSD dataset: {dataset_name}")
    if subset:
        print(f"Using subset: {subset}")
    
    try:
        if subset:
            dataset = load_dataset(dataset_name, subset, cache_dir=cache_dir)
        else:
            dataset = load_dataset(dataset_name, cache_dir=cache_dir)
        
        print(f"Dataset downloaded successfully!")
        print(f"Train split size: {len(dataset['train'])}")
        return dataset
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        print("\nTrying to use mirror site...")
        # 如果失败，尝试直接使用镜像
        if use_mirror:
            try:
                # 直接使用镜像URL
                mirror_name = dataset_name.replace('thuml/', 'hf-mirror.com/thuml/')
                if subset:
                    dataset = load_dataset(mirror_name, subset, cache_dir=cache_dir)
                else:
                    dataset = load_dataset(mirror_name, cache_dir=cache_dir)
                print(f"Dataset downloaded from mirror successfully!")
                return dataset
            except Exception as e2:
                print(f"Failed to download from mirror: {str(e2)}")
                raise
        else:
            raise


class UTSDDataset(Dataset):
    """
    UTSD数据集适配器
    将HuggingFace格式的UTSD数据集转换为训练格式
    """
    
    def __init__(self, hf_dataset, lookback=512, pred_len=96, max_samples=None):
        """
        Args:
            hf_dataset: HuggingFace数据集对象
            lookback: 历史数据长度
            pred_len: 预测长度
            max_samples: 最大样本数（None表示使用所有）
        """
        self.lookback = lookback
        self.pred_len = pred_len
        self.sequences = []
        
        print(f"Processing UTSD dataset...")
        print(f"Total samples: {len(hf_dataset)}")
        
        # 处理数据
        for idx, sample in enumerate(hf_dataset):
            if max_samples and idx >= max_samples:
                break
            
            # 获取时间序列数据
            # UTSD数据集的字段名是'target'
            target = sample.get('target', None)
            if target is None:
                # 尝试其他可能的字段名
                target = sample.get('sequence', None)
                if target is None:
                    continue
            
            # 转换为numpy数组
            if isinstance(target, list):
                ts_data = np.array(target, dtype=np.float32)
            elif isinstance(target, np.ndarray):
                ts_data = target.astype(np.float32)
            else:
                # 尝试直接转换
                try:
                    ts_data = np.array(target, dtype=np.float32)
                except:
                    continue
            
            # 检查数据长度
            min_length = lookback + pred_len
            if len(ts_data) < min_length:
                continue
            
            # 创建滑动窗口样本
            # 对于UTSD，每个样本可能是一个完整的时间序列
            # 我们使用滑动窗口创建多个训练样本
            stride = lookback + pred_len
            for i in range(0, len(ts_data) - min_length + 1, stride):
                segment = ts_data[i:i + min_length]
                if len(segment) == min_length:
                    self.sequences.append(segment)
        
        print(f"Created {len(self.sequences)} training samples")
        print(f"Lookback: {lookback}, Prediction length: {pred_len}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # 前 lookback 个点作为输入
        history = seq[:self.lookback]
        # 后 pred_len 个点作为目标
        target = seq[self.lookback:self.lookback + self.pred_len]
        
        return torch.tensor(history, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def prepare_utsd_for_training(
    dataset_name="thuml/UTSD",
    subset=None,
    lookback=512,
    pred_len=96,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    max_samples=None,
    cache_dir=None,
    use_mirror=True,
    output_dir="data_cache/utsd"
):
    """
    准备UTSD数据集用于训练
    
    Args:
        dataset_name: 数据集名称
        subset: 子集名称
        lookback: 历史数据长度
        pred_len: 预测长度
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        max_samples: 最大样本数
        cache_dir: 缓存目录
        use_mirror: 是否使用镜像
        output_dir: 输出目录
    
    Returns:
        train_dataset, val_dataset, test_dataset, data_config
    """
    import pickle
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 下载数据集
    dataset = download_utsd_dataset(
        dataset_name=dataset_name,
        subset=subset,
        cache_dir=cache_dir,
        use_mirror=use_mirror
    )
    
    # 获取训练集
    train_hf = dataset['train']
    
    # 创建UTSD数据集
    utsd_dataset = UTSDDataset(
        train_hf,
        lookback=lookback,
        pred_len=pred_len,
        max_samples=max_samples
    )
    
    # 划分数据集
    n = len(utsd_dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n_train + n_val))
    test_indices = list(range(n_train + n_val, n))
    
    # 创建子集
    from torch.utils.data import Subset
    train_dataset = Subset(utsd_dataset, train_indices)
    val_dataset = Subset(utsd_dataset, val_indices)
    test_dataset = Subset(utsd_dataset, test_indices)
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    # 保存配置
    data_config = {
        'lookback': lookback,
        'pred_len': pred_len,
        'dataset_name': dataset_name,
        'subset': subset,
        'n_train': len(train_dataset),
        'n_val': len(val_dataset),
        'n_test': len(test_dataset),
    }
    
    with open(os.path.join(output_dir, "data_config.pkl"), "wb") as f:
        pickle.dump(data_config, f)
    
    print(f"[OK] Data config saved to {output_dir}/data_config.pkl")
    
    return train_dataset, val_dataset, test_dataset, data_config

