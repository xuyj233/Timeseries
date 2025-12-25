"""
S3 (Single-Series Sequence) 格式预处理
实现论文中描述的S3格式：归一化、合并、采样
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import pickle


class S3Preprocessor:
    """
    S3格式预处理器
    将异构时间序列转换为统一的单序列序列格式
    """
    
    def __init__(
        self,
        context_length: int = 512,
        train_ratio: float = 0.9,
        normalize: bool = True,
        random_seed: int = 42
    ):
        """
        Args:
            context_length: 固定上下文长度（窗口大小）
            train_ratio: 训练集比例（用于计算归一化统计量）
            normalize: 是否归一化
            random_seed: 随机种子
        """
        self.context_length = context_length
        self.train_ratio = train_ratio
        self.normalize = normalize
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # 存储归一化统计量
        self.normalization_stats = {}  # {series_id: {'mean': float, 'std': float}}
        
        # 存储处理后的序列池
        self.sequence_pool = []  # List of normalized single-variate series
    
    def process_variate(
        self,
        series: np.ndarray,
        series_id: str,
        split_point: Optional[int] = None
    ) -> np.ndarray:
        """
        处理单个变量序列：归一化并返回
        
        Args:
            series: 时间序列数组 (1D)
            series_id: 序列标识符
            split_point: 训练/验证分割点（如果为None，自动计算）
        
        Returns:
            normalized_series: 归一化后的序列
        """
        series = series.astype(np.float32)
        
        # 计算分割点（9:1）
        if split_point is None:
            split_point = int(len(series) * self.train_ratio)
        
        # 使用训练集的统计量进行归一化
        train_series = series[:split_point]
        
        if self.normalize:
            mean = np.mean(train_series)
            std = np.std(train_series)
            
            # 避免除零
            if std < 1e-8:
                std = 1.0
            
            # 保存统计量
            self.normalization_stats[series_id] = {
                'mean': float(mean),
                'std': float(std),
                'split_point': split_point
            }
            
            # 归一化整个序列（使用训练集的统计量）
            normalized_series = (series - mean) / std
        else:
            normalized_series = series
            self.normalization_stats[series_id] = {
                'mean': 0.0,
                'std': 1.0,
                'split_point': split_point
            }
        
        return normalized_series
    
    def merge_to_pool(
        self,
        series_list: List[np.ndarray],
        series_ids: Optional[List[str]] = None
    ):
        """
        将多个归一化后的序列合并到序列池中
        
        Args:
            series_list: 归一化后的序列列表
            series_ids: 序列标识符列表（可选）
        """
        if series_ids is None:
            series_ids = [f"series_{i}" for i in range(len(series_list))]
        
        for series, series_id in zip(series_list, series_ids):
            # 处理每个序列
            normalized_series = self.process_variate(series, series_id)
            
            # 添加到池中
            self.sequence_pool.append({
                'data': normalized_series,
                'id': series_id,
                'length': len(normalized_series)
            })
    
    def sample_sequences(
        self,
        num_samples: Optional[int] = None,
        stride: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        从序列池中均匀采样固定长度的序列
        
        Args:
            num_samples: 采样数量（如果为None，尽可能多采样）
            stride: 采样步长（如果为None，使用context_length）
        
        Returns:
            sampled_sequences: 采样后的序列列表
        """
        if stride is None:
            stride = self.context_length
        
        sampled_sequences = []
        
        # 从每个序列中采样窗口
        for pool_item in self.sequence_pool:
            series = pool_item['data']
            series_length = pool_item['length']
            
            if series_length < self.context_length:
                continue
            
            # 均匀采样窗口
            for start_idx in range(0, series_length - self.context_length + 1, stride):
                window = series[start_idx:start_idx + self.context_length]
                sampled_sequences.append(window)
        
        # 如果指定了采样数量，随机采样
        if num_samples is not None and len(sampled_sequences) > num_samples:
            indices = np.random.choice(
                len(sampled_sequences),
                size=num_samples,
                replace=False
            )
            sampled_sequences = [sampled_sequences[i] for i in indices]
        
        # 打乱顺序
        np.random.shuffle(sampled_sequences)
        
        return sampled_sequences
    
    def process_hf_dataset(
        self,
        hf_dataset,
        max_variates: Optional[int] = None
    ):
        """
        处理HuggingFace格式的UTSD数据集
        
        Args:
            hf_dataset: HuggingFace数据集对象
            max_variates: 最大变量数（用于限制处理数量）
        """
        series_list = []
        series_ids = []
        
        print(f"Processing {len(hf_dataset)} samples from dataset...")
        
        for idx, sample in enumerate(hf_dataset):
            if max_variates and idx >= max_variates:
                break
            
            # 获取时间序列数据
            target = sample.get('target', None)
            if target is None:
                target = sample.get('sequence', None)
                if target is None:
                    continue
            
            # 转换为numpy数组
            if isinstance(target, list):
                ts_data = np.array(target, dtype=np.float32)
            elif isinstance(target, np.ndarray):
                ts_data = target.astype(np.float32)
            else:
                try:
                    ts_data = np.array(target, dtype=np.float32)
                except:
                    continue
            
            # 处理多变量序列：每个变量作为一个独立的序列
            if ts_data.ndim > 1:
                # 多变量序列：按变量维度分割
                for var_idx in range(ts_data.shape[-1]):
                    var_series = ts_data[:, var_idx] if ts_data.ndim == 2 else ts_data[var_idx]
                    if len(var_series) >= self.context_length:
                        series_list.append(var_series)
                        series_ids.append(f"sample_{idx}_var_{var_idx}")
            else:
                # 单变量序列
                if len(ts_data) >= self.context_length:
                    series_list.append(ts_data)
                    series_ids.append(f"sample_{idx}")
        
        print(f"Extracted {len(series_list)} variate series")
        
        # 合并到池中
        self.merge_to_pool(series_list, series_ids)
        
        print(f"Sequence pool size: {len(self.sequence_pool)}")
        total_length = sum(item['length'] for item in self.sequence_pool)
        print(f"Total sequence length: {total_length}")
    
    def process_csv_file(
        self,
        csv_path: str,
        date_col: Optional[str] = 'datetime',
        exclude_cols: Optional[List[str]] = None,
        max_variates: Optional[int] = None
    ):
        """
        处理CSV文件，将每一列作为一个独立的时间序列
        
        Args:
            csv_path: CSV文件路径
            date_col: 日期列名（如果存在，将被排除）
            exclude_cols: 要排除的列名列表
            max_variates: 最大变量数（用于限制处理数量）
        """
        import pandas as pd
        
        print(f"Loading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        
        print(f"CSV shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        
        # 确定要处理的列
        if exclude_cols is None:
            exclude_cols = []
        if date_col and date_col in df.columns:
            exclude_cols.append(date_col)
        
        # 获取数值列
        numeric_cols = []
        for col in df.columns:
            if col not in exclude_cols:
                if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    numeric_cols.append(col)
        
        print(f"Found {len(numeric_cols)} numeric columns to process")
        
        if max_variates:
            numeric_cols = numeric_cols[:max_variates]
            print(f"Limited to {len(numeric_cols)} columns (max_variates={max_variates})")
        
        series_list = []
        series_ids = []
        
        # 处理每一列
        for col_idx, col_name in enumerate(numeric_cols):
            series = df[col_name].values.astype(np.float32)
            
            # 移除 NaN 值（使用前向填充）
            if pd.isna(series).any():
                series = pd.Series(series).ffill().fillna(0.0).values.astype(np.float32)
            
            # 只处理长度足够的序列
            if len(series) >= self.context_length:
                series_list.append(series)
                series_ids.append(col_name)
            else:
                print(f"[WARNING] Column '{col_name}' is too short ({len(series)} < {self.context_length}), skipping")
        
        print(f"Extracted {len(series_list)} valid series from CSV")
        
        # 合并到池中
        self.merge_to_pool(series_list, series_ids)
        
        print(f"Sequence pool size: {len(self.sequence_pool)}")
        total_length = sum(item['length'] for item in self.sequence_pool)
        print(f"Total sequence length: {total_length}")
    
    def get_statistics(self) -> Dict:
        """获取预处理统计信息"""
        return {
            'num_series': len(self.sequence_pool),
            'total_length': sum(item['length'] for item in self.sequence_pool),
            'context_length': self.context_length,
            'normalization_stats': self.normalization_stats,
            'train_ratio': self.train_ratio
        }
    
    def save(self, output_dir: str):
        """保存预处理结果和统计信息"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存统计信息
        stats = self.get_statistics()
        with open(os.path.join(output_dir, "s3_stats.pkl"), "wb") as f:
            pickle.dump(stats, f)
        
        # 保存序列池（可选，如果内存允许）
        # 注意：如果序列池很大，可能不适合全部保存
        print(f"[OK] S3 preprocessing statistics saved to {output_dir}/s3_stats.pkl")


class S3Dataset(Dataset):
    """
    S3格式数据集
    用于预训练的单序列序列数据集
    """
    
    def __init__(
        self,
        sequences: List[np.ndarray],
        lookback: int,
        pred_len: int
    ):
        """
        Args:
            sequences: S3格式的序列列表（每个序列长度应该 >= lookback + pred_len）
            lookback: 历史数据长度
            pred_len: 预测长度
        """
        self.lookback = lookback
        self.pred_len = pred_len
        self.context_length = lookback + pred_len
        
        # 过滤长度不足的序列
        original_count = len(sequences)
        self.sequences = [
            seq for seq in sequences 
            if len(seq) >= self.context_length
        ]
        
        filtered_count = original_count - len(self.sequences)
        if filtered_count > 0:
            print(f"[WARNING] Filtered out {filtered_count} sequences (length < {self.context_length})")
        
        print(f"S3 Dataset: {len(self.sequences)} sequences")
        print(f"Context length: {self.context_length} (lookback={lookback}, pred_len={pred_len})")
        
        if len(self.sequences) == 0:
            raise ValueError(
                f"No valid sequences found! All {original_count} sequences are shorter than "
                f"required length {self.context_length} (lookback={lookback} + pred_len={pred_len}). "
                f"Please increase context_length parameter in prepare_s3_for_pretraining."
            )
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # 确保序列长度正确（sequences应该已经是固定长度）
        # 如果序列长度大于context_length，截取前context_length部分
        # 注意：不使用随机采样，保持数据访问的确定性
        if len(seq) > self.context_length:
            seq = seq[:self.context_length]
        elif len(seq) < self.context_length:
            # 如果序列太短，用零填充（不应该发生，因为在__init__中已过滤）
            padding = np.zeros(self.context_length - len(seq), dtype=np.float32)
            seq = np.concatenate([seq, padding])
        
        # 前 lookback 个点作为输入
        history = seq[:self.lookback]
        # 后 pred_len 个点作为目标（用于自回归训练）
        target = seq[self.lookback:self.lookback + self.pred_len]
        
        return torch.tensor(history, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def prepare_s3_for_pretraining(
    hf_dataset,
    context_length: int = 512,
    lookback: int = 512,
    pred_len: int = 96,
    train_ratio: float = 0.9,
    val_ratio: float = 0.1,
    max_variates: Optional[int] = None,
    num_train_samples: Optional[int] = None,
    num_val_samples: Optional[int] = None,
    output_dir: str = "data_cache/s3",
    random_seed: int = 42,
    use_cache: bool = True
) -> Tuple[S3Dataset, S3Dataset, Dict]:
    """
    准备S3格式数据用于预训练
    
    Args:
        hf_dataset: HuggingFace数据集对象
        context_length: 固定上下文长度（用于采样）
        lookback: 历史数据长度
        pred_len: 预测长度
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        max_variates: 最大变量数
        num_train_samples: 训练样本数量（如果为None，尽可能多采样）
        num_val_samples: 验证样本数量
        output_dir: 输出目录
        random_seed: 随机种子
    
    Returns:
        train_dataset, val_dataset, data_config
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 验证参数：context_length必须 >= lookback + pred_len
    required_length = lookback + pred_len
    if context_length < required_length:
        print(f"[WARNING] context_length ({context_length}) < lookback + pred_len ({required_length})")
        print(f"[WARNING] Adjusting context_length to {required_length}")
        context_length = required_length
    
    # 检查缓存文件是否存在
    cache_config_file = os.path.join(output_dir, "data_config.pkl")
    cache_train_file = os.path.join(output_dir, "train_sequences.pkl")
    cache_val_file = os.path.join(output_dir, "val_sequences.pkl")
    
    if use_cache and os.path.exists(cache_config_file) and \
       os.path.exists(cache_train_file) and os.path.exists(cache_val_file):
        print("\n[INFO] Found cached data, loading from local files...")
        print(f"  Cache directory: {output_dir}")
        
        try:
            # 加载配置
            with open(cache_config_file, 'rb') as f:
                cached_config = pickle.load(f)
            
            # 检查配置是否匹配
            if (cached_config.get('lookback') == lookback and 
                cached_config.get('pred_len') == pred_len and 
                cached_config.get('context_length') == context_length):
                
                # 加载序列数据
                with open(cache_train_file, 'rb') as f:
                    train_sequences = pickle.load(f)
                with open(cache_val_file, 'rb') as f:
                    val_sequences = pickle.load(f)
                
                print(f"  Loaded {len(train_sequences)} train sequences")
                print(f"  Loaded {len(val_sequences)} val sequences")
                
                # 创建数据集
                train_dataset = S3Dataset(train_sequences, lookback, pred_len)
                val_dataset = S3Dataset(val_sequences, lookback, pred_len)
                
                print("[OK] Successfully loaded cached data!")
                return train_dataset, val_dataset, cached_config
            else:
                print("[WARNING] Cached config doesn't match, will reprocess data...")
        except Exception as e:
            print(f"[WARNING] Failed to load cache: {str(e)}, will reprocess data...")
    
    # 如果没有hf_dataset且没有有效缓存，报错
    if hf_dataset is None:
        raise ValueError(
            "No HuggingFace dataset provided and no valid cache found. "
            "Please provide hf_dataset or ensure cache exists."
        )
    
    print("\n[INFO] Processing data from HuggingFace dataset...")
    
    # 创建预处理器
    preprocessor = S3Preprocessor(
        context_length=context_length,
        train_ratio=train_ratio,
        normalize=True,
        random_seed=random_seed
    )
    
    # 处理数据集
    preprocessor.process_hf_dataset(hf_dataset, max_variates=max_variates)
    
    # 从池中采样训练和验证序列
    # 注意：这里我们使用不同的采样策略来区分训练和验证
    # 实际实现中，可以根据序列的split_point来区分
    
    # 采样训练序列
    train_sequences = preprocessor.sample_sequences(
        num_samples=num_train_samples,
        stride=context_length // 2  # 使用重叠采样
    )
    
    # 采样验证序列（使用不同的随机种子）
    np.random.seed(random_seed + 1)
    val_sequences = preprocessor.sample_sequences(
        num_samples=num_val_samples,
        stride=context_length  # 验证集使用非重叠采样
    )
    
    print(f"\nSampled sequences:")
    print(f"  Train: {len(train_sequences)} sequences")
    print(f"  Val: {len(val_sequences)} sequences")
    
    # 创建数据集
    train_dataset = S3Dataset(train_sequences, lookback, pred_len)
    val_dataset = S3Dataset(val_sequences, lookback, pred_len)
    
    # 保存统计信息
    preprocessor.save(output_dir)
    
    # 数据配置
    data_config = {
        'lookback': lookback,
        'pred_len': pred_len,
        'context_length': context_length,
        'train_ratio': train_ratio,
        'n_train': len(train_dataset),
        'n_val': len(val_dataset),
        'format': 'S3',
        'normalization': True
    }
    
    # 保存配置
    with open(os.path.join(output_dir, "data_config.pkl"), "wb") as f:
        pickle.dump(data_config, f)
    print(f"[OK] Data config saved to {output_dir}/data_config.pkl")
    
    # 保存序列数据到缓存
    if use_cache:
        print("\n[INFO] Saving processed sequences to cache...")
        with open(os.path.join(output_dir, "train_sequences.pkl"), "wb") as f:
            pickle.dump(train_sequences, f)
        print(f"  Saved train sequences to {output_dir}/train_sequences.pkl")
        
        with open(os.path.join(output_dir, "val_sequences.pkl"), "wb") as f:
            pickle.dump(val_sequences, f)
        print(f"  Saved val sequences to {output_dir}/val_sequences.pkl")
        print("[OK] Cache saved successfully!")
    
    return train_dataset, val_dataset, data_config


def prepare_csv_for_pretraining(
    csv_path: str,
    context_length: int = 512,
    lookback: int = 512,
    pred_len: int = 96,
    train_ratio: float = 0.9,
    val_ratio: float = 0.1,
    date_col: Optional[str] = 'datetime',
    exclude_cols: Optional[List[str]] = None,
    max_variates: Optional[int] = None,
    num_train_samples: Optional[int] = None,
    num_val_samples: Optional[int] = None,
    output_dir: str = "data_cache/s3",
    random_seed: int = 42,
    use_cache: bool = True
) -> Tuple[S3Dataset, S3Dataset, Dict]:
    """
    准备CSV文件数据用于S3格式预训练
    
    Args:
        csv_path: CSV文件路径
        context_length: 固定上下文长度（用于采样）
        lookback: 历史数据长度
        pred_len: 预测长度
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        date_col: 日期列名（如果存在，将被排除）
        exclude_cols: 要排除的列名列表
        max_variates: 最大变量数（用于限制处理数量）
        num_train_samples: 训练样本数量（如果为None，尽可能多采样）
        num_val_samples: 验证样本数量
        output_dir: 输出目录
        random_seed: 随机种子
        use_cache: 是否使用缓存
    
    Returns:
        train_dataset, val_dataset, data_config
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 验证参数：context_length必须 >= lookback + pred_len
    required_length = lookback + pred_len
    if context_length < required_length:
        print(f"[WARNING] context_length ({context_length}) < lookback + pred_len ({required_length})")
        print(f"[WARNING] Adjusting context_length to {required_length}")
        context_length = required_length
    
    # 检查缓存文件是否存在
    cache_config_file = os.path.join(output_dir, "data_config.pkl")
    cache_train_file = os.path.join(output_dir, "train_sequences.pkl")
    cache_val_file = os.path.join(output_dir, "val_sequences.pkl")
    
    if use_cache and os.path.exists(cache_config_file) and \
       os.path.exists(cache_train_file) and os.path.exists(cache_val_file):
        print("\n[INFO] Found cached data, loading from local files...")
        print(f"  Cache directory: {output_dir}")
        
        try:
            # 加载配置
            with open(cache_config_file, 'rb') as f:
                cached_config = pickle.load(f)
            
            # 检查配置是否匹配
            if (cached_config.get('lookback') == lookback and 
                cached_config.get('pred_len') == pred_len and 
                cached_config.get('context_length') == context_length):
                
                # 加载序列数据
                with open(cache_train_file, 'rb') as f:
                    train_sequences = pickle.load(f)
                with open(cache_val_file, 'rb') as f:
                    val_sequences = pickle.load(f)
                
                print(f"  Loaded {len(train_sequences)} train sequences")
                print(f"  Loaded {len(val_sequences)} val sequences")
                
                # 创建数据集
                train_dataset = S3Dataset(train_sequences, lookback, pred_len)
                val_dataset = S3Dataset(val_sequences, lookback, pred_len)
                
                print("[OK] Successfully loaded cached data!")
                return train_dataset, val_dataset, cached_config
            else:
                print("[WARNING] Cached config doesn't match, will reprocess data...")
        except Exception as e:
            print(f"[WARNING] Failed to load cache: {str(e)}, will reprocess data...")
    
    print("\n[INFO] Processing data from CSV file...")
    
    # 创建预处理器
    preprocessor = S3Preprocessor(
        context_length=context_length,
        train_ratio=train_ratio,
        normalize=True,
        random_seed=random_seed
    )
    
    # 处理CSV文件
    preprocessor.process_csv_file(
        csv_path=csv_path,
        date_col=date_col,
        exclude_cols=exclude_cols,
        max_variates=max_variates
    )
    
    # 从池中采样训练和验证序列
    # 采样训练序列
    train_sequences = preprocessor.sample_sequences(
        num_samples=num_train_samples,
        stride=context_length // 2  # 使用重叠采样
    )
    
    # 采样验证序列（使用不同的随机种子）
    np.random.seed(random_seed + 1)
    val_sequences = preprocessor.sample_sequences(
        num_samples=num_val_samples,
        stride=context_length  # 验证集使用非重叠采样
    )
    
    print(f"\nSampled sequences:")
    print(f"  Train: {len(train_sequences)} sequences")
    print(f"  Val: {len(val_sequences)} sequences")
    
    # 创建数据集
    train_dataset = S3Dataset(train_sequences, lookback, pred_len)
    val_dataset = S3Dataset(val_sequences, lookback, pred_len)
    
    # 保存统计信息
    preprocessor.save(output_dir)
    
    # 数据配置
    data_config = {
        'lookback': lookback,
        'pred_len': pred_len,
        'context_length': context_length,
        'train_ratio': train_ratio,
        'n_train': len(train_dataset),
        'n_val': len(val_dataset),
        'format': 'S3',
        'normalization': True,
        'csv_path': csv_path
    }
    
    # 保存配置
    with open(os.path.join(output_dir, "data_config.pkl"), "wb") as f:
        pickle.dump(data_config, f)
    print(f"[OK] Data config saved to {output_dir}/data_config.pkl")
    
    # 保存序列数据到缓存
    if use_cache:
        print("\n[INFO] Saving processed sequences to cache...")
        with open(os.path.join(output_dir, "train_sequences.pkl"), "wb") as f:
            pickle.dump(train_sequences, f)
        print(f"  Saved train sequences to {output_dir}/train_sequences.pkl")
        
        with open(os.path.join(output_dir, "val_sequences.pkl"), "wb") as f:
            pickle.dump(val_sequences, f)
        print(f"  Saved val sequences to {output_dir}/val_sequences.pkl")
        print("[OK] Cache saved successfully!")
    
    return train_dataset, val_dataset, data_config

