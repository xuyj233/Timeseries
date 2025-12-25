"""
S3 (Single-Series Sequence) format preprocessing
Implements S3 format described in the paper: normalization, merging, sampling
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
    S3 format preprocessor
    Converts heterogeneous time series to unified single-series sequence format
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
            context_length: Fixed context length (window size)
            train_ratio: Training set ratio (for calculating normalization statistics)
            normalize: Whether to normalize
            random_seed: Random seed
        """
        self.context_length = context_length
        self.train_ratio = train_ratio
        self.normalize = normalize
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Store normalization statistics
        self.normalization_stats = {}  # {series_id: {'mean': float, 'std': float}}
        
        # Store processed sequence pool
        self.sequence_pool = []  # List of normalized single-variate series
    
    def process_variate(
        self,
        series: np.ndarray,
        series_id: str,
        split_point: Optional[int] = None
    ) -> np.ndarray:
        """
        Process single variate series: normalize and return
        
        Args:
            series: Time series array (1D)
            series_id: Series identifier
            split_point: Train/validation split point (if None, auto-calculate)
        
        Returns:
            normalized_series: Normalized series
        """
        series = series.astype(np.float32)
        
        # Calculate split point (9:1)
        if split_point is None:
            split_point = int(len(series) * self.train_ratio)
        
        # Normalize using training set statistics
        train_series = series[:split_point]
        
        if self.normalize:
            mean = np.mean(train_series)
            std = np.std(train_series)
            
            # Avoid division by zero
            if std < 1e-8:
                std = 1.0
            
            # Save statistics
            self.normalization_stats[series_id] = {
                'mean': float(mean),
                'std': float(std),
                'split_point': split_point
            }
            
            # Normalize entire series (using training set statistics)
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
        Merge multiple normalized sequences into sequence pool
        
        Args:
            series_list: List of normalized sequences
            series_ids: List of series identifiers (optional)
        """
        if series_ids is None:
            series_ids = [f"series_{i}" for i in range(len(series_list))]
        
        for series, series_id in zip(series_list, series_ids):
            # Process each series
            normalized_series = self.process_variate(series, series_id)
            
            # Add to pool
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
        Uniformly sample fixed-length sequences from sequence pool
        
        Args:
            num_samples: Number of samples (if None, sample as many as possible)
            stride: Sampling stride (if None, use context_length)
        
        Returns:
            sampled_sequences: List of sampled sequences
        """
        if stride is None:
            stride = self.context_length
        
        sampled_sequences = []
        
        # Sample windows from each sequence
        for pool_item in self.sequence_pool:
            series = pool_item['data']
            series_length = pool_item['length']
            
            if series_length < self.context_length:
                continue
            
            # Uniformly sample windows
            for start_idx in range(0, series_length - self.context_length + 1, stride):
                window = series[start_idx:start_idx + self.context_length]
                sampled_sequences.append(window)
        
        # If sampling number is specified, random sample
        if num_samples is not None and len(sampled_sequences) > num_samples:
            indices = np.random.choice(
                len(sampled_sequences),
                size=num_samples,
                replace=False
            )
            sampled_sequences = [sampled_sequences[i] for i in indices]
        
        # Shuffle order
        np.random.shuffle(sampled_sequences)
        
        return sampled_sequences
    
    def process_hf_dataset(
        self,
        hf_dataset,
        max_variates: Optional[int] = None
    ):
        """
        Process HuggingFace format UTSD dataset
        
        Args:
            hf_dataset: HuggingFace dataset object
            max_variates: Maximum number of variates (to limit processing)
        """
        series_list = []
        series_ids = []
        
        print(f"Processing {len(hf_dataset)} samples from dataset...")
        
        for idx, sample in enumerate(hf_dataset):
            if max_variates and idx >= max_variates:
                break
            
            # Get time series data
            target = sample.get('target', None)
            if target is None:
                target = sample.get('sequence', None)
                if target is None:
                    continue
            
            # Convert to numpy array
            if isinstance(target, list):
                ts_data = np.array(target, dtype=np.float32)
            elif isinstance(target, np.ndarray):
                ts_data = target.astype(np.float32)
            else:
                try:
                    ts_data = np.array(target, dtype=np.float32)
                except:
                    continue
            
            # Process multivariate sequences: each variable as an independent sequence
            if ts_data.ndim > 1:
                # Multivariate sequence: split by variable dimension
                for var_idx in range(ts_data.shape[-1]):
                    var_series = ts_data[:, var_idx] if ts_data.ndim == 2 else ts_data[var_idx]
                    if len(var_series) >= self.context_length:
                        series_list.append(var_series)
                        series_ids.append(f"sample_{idx}_var_{var_idx}")
            else:
                # Univariate sequence
                if len(ts_data) >= self.context_length:
                    series_list.append(ts_data)
                    series_ids.append(f"sample_{idx}")
        
        print(f"Extracted {len(series_list)} variate series")
        
        # Merge to pool
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
        Process CSV file, treat each column as an independent time series
        
        Args:
            csv_path: CSV file path
            date_col: Date column name (if exists, will be excluded)
            exclude_cols: List of column names to exclude
            max_variates: Maximum number of variates (to limit processing)
        """
        import pandas as pd
        
        print(f"Loading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        
        print(f"CSV shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        
        # Determine columns to process
        if exclude_cols is None:
            exclude_cols = []
        if date_col and date_col in df.columns:
            exclude_cols.append(date_col)
        
        # Get numeric columns
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
        
        # Process each column
        for col_idx, col_name in enumerate(numeric_cols):
            series = df[col_name].values.astype(np.float32)
            
            # Remove NaN values (use forward fill)
            if pd.isna(series).any():
                series = pd.Series(series).ffill().fillna(0.0).values.astype(np.float32)
            
            # Only process sequences with sufficient length
            if len(series) >= self.context_length:
                series_list.append(series)
                series_ids.append(col_name)
            else:
                print(f"[WARNING] Column '{col_name}' is too short ({len(series)} < {self.context_length}), skipping")
        
        print(f"Extracted {len(series_list)} valid series from CSV")
        
        # Merge to pool
        self.merge_to_pool(series_list, series_ids)
        
        print(f"Sequence pool size: {len(self.sequence_pool)}")
        total_length = sum(item['length'] for item in self.sequence_pool)
        print(f"Total sequence length: {total_length}")
    
    def get_statistics(self) -> Dict:
        """Get preprocessing statistics"""
        return {
            'num_series': len(self.sequence_pool),
            'total_length': sum(item['length'] for item in self.sequence_pool),
            'context_length': self.context_length,
            'normalization_stats': self.normalization_stats,
            'train_ratio': self.train_ratio
        }
    
    def save(self, output_dir: str):
        """Save preprocessing results and statistics"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save statistics
        stats = self.get_statistics()
        with open(os.path.join(output_dir, "s3_stats.pkl"), "wb") as f:
            pickle.dump(stats, f)
        
        # Save sequence pool (optional, if memory allows)
        # Note: If sequence pool is very large, may not be suitable to save all
        print(f"[OK] S3 preprocessing statistics saved to {output_dir}/s3_stats.pkl")


class S3Dataset(Dataset):
    """
    S3 format dataset
    Single-series sequence dataset for pretraining
    """
    
    def __init__(
        self,
        sequences: List[np.ndarray],
        lookback: int,
        pred_len: int
    ):
        """
        Args:
            sequences: List of S3 format sequences (each sequence length should >= lookback + pred_len)
            lookback: Historical data length
            pred_len: Prediction length
        """
        self.lookback = lookback
        self.pred_len = pred_len
        self.context_length = lookback + pred_len
        
        # Filter sequences with insufficient length
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
        
        # Ensure sequence length is correct (sequences should already be fixed length)
        # If sequence length is greater than context_length, take first context_length part
        # Note: Don't use random sampling, maintain deterministic data access
        if len(seq) > self.context_length:
            seq = seq[:self.context_length]
        elif len(seq) < self.context_length:
            # If sequence is too short, pad with zeros (shouldn't happen, already filtered in __init__)
            padding = np.zeros(self.context_length - len(seq), dtype=np.float32)
            seq = np.concatenate([seq, padding])
        
        # First lookback points as input
        history = seq[:self.lookback]
        # Last pred_len points as target (for autoregressive training)
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
    output_dir: str = "data/s3",
    random_seed: int = 42,
    use_cache: bool = True
) -> Tuple[S3Dataset, S3Dataset, Dict]:
    """
    Prepare S3 format data for pretraining
    
    Args:
        hf_dataset: HuggingFace dataset object
        context_length: Fixed context length (for sampling)
        lookback: Historical data length
        pred_len: Prediction length
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        max_variates: Maximum number of variates
        num_train_samples: Number of training samples (if None, sample as many as possible)
        num_val_samples: Number of validation samples
        output_dir: Output directory
        random_seed: Random seed
    
    Returns:
        train_dataset, val_dataset, data_config
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate parameters: context_length must >= lookback + pred_len
    required_length = lookback + pred_len
    if context_length < required_length:
        print(f"[WARNING] context_length ({context_length}) < lookback + pred_len ({required_length})")
        print(f"[WARNING] Adjusting context_length to {required_length}")
        context_length = required_length
    
    # Check if cache files exist
    cache_config_file = os.path.join(output_dir, "data_config.pkl")
    cache_train_file = os.path.join(output_dir, "train_sequences.pkl")
    cache_val_file = os.path.join(output_dir, "val_sequences.pkl")
    
    if use_cache and os.path.exists(cache_config_file) and \
       os.path.exists(cache_train_file) and os.path.exists(cache_val_file):
        print("\n[INFO] Found cached data, loading from local files...")
        print(f"  Cache directory: {output_dir}")
        
        try:
            # Load configuration
            with open(cache_config_file, 'rb') as f:
                cached_config = pickle.load(f)
            
            # Check if configuration matches
            if (cached_config.get('lookback') == lookback and 
                cached_config.get('pred_len') == pred_len and 
                cached_config.get('context_length') == context_length):
                
                # Load sequence data
                with open(cache_train_file, 'rb') as f:
                    train_sequences = pickle.load(f)
                with open(cache_val_file, 'rb') as f:
                    val_sequences = pickle.load(f)
                
                print(f"  Loaded {len(train_sequences)} train sequences")
                print(f"  Loaded {len(val_sequences)} val sequences")
                
                # Create datasets
                train_dataset = S3Dataset(train_sequences, lookback, pred_len)
                val_dataset = S3Dataset(val_sequences, lookback, pred_len)
                
                print("[OK] Successfully loaded cached data!")
                return train_dataset, val_dataset, cached_config
            else:
                print("[WARNING] Cached config doesn't match, will reprocess data...")
        except Exception as e:
            print(f"[WARNING] Failed to load cache: {str(e)}, will reprocess data...")
    
    # If no hf_dataset and no valid cache, raise error
    if hf_dataset is None:
        raise ValueError(
            "No HuggingFace dataset provided and no valid cache found. "
            "Please provide hf_dataset or ensure cache exists."
        )
    
    print("\n[INFO] Processing data from HuggingFace dataset...")
    
    # Create preprocessor
    preprocessor = S3Preprocessor(
        context_length=context_length,
        train_ratio=train_ratio,
        normalize=True,
        random_seed=random_seed
    )
    
    # Process dataset
    preprocessor.process_hf_dataset(hf_dataset, max_variates=max_variates)
    
    # Sample training and validation sequences from pool
    # Note: Here we use different sampling strategies to distinguish training and validation
    # In actual implementation, can distinguish based on sequence split_point
    
    # Sample training sequences
    train_sequences = preprocessor.sample_sequences(
        num_samples=num_train_samples,
        stride=context_length // 2  # Use overlapping sampling
    )
    
    # Sample validation sequences (use different random seed)
    np.random.seed(random_seed + 1)
    val_sequences = preprocessor.sample_sequences(
        num_samples=num_val_samples,
        stride=context_length  # Validation set uses non-overlapping sampling
    )
    
    print(f"\nSampled sequences:")
    print(f"  Train: {len(train_sequences)} sequences")
    print(f"  Val: {len(val_sequences)} sequences")
    
    # Create datasets
    train_dataset = S3Dataset(train_sequences, lookback, pred_len)
    val_dataset = S3Dataset(val_sequences, lookback, pred_len)
    
    # Save statistics
    preprocessor.save(output_dir)
    
    # Data configuration
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
    
    # Save configuration
    with open(os.path.join(output_dir, "data_config.pkl"), "wb") as f:
        pickle.dump(data_config, f)
    print(f"[OK] Data config saved to {output_dir}/data_config.pkl")
    
    # Save sequence data to cache
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
    output_dir: str = "data/s3",
    random_seed: int = 42,
    use_cache: bool = True
) -> Tuple[S3Dataset, S3Dataset, Dict]:
    """
    Prepare CSV file data for S3 format pretraining
    
    Args:
        csv_path: CSV file path
        context_length: Fixed context length (for sampling)
        lookback: Historical data length
        pred_len: Prediction length
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        date_col: Date column name (if exists, will be excluded)
        exclude_cols: List of column names to exclude
        max_variates: Maximum number of variates (to limit processing)
        num_train_samples: Number of training samples (if None, sample as many as possible)
        num_val_samples: Number of validation samples
        output_dir: Output directory
        random_seed: Random seed
        use_cache: Whether to use cache
    
    Returns:
        train_dataset, val_dataset, data_config
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate parameters: context_length must >= lookback + pred_len
    required_length = lookback + pred_len
    if context_length < required_length:
        print(f"[WARNING] context_length ({context_length}) < lookback + pred_len ({required_length})")
        print(f"[WARNING] Adjusting context_length to {required_length}")
        context_length = required_length
    
    # Check if cache files exist
    cache_config_file = os.path.join(output_dir, "data_config.pkl")
    cache_train_file = os.path.join(output_dir, "train_sequences.pkl")
    cache_val_file = os.path.join(output_dir, "val_sequences.pkl")
    
    if use_cache and os.path.exists(cache_config_file) and \
       os.path.exists(cache_train_file) and os.path.exists(cache_val_file):
        print("\n[INFO] Found cached data, loading from local files...")
        print(f"  Cache directory: {output_dir}")
        
        try:
            # Load configuration
            with open(cache_config_file, 'rb') as f:
                cached_config = pickle.load(f)
            
            # Check if configuration matches
            if (cached_config.get('lookback') == lookback and 
                cached_config.get('pred_len') == pred_len and 
                cached_config.get('context_length') == context_length):
                
                # Load sequence data
                with open(cache_train_file, 'rb') as f:
                    train_sequences = pickle.load(f)
                with open(cache_val_file, 'rb') as f:
                    val_sequences = pickle.load(f)
                
                print(f"  Loaded {len(train_sequences)} train sequences")
                print(f"  Loaded {len(val_sequences)} val sequences")
                
                # Create datasets
                train_dataset = S3Dataset(train_sequences, lookback, pred_len)
                val_dataset = S3Dataset(val_sequences, lookback, pred_len)
                
                print("[OK] Successfully loaded cached data!")
                return train_dataset, val_dataset, cached_config
            else:
                print("[WARNING] Cached config doesn't match, will reprocess data...")
        except Exception as e:
            print(f"[WARNING] Failed to load cache: {str(e)}, will reprocess data...")
    
    print("\n[INFO] Processing data from CSV file...")
    
    # Create preprocessor
    preprocessor = S3Preprocessor(
        context_length=context_length,
        train_ratio=train_ratio,
        normalize=True,
        random_seed=random_seed
    )
    
    # Process CSV file
    preprocessor.process_csv_file(
        csv_path=csv_path,
        date_col=date_col,
        exclude_cols=exclude_cols,
        max_variates=max_variates
    )
    
    # Sample training and validation sequences from pool
    # Sample training sequences
    train_sequences = preprocessor.sample_sequences(
        num_samples=num_train_samples,
        stride=context_length // 2  # Use overlapping sampling
    )
    
    # Sample validation sequences (use different random seed)
    np.random.seed(random_seed + 1)
    val_sequences = preprocessor.sample_sequences(
        num_samples=num_val_samples,
        stride=context_length  # Validation set uses non-overlapping sampling
    )
    
    print(f"\nSampled sequences:")
    print(f"  Train: {len(train_sequences)} sequences")
    print(f"  Val: {len(val_sequences)} sequences")
    
    # Create datasets
    train_dataset = S3Dataset(train_sequences, lookback, pred_len)
    val_dataset = S3Dataset(val_sequences, lookback, pred_len)
    
    # Save statistics
    preprocessor.save(output_dir)
    
    # Data configuration
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
    
    # Save configuration
    with open(os.path.join(output_dir, "data_config.pkl"), "wb") as f:
        pickle.dump(data_config, f)
    print(f"[OK] Data config saved to {output_dir}/data_config.pkl")
    
    # Save sequence data to cache
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

