"""
UTSD dataset support
Download and load UTSD dataset from HuggingFace
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
    Download UTSD dataset
    
    Args:
        dataset_name: Dataset name
        subset: Subset name (e.g., 'UTSD-1G', 'UTSD-2G', 'UTSD-4G', 'UTSD-12G')
        cache_dir: Cache directory
        use_mirror: Whether to use mirror
    
    Returns:
        dataset: HuggingFace dataset object
    """
    if use_mirror:
        # Set mirror endpoint
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
        # If failed, try to use mirror directly
        if use_mirror:
            try:
                # Use mirror URL directly
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
    UTSD dataset adapter
    Convert HuggingFace format UTSD dataset to training format
    """
    
    def __init__(self, hf_dataset, lookback=512, pred_len=96, max_samples=None):
        """
        Args:
            hf_dataset: HuggingFace dataset object
            lookback: Historical data length
            pred_len: Prediction length
            max_samples: Maximum number of samples (None means use all)
        """
        self.lookback = lookback
        self.pred_len = pred_len
        self.sequences = []
        
        print(f"Processing UTSD dataset...")
        print(f"Total samples: {len(hf_dataset)}")
        
        # Process data
        for idx, sample in enumerate(hf_dataset):
            if max_samples and idx >= max_samples:
                break
            
            # Get time series data
            # UTSD dataset field name is 'target'
            target = sample.get('target', None)
            if target is None:
                # Try other possible field names
                target = sample.get('sequence', None)
                if target is None:
                    continue
            
            # Convert to numpy array
            if isinstance(target, list):
                ts_data = np.array(target, dtype=np.float32)
            elif isinstance(target, np.ndarray):
                ts_data = target.astype(np.float32)
            else:
                # Try direct conversion
                try:
                    ts_data = np.array(target, dtype=np.float32)
                except:
                    continue
            
            # Check data length
            min_length = lookback + pred_len
            if len(ts_data) < min_length:
                continue
            
            # Create sliding window samples
            # For UTSD, each sample may be a complete time series
            # We use sliding window to create multiple training samples
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
        # First lookback points as input
        history = seq[:self.lookback]
        # Last pred_len points as target
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
        output_dir="data/utsd"
):
    """
    Prepare UTSD dataset for training
    
    Args:
        dataset_name: Dataset name
        subset: Subset name
        lookback: Historical data length
        pred_len: Prediction length
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        max_samples: Maximum number of samples
        cache_dir: Cache directory
        use_mirror: Whether to use mirror
        output_dir: Output directory
    
    Returns:
        train_dataset, val_dataset, test_dataset, data_config
    """
    import pickle
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Download dataset
    dataset = download_utsd_dataset(
        dataset_name=dataset_name,
        subset=subset,
        cache_dir=cache_dir,
        use_mirror=use_mirror
    )
    
    # Get training set
    train_hf = dataset['train']
    
    # Create UTSD dataset
    utsd_dataset = UTSDDataset(
        train_hf,
        lookback=lookback,
        pred_len=pred_len,
        max_samples=max_samples
    )
    
    # Split dataset
    n = len(utsd_dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n_train + n_val))
    test_indices = list(range(n_train + n_val, n))
    
    # Create subsets
    from torch.utils.data import Subset
    train_dataset = Subset(utsd_dataset, train_indices)
    val_dataset = Subset(utsd_dataset, val_indices)
    test_dataset = Subset(utsd_dataset, test_indices)
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    # Save configuration
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

