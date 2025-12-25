"""
Test data processing modules
"""
import sys
from pathlib import Path
import numpy as np
import torch
import pytest
import tempfile
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processing import TimeSeriesDataset, S3Preprocessor, S3Dataset


class TestTimeSeriesDataset:
    """Test TimeSeriesDataset class"""
    
    def test_dataset_creation(self):
        """Test dataset creation"""
        sequences = [
            np.random.randn(200) for _ in range(10)
        ]
        lookback = 100
        pred_len = 50
        
        dataset = TimeSeriesDataset(sequences, lookback, pred_len)
        assert len(dataset) == 10
    
    def test_dataset_getitem(self):
        """Test dataset __getitem__"""
        sequences = [
            np.random.randn(200) for _ in range(5)
        ]
        lookback = 100
        pred_len = 50
        
        dataset = TimeSeriesDataset(sequences, lookback, pred_len)
        
        history, target = dataset[0]
        assert isinstance(history, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert history.shape[0] == lookback
        assert target.shape[0] == pred_len


class TestS3Preprocessor:
    """Test S3Preprocessor class"""
    
    def test_preprocessor_creation(self):
        """Test preprocessor creation"""
        preprocessor = S3Preprocessor(
            context_length=512,
            train_ratio=0.9,
            normalize=True,
            random_seed=42
        )
        assert preprocessor is not None
    
    def test_process_single_series(self):
        """Test processing single series"""
        preprocessor = S3Preprocessor(
            context_length=512,
            train_ratio=0.9,
            normalize=True,
            random_seed=42
        )
        
        series = np.random.randn(1000).astype(np.float32)
        normalized_series = preprocessor.process_variate(series, "test_series")
        
        assert normalized_series is not None
        assert len(normalized_series) == len(series)
        assert "test_series" in preprocessor.normalization_stats
    
    def test_merge_sequences(self):
        """Test merging sequences"""
        preprocessor = S3Preprocessor(
            context_length=512,
            train_ratio=0.9,
            normalize=True,
            random_seed=42
        )
        
        # First process and normalize series
        series_list = []
        for i in range(5):
            series = np.random.randn(100).astype(np.float32)
            normalized = preprocessor.process_variate(series, f"series_{i}")
            series_list.append(normalized)
        
        preprocessor.merge_to_pool(series_list)
        
        assert len(preprocessor.sequence_pool) > 0
    
    def test_sample_sequences(self):
        """Test sampling sequences"""
        preprocessor = S3Preprocessor(
            context_length=512,
            train_ratio=0.9,
            normalize=True,
            random_seed=42
        )
        
        # Add some sequences to pool (first normalize them)
        series_list = []
        for i in range(5):
            series = np.random.randn(1000).astype(np.float32)
            normalized = preprocessor.process_variate(series, f"series_{i}")
            series_list.append(normalized)
        
        preprocessor.merge_to_pool(series_list)
        
        # Sample sequences
        sampled = preprocessor.sample_sequences(num_samples=10, stride=256)
        assert len(sampled) > 0
        assert all(len(seq) == 512 for seq in sampled)


class TestS3Dataset:
    """Test S3Dataset class"""
    
    def test_s3_dataset_creation(self):
        """Test S3Dataset creation"""
        sequences = [
            np.random.randn(600) for _ in range(10)  # Length >= lookback + pred_len
        ]
        lookback = 400
        pred_len = 100
        
        dataset = S3Dataset(sequences, lookback, pred_len)
        assert len(dataset) > 0
    
    def test_s3_dataset_getitem(self):
        """Test S3Dataset __getitem__"""
        sequences = [
            np.random.randn(600) for _ in range(5)
        ]
        lookback = 400
        pred_len = 100
        
        dataset = S3Dataset(sequences, lookback, pred_len)
        
        history, target = dataset[0]
        assert isinstance(history, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert history.shape[0] == lookback
        assert target.shape[0] == pred_len


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

