"""
测试脚本：验证所有导入是否正常
Test script: Verify all imports work correctly
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """测试所有关键模块的导入"""
    print("=" * 60)
    print("Testing Imports")
    print("=" * 60)
    
    errors = []
    
    # 测试基础依赖
    print("\n1. Testing basic dependencies...")
    try:
        import torch
        print(f"   [OK] torch {torch.__version__}")
    except Exception as e:
        errors.append(f"torch: {str(e)}")
        print(f"   [FAIL] torch: {str(e)}")
    
    try:
        import transformers
        print(f"   [OK] transformers {transformers.__version__}")
    except Exception as e:
        errors.append(f"transformers: {str(e)}")
        print(f"   [FAIL] transformers: {str(e)}")
    
    try:
        import datasets
        print(f"   [OK] datasets (HuggingFace) {datasets.__version__}")
    except Exception as e:
        errors.append(f"datasets: {str(e)}")
        print(f"   [FAIL] datasets: {str(e)}")
    
    try:
        import pandas
        print(f"   [OK] pandas {pandas.__version__}")
    except Exception as e:
        errors.append(f"pandas: {str(e)}")
        print(f"   [FAIL] pandas: {str(e)}")
    
    try:
        import numpy
        print(f"   [OK] numpy {numpy.__version__}")
    except Exception as e:
        errors.append(f"numpy: {str(e)}")
        print(f"   [FAIL] numpy: {str(e)}")
    
    try:
        import matplotlib
        print(f"   [OK] matplotlib {matplotlib.__version__}")
    except Exception as e:
        errors.append(f"matplotlib: {str(e)}")
        print(f"   [FAIL] matplotlib: {str(e)}")
    
    # 测试项目模块
    print("\n2. Testing project modules...")
    try:
        from models import TimerConfig, TimerForPrediction
        print("   [OK] models (TimerConfig, TimerForPrediction)")
    except Exception as e:
        errors.append(f"models: {str(e)}")
        print(f"   [FAIL] models: {str(e)}")
    
    try:
        from data_processing import (
            TimeSeriesDataset,
            create_dataloaders,
            download_utsd_dataset,
            UTSDDataset,
            prepare_utsd_for_training,
            S3Preprocessor,
            S3Dataset,
            prepare_s3_for_pretraining,
            StandardTimeSeriesDataset,
            load_standard_dataset,
            prepare_multiple_datasets,
            download_dataset
        )
        print("   [OK] data_processing module (all data processing modules)")
    except Exception as e:
        errors.append(f"data_processing module: {str(e)}")
        print(f"   [FAIL] data_processing module: {str(e)}")
        import traceback
        traceback.print_exc()
    
    try:
        from training import Trainer, FineTuneTrainer
        print("   [OK] training (Trainer, FineTuneTrainer)")
    except Exception as e:
        errors.append(f"training: {str(e)}")
        print(f"   [FAIL] training: {str(e)}")
    
    try:
        from utils import count_parameters, load_pretrained_model
        print("   [OK] utils (count_parameters, load_pretrained_model)")
    except Exception as e:
        errors.append(f"utils: {str(e)}")
        print(f"   [FAIL] utils: {str(e)}")
    
    # 测试CUDA
    print("\n3. Testing CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   [OK] CUDA is available")
            print(f"   [OK] CUDA version: {torch.version.cuda}")
            print(f"   [OK] Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   [OK] GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("   [WARN] CUDA not available, will use CPU")
    except Exception as e:
        print(f"   [FAIL] CUDA check failed: {str(e)}")
    
    # 总结
    print("\n" + "=" * 60)
    if errors:
        print(f"[FAIL] Tests failed! Found {len(errors)} errors:")
        for error in errors:
            print(f"   - {error}")
        print("=" * 60)
        return False
    else:
        print("[SUCCESS] All tests passed!")
        print("=" * 60)
        return True


def test_model_creation():
    """测试模型创建"""
    print("\n" + "=" * 60)
    print("Testing Model Creation")
    print("=" * 60)
    
    try:
        from models import TimerConfig, TimerForPrediction
        
        # 创建一个小模型用于测试
        config = TimerConfig(
            input_token_len=96,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            output_token_lens=[96]
        )
        
        model = TimerForPrediction(config)
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n[OK] Model created successfully!")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return True
    except Exception as e:
        print(f"\n[FAIL] Model creation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n")
    print("=" * 60)
    print("           TIMER Project Test Suite")
    print("=" * 60)
    
    # 运行测试
    import_success = test_imports()
    model_success = test_model_creation()
    
    # 最终结果
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    
    if import_success and model_success:
        print("[SUCCESS] All tests passed! Project is configured correctly.")
        print("\nNext steps:")
        print("  1. Run pretraining: bash scripts/run_pretrain_and_eval.sh")
        print("  2. Or run training: python scripts/train.py --help")
        sys.exit(0)
    else:
        print("[FAIL] Some tests failed! Please check the error messages.")
        print("\nSuggestions:")
        print("  1. Check dependencies: pip install -r requirements.txt")
        print("  2. Check Python version: python --version (requires >= 3.7)")
        sys.exit(1)
