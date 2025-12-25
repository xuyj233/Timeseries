"""
Script to download UTSD dataset
Supports downloading from mirror
"""
import os
import argparse

# Set mirror
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import download_utsd_dataset


def main():
    parser = argparse.ArgumentParser(description="Download UTSD dataset")
    parser.add_argument("--subset", type=str, default=None,
                       choices=[None, 'UTSD-1G', 'UTSD-2G', 'UTSD-4G', 'UTSD-12G'],
                       help="UTSD subset to download")
    parser.add_argument("--cache-dir", type=str, default=None,
                       help="Cache directory for dataset")
    parser.add_argument("--use-mirror", action="store_true", default=True,
                       help="Use mirror site (hf-mirror.com)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("UTSD Dataset Downloader")
    print("="*60)
    print(f"Subset: {args.subset if args.subset else 'default (all)'}")
    print(f"Cache dir: {args.cache_dir if args.cache_dir else 'default'}")
    print(f"Use mirror: {args.use_mirror}")
    print("="*60)
    
    try:
        dataset = download_utsd_dataset(
            dataset_name="thuml/UTSD",
            subset=args.subset,
            cache_dir=args.cache_dir,
            use_mirror=args.use_mirror
        )
        
        print("\n" + "="*60)
        print("Download completed successfully!")
        print("="*60)
        print(f"Train split size: {len(dataset['train'])}")
        print(f"\nYou can now use this dataset for training:")
        print(f"  python scripts/train.py --mode pretrain --data-source utsd --utsd-subset {args.subset or 'default'}")
        print("="*60)
        
    except Exception as e:
        print(f"\nError downloading dataset: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Make sure HF_ENDPOINT is set correctly")
        print("3. Try running: export HF_ENDPOINT=https://hf-mirror.com")
        raise


if __name__ == "__main__":
    main()

