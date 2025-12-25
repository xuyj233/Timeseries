"""
Unified training entry point
Supports pretraining and fine-tuning with multiple model structures and datasets
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader

# Set mirror endpoint
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import TimerConfig, TimerForPrediction
from data_processing import (
    create_dataloaders,
    prepare_utsd_for_training,
    prepare_s3_for_pretraining,
    prepare_csv_for_pretraining,
    download_utsd_dataset,
    load_standard_dataset,
    prepare_multiple_datasets
)
from training import Trainer, FineTuneTrainer
from utils import count_parameters, load_pretrained_model


def create_model_from_config(config_dict, pretrained_path=None):
    """
    Create model from configuration
    
    Args:
        config_dict: Model configuration dictionary
        pretrained_path: Path to pretrained model (if None, create from scratch)
    
    Returns:
        model: Timer model
    """
    # Create model configuration
    model_config = TimerConfig(
        input_token_len=config_dict.get('input_token_len', 96),
        hidden_size=config_dict.get('hidden_size', 1024),
        intermediate_size=config_dict.get('intermediate_size', 2048),
        num_hidden_layers=config_dict.get('num_hidden_layers', 8),
        num_attention_heads=config_dict.get('num_attention_heads', 8),
        output_token_lens=config_dict.get('output_token_lens', [96]),
        max_position_embeddings=config_dict.get('max_position_embeddings', 10000),
    )
    
    if pretrained_path and os.path.exists(pretrained_path):
        # Load pretrained model
        print(f"Loading pretrained model from: {pretrained_path}")
        model = load_pretrained_model(pretrained_path)
    else:
        # Create model from scratch
        print("Creating model from scratch...")
        model = TimerForPrediction(model_config)
    
    return model


def load_huggingface_model(model_name, use_mirror=True):
    """
    Load model from HuggingFace (for fine-tuning)
    
    Args:
        model_name: Model name (e.g., 'thuml/timer-base-84m')
        use_mirror: Whether to use mirror
    
    Returns:
        model: HuggingFace model
    """
    if use_mirror:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    from transformers import AutoModelForCausalLM
    
    print(f"Loading HuggingFace model: {model_name}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Unified Timer Training (Pretrain or Finetune)")
    
    # Training mode
    parser.add_argument("--mode", type=str, choices=['pretrain', 'finetune'], required=True,
                       help="Training mode: pretrain or finetune")
    
    # Data related
    parser.add_argument("--data-source", type=str, 
                       choices=['local', 'utsd', 'utsd-s3', 'standard', 'csv'],
                       default='local',
                       help="Data source: local, utsd, utsd-s3, standard (ETTH1, ECL, etc.), or csv")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Data directory (for local data and cache)")
    parser.add_argument("--utsd-subset", type=str, default=None,
                       help="UTSD subset (UTSD-1G, UTSD-2G, UTSD-4G, UTSD-12G)")
    parser.add_argument("--utsd-max-samples", type=int, default=None,
                       help="Maximum samples from UTSD (None for all)")
    parser.add_argument("--use-s3", action="store_true",
                       help="Use S3 format preprocessing (recommended for pretraining)")
    parser.add_argument("--context-length", type=int, default=512,
                       help="Context length for S3 format (default: 512)")
    parser.add_argument("--s3-train-samples", type=int, default=None,
                       help="Number of training samples for S3 format (None = use all available)")
    parser.add_argument("--s3-val-samples", type=int, default=None,
                       help="Number of validation samples for S3 format")
    parser.add_argument("--use-cache", action="store_true",
                       help="Use cached processed data if available (default: True)")
    parser.add_argument("--no-cache", action="store_true",
                       help="Force reprocess data, ignore cache")
    parser.add_argument("--standard-dataset", type=str, default=None,
                       choices=['ETTH1', 'ETTH2', 'ETTM1', 'ETTM2', 'ECL', 'TRAFFIC', 'WEATHER', 
                               'PEMS03', 'PEMS04', 'PEMS07', 'PEMS08'],
                       help="Standard dataset name (for --data-source standard)")
    parser.add_argument("--standard-datasets", type=str, nargs='+', default=None,
                       help="Multiple standard datasets (e.g., ETTH1 ECL TRAFFIC)")
    parser.add_argument("--csv-path", type=str, default=None,
                       help="Path to CSV file (for --data-source csv)")
    parser.add_argument("--csv-date-col", type=str, default='datetime',
                       help="Date column name in CSV (default: datetime)")
    parser.add_argument("--csv-exclude-cols", type=str, nargs='+', default=None,
                       help="Columns to exclude from CSV processing")
    parser.add_argument("--lookback", type=int, default=672,
                       help="Lookback length (default: 672)")
    parser.add_argument("--pred-len", type=int, default=96,
                       help="Prediction length (default: 96)")
    
    # Model related
    parser.add_argument("--model-structure", type=str, default="base",
                       choices=['tiny', 'small', 'base', 'large'],
                       help="Model structure size")
    parser.add_argument("--pretrained-model", type=str, default=None,
                       help="Path to pretrained model (for finetune or continue pretraining)")
    parser.add_argument("--huggingface-model", type=str, default="thuml/timer-base-84m",
                       help="HuggingFace model name (for finetune from HF)")
    
    # Training configuration
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Output directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-5, 
                       help="Base learning rate (paper: 5e-5)")
    parser.add_argument("--min-learning-rate", type=float, default=2e-6,
                       help="Minimum learning rate for cosine annealing (paper: 2e-6)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio (for linear scheduler)")
    parser.add_argument("--scheduler-type", type=str, default="cosine", choices=["cosine", "linear"],
                       help="Learning rate scheduler type: cosine (paper) or linear")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--deepspeed-config", type=str, default=None,
                       help="Path to DeepSpeed configuration file (optional, enables DeepSpeed training)")
    
    # Model structure configuration (optional, overrides model-structure)
    parser.add_argument("--input-token-len", type=int, default=None, help="Input token length")
    parser.add_argument("--hidden-size", type=int, default=None, help="Hidden size")
    parser.add_argument("--intermediate-size", type=int, default=None, help="Intermediate size")
    parser.add_argument("--num-layers", type=int, default=None, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=None, help="Number of attention heads")
    parser.add_argument("--output-token-lens", type=int, nargs="+", default=None,
                       help="Output token lengths")
    
    args = parser.parse_args()
    
    # Device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Training mode: {args.mode}")
    print(f"Data source: {args.data_source}")
    
    # Model structure configuration
    model_structures = {
        'tiny': {
            'hidden_size': 256,
            'intermediate_size': 512,
            'num_hidden_layers': 4,
            'num_attention_heads': 4,
        },
        'small': {
            'hidden_size': 512,
            'intermediate_size': 1024,
            'num_hidden_layers': 6,
            'num_attention_heads': 8,
        },
        'base': {
            'hidden_size': 1024,
            'intermediate_size': 2048,
            'num_hidden_layers': 8,
            'num_attention_heads': 8,
        },
        'large': {
            'hidden_size': 2048,
            'intermediate_size': 4096,
            'num_hidden_layers': 12,
            'num_attention_heads': 16,
        }
    }
    
    base_config = model_structures[args.model_structure].copy()
    
    # Override configuration (if specific parameters are provided)
    if args.input_token_len is not None:
        base_config['input_token_len'] = args.input_token_len
    else:
        # Paper uses N=15 as token count for pretraining
        # But here we use 96 as default (can be adjusted as needed)
        base_config['input_token_len'] = 96
    
    if args.hidden_size is not None:
        base_config['hidden_size'] = args.hidden_size
    if args.intermediate_size is not None:
        base_config['intermediate_size'] = args.intermediate_size
    if args.num_layers is not None:
        base_config['num_hidden_layers'] = args.num_layers
    if args.num_heads is not None:
        base_config['num_attention_heads'] = args.num_heads
    if args.output_token_lens is not None:
        base_config['output_token_lens'] = args.output_token_lens
    else:
        base_config['output_token_lens'] = [96]
    
    base_config['max_position_embeddings'] = 10000
    
    # Load data
    print("\nLoading data...")
    if args.data_source == 'utsd-s3' or (args.data_source == 'utsd' and args.use_s3):
        # Use S3 format preprocessing (recommended for pretraining)
        print("Using S3 format preprocessing for pretraining...")
        
        # Determine whether to use cache
        use_cache = not args.no_cache  # Use cache by default, unless --no-cache is specified
        
        # Check if cache exists
        cache_dir = args.data_dir
        cache_exists = (
            os.path.exists(os.path.join(cache_dir, "data_config.pkl")) and
            os.path.exists(os.path.join(cache_dir, "train_sequences.pkl")) and
            os.path.exists(os.path.join(cache_dir, "val_sequences.pkl"))
        )
        
        # If using cache and cache exists, don't download dataset
        if use_cache and cache_exists:
            print(f"\n[INFO] Cache found at {cache_dir}, will try to use it...")
            hf_dataset = None  # No need to download
        else:
            # Download dataset
            print("\n[INFO] Downloading UTSD dataset...")
            hf_dataset = download_utsd_dataset(
                dataset_name="thuml/UTSD",
                subset=args.utsd_subset,
                use_mirror=True
            )
        
        # Prepare S3 format data
        train_dataset, val_dataset, data_config = prepare_s3_for_pretraining(
            hf_dataset=hf_dataset['train'] if hf_dataset else None,
            context_length=args.context_length,
            lookback=base_config.get('input_token_len', 96) * 5,  # Default 5x input_token_len
            pred_len=base_config.get('input_token_len', 96),
            train_ratio=0.9,
            val_ratio=0.1,
            max_variates=args.utsd_max_samples,
            num_train_samples=args.s3_train_samples,
            num_val_samples=args.s3_val_samples,
            output_dir=cache_dir,
            random_seed=42,
            use_cache=use_cache
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == 'cuda')
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == 'cuda')
        )
        
        lookback = data_config['lookback']
        pred_len = data_config['pred_len']
        test_loader = None  # S3 format is mainly for pretraining, no test set provided
        
    elif args.data_source == 'utsd':
        # Use UTSD dataset (original format)
        train_dataset, val_dataset, test_dataset, data_config = prepare_utsd_for_training(
            subset=args.utsd_subset,
            lookback=512,
            pred_len=96,
            max_samples=args.utsd_max_samples,
            use_mirror=True,
            output_dir=args.data_dir
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == 'cuda')
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == 'cuda')
        )
        
        lookback = data_config['lookback']
        pred_len = data_config['pred_len']
        test_loader = None
        
    elif args.data_source == 'standard':
        # Use standard datasets (ETTH1, ECL, TRAFFIC, etc.)
        print("Using standard time series datasets...")
        
        if args.standard_datasets:
            # Multiple datasets
            dataset_names = args.standard_datasets
            print(f"Loading multiple datasets: {', '.join(dataset_names)}")
            train_dataset, val_dataset, test_dataset, data_config = prepare_multiple_datasets(
                dataset_names=dataset_names,
                lookback=args.lookback,
                pred_len=args.pred_len,
                data_dir=args.data_dir,
                output_dir=os.path.join(args.data_dir, "combined"),
                download=True
            )
        elif args.standard_dataset:
            # Single dataset
            print(f"Loading dataset: {args.standard_dataset}")
            train_dataset, val_dataset, test_dataset, data_config = load_standard_dataset(
                dataset_name=args.standard_dataset,
                lookback=args.lookback,
                pred_len=args.pred_len,
                data_dir=args.data_dir,
                download=True
            )
        else:
            raise ValueError("Please specify --standard-dataset or --standard-datasets")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == 'cuda')
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == 'cuda')
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == 'cuda')
        )
        
        lookback = data_config['lookback']
        pred_len = data_config['pred_len']
        
    elif args.data_source == 'csv':
        # Use CSV file for S3 format pretraining
        print("Using CSV file for S3 format pretraining...")
        
        if not args.csv_path:
            raise ValueError("Please specify --csv-path when using --data-source csv")
        
        if not os.path.exists(args.csv_path):
            raise FileNotFoundError(f"CSV file not found: {args.csv_path}")
        
        # Determine whether to use cache
        use_cache = not args.no_cache  # Use cache by default, unless --no-cache is specified
        
        # Prepare S3 format data
        train_dataset, val_dataset, data_config = prepare_csv_for_pretraining(
            csv_path=args.csv_path,
            context_length=args.context_length,
            lookback=base_config.get('input_token_len', 96) * 5,  # Default 5x input_token_len
            pred_len=base_config.get('input_token_len', 96),
            train_ratio=0.9,
            val_ratio=0.1,
            date_col=args.csv_date_col,
            exclude_cols=args.csv_exclude_cols,
            max_variates=args.utsd_max_samples,
            num_train_samples=args.s3_train_samples,
            num_val_samples=args.s3_val_samples,
            output_dir=args.data_dir,
            random_seed=42,
            use_cache=use_cache
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == 'cuda')
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == 'cuda')
        )
        
        lookback = data_config['lookback']
        pred_len = data_config['pred_len']
        test_loader = None  # S3 format is mainly for pretraining, no test set provided
        
    else:
        # Use local data
        train_loader, val_loader, test_loader, data_config = create_dataloaders(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=(device.type == 'cuda')
        )
        lookback = data_config['lookback']
        pred_len = data_config['pred_len']
    
    print(f"Data loaded: lookback={lookback}, pred_len={pred_len}")
    
    # Create or load model
    if args.mode == 'pretrain':
        # Pretraining mode: train from scratch or continue training
        model = create_model_from_config(base_config, args.pretrained_model)
    else:
        # Fine-tuning mode
        if args.pretrained_model and os.path.exists(args.pretrained_model):
            # Fine-tune from local pretrained model
            print("Fine-tuning from local pretrained model...")
            model = create_model_from_config(base_config, args.pretrained_model)
        else:
            # Fine-tune from HuggingFace model
            print("Fine-tuning from HuggingFace model...")
            model = load_huggingface_model(args.huggingface_model, use_mirror=True)
    
    # Count parameters
    param_stats = count_parameters(model)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {param_stats['total']:,}")
    print(f"  Trainable parameters: {param_stats['trainable']:,}")
    print(f"  Trainable ratio: {param_stats['trainable_ratio']*100:.2f}%")
    
    # Training configuration
    train_config = {
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'min_learning_rate': args.min_learning_rate,
        'weight_decay': args.weight_decay,
        'warmup_ratio': args.warmup_ratio,
        'scheduler_type': args.scheduler_type,
        'lookback': lookback,
        'pred_len': pred_len,
    }
    
    # Select trainer
    deepspeed_config = args.deepspeed_config if args.deepspeed_config and os.path.exists(args.deepspeed_config) else None
    
    if args.mode == 'pretrain':
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=train_config,
            device=device,
            output_dir=args.output_dir,
            deepspeed_config=deepspeed_config
        )
    else:
        trainer = FineTuneTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=train_config,
            device=device,
            output_dir=args.output_dir,
            deepspeed_config=deepspeed_config
        )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()

