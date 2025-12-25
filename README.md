# TIMER Unified Training Framework

A complete Timer model training framework supporting both pretraining and fine-tuning, with support for multiple model structures and datasets.

## Table of Contents

1. [Features](#-features)
2. [Installation](#-installation)
3. [Quick Start](#-quick-start)
4. [Data Sources](#-data-sources)
   - [UTSD Dataset](#utsd-dataset)
   - [Standard Time Series Datasets](#standard-time-series-datasets)
   - [CSV Files (Cryptocurrency Dataset)](#csv-files-cryptocurrency-dataset)
   - [Local Data](#local-data)
5. [Training](#-training)
   - [Training Modes](#training-modes)
   - [Model Structures](#model-structures)
   - [Hyperparameters](#hyperparameters)
   - [S3 Format](#s3-format)
6. [Evaluation](#-evaluation)
7. [Configuration](#-configuration)
8. [Project Structure](#-project-structure)
9. [Examples](#-examples)
10. [Advanced Usage](#-advanced-usage)
11. [Troubleshooting](#-troubleshooting)
12. [References](#-references)

---

## ✨ Features

- **Unified Training Entry**: One script supports both pretraining and fine-tuning
- **Multiple Model Structures**: Supports tiny/small/base/large model structures
- **Multiple Data Sources**: Supports UTSD, standard datasets, CSV files, and local data
- **Cryptocurrency Dataset Support**: Direct pretraining on CSV files with multiple factor columns
- **DeepSpeed Acceleration**: Optional DeepSpeed support for faster training and larger models
- **Mirror Support**: Automatically downloads models and datasets from hf-mirror.com
- **Flexible Configuration**: Supports command-line arguments and configuration files
- **Modular Design**: Clear code structure, easy to maintain and extend

---

## 📦 Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### One-Click Run (Pretraining + Evaluation)

Use the provided scripts to complete pretraining and evaluation in one go:

**Linux/Mac:**
```bash
bash scripts/run_pretrain_and_eval.sh
```

**Windows:**
```cmd
scripts\run_pretrain_and_eval.bat
```

**Options:**
- `--skip-pretrain`: Skip pretraining, only run evaluation
- `--skip-eval`: Skip evaluation, only run pretraining
- `--help`: Show help message

The scripts will automatically:
- ✅ Check Python environment and dependencies
- ✅ Check if CUDA is available
- ✅ Create necessary directory structure
- ✅ Download UTSD dataset (S3 format)
- ✅ Perform pretraining (using paper-recommended hyperparameters)
- ✅ Evaluate on ETTH1, ECL, TRAFFIC, WEATHER, PEMS03, PEMS04
- ✅ Save results to `outputs/` directory

### Basic Training Commands

**Pretraining:**
```bash
python scripts/train.py --mode pretrain --data-source utsd-s3 --utsd-subset UTSD-1G
```

**Fine-tuning:**
```bash
python scripts/train.py --mode finetune --data-source standard --standard-dataset ETTH1
```

**CSV Pretraining (Cryptocurrency):**
```bash
python scripts/train.py --mode pretrain --data-source csv --csv-path data/selected_factors.csv
```

---

## 📊 Data Sources

The framework supports multiple data sources for different use cases.

### UTSD Dataset

**Description**: Large-scale univariate time series dataset for pretraining.

**Subsets**:
- `UTSD-1G`: 1GB data subset (~68.7k samples)
- `UTSD-2G`: 2GB data subset (~75.4k samples)
- `UTSD-4G`: 4GB data subset
- `UTSD-12G`: 12GB data subset
- Not specified: Full dataset (~434k samples)

**Usage**:
```bash
# S3 format (recommended for pretraining)
python scripts/train.py \
    --mode pretrain \
    --data-source utsd-s3 \
    --utsd-subset UTSD-1G \
    --context-length 512

# Original format
python scripts/train.py \
    --mode pretrain \
    --data-source utsd \
    --utsd-subset UTSD-1G
```

**Download**:
```bash
python scripts/download_utsd.py --subset UTSD-1G
```

### Standard Time Series Datasets

**Description**: Standard benchmark datasets for time series forecasting.

**Supported Datasets**:
- `ETTH1`, `ETTH2`: Electric transformer temperature data
- `ETTM1`, `ETTM2`: Electric transformer temperature data (minute-level)
- `ECL`: Electricity consumption data
- `TRAFFIC`: Traffic flow data
- `WEATHER`: Weather data
- `PEMS03`, `PEMS04`, `PEMS07`, `PEMS08`: Traffic sensor data

**Default Settings**:
- Lookback length: 672
- Prediction length: 96

**Usage**:
```bash
# Single dataset
python scripts/train.py \
    --mode finetune \
    --data-source standard \
    --standard-dataset ETTH1 \
    --lookback 672 \
    --pred-len 96

# Multiple datasets
python scripts/train.py \
    --mode finetune \
    --data-source standard \
    --standard-datasets ETTH1 ECL TRAFFIC WEATHER PEMS03 PEMS04
```

### CSV Files (Cryptocurrency Dataset)

**Description**: Direct pretraining on CSV files, particularly useful for cryptocurrency time series data. Each column is treated as an independent time series variable.

**Dataset Format**:
- CSV file with datetime column (optional, automatically excluded)
- Multiple numeric columns representing different features/factors
- Each column is processed as a separate univariate time series

**Example**: `selected_factors.csv` contains cryptocurrency factor data with:
- `datetime`: Timestamp column (automatically excluded)
- Multiple factor columns: Technical indicators and features (e.g., alpha_volumeBS_2MA, alpha_opint_volume, caspar_hf_factor, etc.)

**Usage**:
```bash
# Direct pretraining on CSV
python scripts/train.py \
    --mode pretrain \
    --data-source csv \
    --csv-path data/selected_factors.csv \
    --csv-date-col datetime \
    --model-structure base \
    --context-length 512 \
    --batch-size 4 \
    --num-epochs 10 \
    --output-dir pretrain_csv_outputs

# Or use convenience script (Windows)
scripts\pretrain_csv.bat
```

**Processing Pipeline**:
1. Each numeric column is extracted as an independent time series
2. Each series is normalized using training set statistics (9:1 split)
3. Normalized sequences are merged into a single sequence pool
4. Fixed-length windows are sampled uniformly from the pool
5. Model is trained on these windows using S3 format

**Parameters**:
- `--csv-path`: Path to CSV file (required)
- `--csv-date-col`: Date column name to exclude (default: 'datetime')
- `--csv-exclude-cols`: Additional columns to exclude (optional)
- `--max-variates`: Maximum number of columns to process (optional)

### Local Data

**Description**: Use locally prepared data files.

**Preparation**:
```bash
python scripts/prepare_data.py --csv-path <your_data.csv> --output-dir data
```

**Usage**:
```bash
python scripts/train.py \
    --mode finetune \
    --data-source local \
    --data-dir data \
    --batch-size 4 \
    --num-epochs 10
```

---

## 🎓 Training

### Training Modes

- **`--mode pretrain`**: Pretrain from scratch
  - Suitable for: UTSD dataset, CSV files
  - Output: Pretrained model weights

- **`--mode finetune`**: Fine-tune from pretrained model
  - Suitable for: Standard datasets, local data
  - Input: Pretrained model (local or HuggingFace)
  - Output: Fine-tuned model weights

### Model Structures

Predefined model structures:

| Structure | Hidden Size | Layers | Heads | Use Case |
|-----------|-------------|--------|-------|----------|
| `tiny`    | 256         | 4      | 4     | Quick testing |
| `small`   | 512         | 6      | 8     | Small-scale experiments |
| `base`    | 1024        | 8      | 8     | Standard training (recommended) |
| `large`   | 2048        | 12     | 16    | Large-scale training |

**Usage**:
```bash
--model-structure base
```

**Custom Parameters**:
```bash
--hidden-size 512 --num-layers 6 --num-heads 8
```

### Hyperparameters

**Default Settings (Paper Recommendations)**:

- **Optimizer**: AdamW
- **Learning Rate Schedule**: Cosine Annealing
  - Base learning rate: `5e-5`
  - Minimum learning rate: `2e-6`
  - Decay steps: Proportional to training steps for 10 epochs
- **Batch Size**: Adjust based on GPU memory (paper uses 8192)
- **Weight Decay**: `0.01`
- **Pretraining Token Count**: N=15 (configurable via `--input-token-len`)

**Configuration**:
```bash
--learning-rate 5e-5 \
--min-learning-rate 2e-6 \
--scheduler-type cosine \
--batch-size 4 \
--num-epochs 10
```

### S3 Format

**Description**: Single-Series Sequence (S3) format is a preprocessing method proposed in the paper, suitable for pretraining on heterogeneous time series.

**Key Features**:
- Each variable sequence is split 9:1 (train/val)
- Normalized using training set statistics
- Normalized sequences merged into single-variate sequence pool
- Fixed-length windows uniformly sampled from pool
- No time alignment required
- Suitable for univariate and irregular time series

**Parameters**:
- `--context-length`: Context length for S3 format (default: 512)
- `--s3-train-samples`: Number of training samples (None = use all)
- `--s3-val-samples`: Number of validation samples

**Cache**:
- Processed data automatically saved to `data/` directory
- Second run uses cache automatically
- Use `--no-cache` to force reprocessing

---

## 📈 Evaluation

### Evaluate on Standard Datasets

```bash
# Using pretrained model
python scripts/evaluate.py \
    --model-path pretrain_outputs/best_model \
    --datasets ETTH1 ECL TRAFFIC WEATHER PEMS03 PEMS04 \
    --lookback 672 \
    --pred-len 96 \
    --batch-size 32 \
    --output-dir evaluation_results

# Using HuggingFace model
python scripts/evaluate.py \
    --huggingface-model thuml/timer-base-84m \
    --datasets ETTH1 ECL TRAFFIC WEATHER PEMS03 PEMS04 \
    --lookback 672 \
    --pred-len 96 \
    --output-dir evaluation_results
```

### Evaluation Metrics

- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **Direction Acc**: Direction Accuracy (prediction direction correctness)

Results are saved as JSON file and printed as summary table.

---

## ⚙️ Configuration

### Complete Parameter List

```bash
python scripts/train.py --help
```

### Key Parameters

**Data Source**:
- `--data-source`: `local`, `utsd`, `utsd-s3`, `standard`, `csv`
- `--data-dir`: Data directory path
- `--csv-path`: CSV file path (for CSV source)
- `--csv-date-col`: Date column name (default: 'datetime')
- `--standard-dataset`: Single standard dataset name
- `--standard-datasets`: Multiple standard dataset names

**Model**:
- `--model-structure`: `tiny`, `small`, `base`, `large`
- `--pretrained-model`: Path to pretrained model
- `--huggingface-model`: HuggingFace model name

**Training**:
- `--mode`: `pretrain` or `finetune`
- `--batch-size`: Batch size
- `--num-epochs`: Number of epochs
- `--learning-rate`: Base learning rate
- `--min-learning-rate`: Minimum learning rate
- `--scheduler-type`: `cosine` or `linear`
- `--output-dir`: Output directory

**Data Processing**:
- `--context-length`: S3 format context length
- `--lookback`: Lookback window length
- `--pred-len`: Prediction length
- `--use-cache`: Use cached data (default: True)
- `--no-cache`: Force reprocessing

**DeepSpeed**:
- `--deepspeed-config`: Path to DeepSpeed configuration file (enables DeepSpeed training)

---

## 🔧 Project Structure

```
timer_finetune/
├── models/                  # Model modules
│   ├── timer_config.py      # Timer model configuration
│   └── timer_model.py       # Timer model implementation
│
├── data_processing/         # Data processing modules
│   ├── dataset.py           # Time series dataset classes
│   ├── data_loader.py       # Data loaders
│   ├── utsd_dataset.py      # UTSD dataset support
│   ├── s3_preprocessor.py  # S3 format preprocessing
│   └── standard_datasets.py # Standard dataset support
│
├── data/                    # Data directory
│   ├── utsd/                # UTSD dataset cache
│   ├── s3/                  # S3 format data
│   └── standard_datasets/    # Standard datasets
│
├── training/                # Training modules
│   ├── trainer.py           # Pretraining trainer
│   └── finetune_trainer.py  # Fine-tuning trainer
│
├── utils/                   # Utility functions
│   └── model_utils.py       # Model utilities
│
├── scripts/                 # Scripts
│   ├── train.py             # Unified training entry
│   ├── evaluate.py          # Model evaluation
│   ├── run_pretrain_and_eval.sh  # One-click script (Linux/Mac)
│   ├── run_pretrain_and_eval.bat # One-click script (Windows)
│   ├── pretrain_csv.bat     # CSV pretraining script
│   └── prepare_data.py      # Data preparation
│
├── outputs/                 # Output directory
│
├── README.md                # This document
├── requirements.txt         # Dependencies
└── LICENSE                  # License
```

**Note**:
- `data_processing/`: Code modules for data processing
- `data/`: Actual data files (downloaded datasets, preprocessed data)
- `outputs/`: Training outputs (model weights, training history, evaluation results)

---

## 💡 Examples

### Example 1: Pretrain on Cryptocurrency Dataset

```bash
# Pretrain on selected_factors.csv
python scripts/train.py \
    --mode pretrain \
    --data-source csv \
    --csv-path data/selected_factors.csv \
    --csv-date-col datetime \
    --model-structure base \
    --context-length 512 \
    --batch-size 4 \
    --num-epochs 10 \
    --learning-rate 5e-5 \
    --min-learning-rate 2e-6 \
    --scheduler-type cosine \
    --output-dir outputs/pretrain_crypto
```

### Example 2: Pretrain on UTSD (S3 Format)

```bash
python scripts/train.py \
    --mode pretrain \
    --data-source utsd-s3 \
    --utsd-subset UTSD-1G \
    --model-structure base \
    --context-length 512 \
    --batch-size 4 \
    --num-epochs 10 \
    --output-dir outputs/pretrain_utsd
```

### Example 3: Fine-tune on Standard Dataset

```bash
python scripts/train.py \
    --mode finetune \
    --data-source standard \
    --standard-dataset ETTH1 \
    --pretrained-model outputs/pretrain_crypto/best_model \
    --lookback 672 \
    --pred-len 96 \
    --batch-size 4 \
    --num-epochs 10 \
    --output-dir outputs/finetune_etth1
```

### Example 4: Complete Workflow

```bash
# Step 1: Pretrain on cryptocurrency data
python scripts/train.py \
    --mode pretrain \
    --data-source csv \
    --csv-path data/selected_factors.csv \
    --output-dir outputs/pretrain

# Step 2: Evaluate pretrained model
python scripts/evaluate.py \
    --model-path outputs/pretrain/best_model \
    --datasets ETTH1 ECL TRAFFIC \
    --output-dir outputs/evaluation

# Step 3: Fine-tune on specific dataset
python scripts/train.py \
    --mode finetune \
    --data-source standard \
    --standard-dataset ETTH1 \
    --pretrained-model outputs/pretrain/best_model \
    --output-dir outputs/finetune
```

---

## 🔬 Advanced Usage

### DeepSpeed Training Acceleration

DeepSpeed provides significant speedup and memory optimization for large-scale training.

**Benefits**:
- **Faster Training**: Optimized communication and computation
- **Memory Efficiency**: ZeRO optimizer states partitioning
- **Larger Models**: Train models that don't fit in single GPU memory
- **Mixed Precision**: Automatic FP16 support

**Prerequisites**:
```bash
pip install deepspeed
```

**Usage**:
```bash
# Use DeepSpeed with ZeRO Stage 2 (recommended for most cases)
python scripts/train.py \
    --mode pretrain \
    --data-source csv \
    --csv-path data/selected_factors.csv \
    --deepspeed-config deepspeed_config.json \
    --model-structure base \
    --batch-size 8 \
    --num-epochs 10

# Use DeepSpeed with ZeRO Stage 3 (for very large models)
python scripts/train.py \
    --mode pretrain \
    --data-source csv \
    --csv-path data/selected_factors.csv \
    --deepspeed-config deepspeed_config_zero3.json \
    --model-structure large \
    --batch-size 4
```

**DeepSpeed Configuration Files**:
- `deepspeed_config.json`: ZeRO Stage 2 (recommended for most cases)
  - Optimizer state partitioning
  - Gradient partitioning
  - FP16 mixed precision
- `deepspeed_config_zero3.json`: ZeRO Stage 3 (for very large models)
  - Parameter, optimizer, and gradient partitioning
  - CPU offloading support
  - Maximum memory efficiency

**Multi-GPU Training**:
```bash
# DeepSpeed automatically handles multi-GPU training
deepspeed --num_gpus=4 scripts/train.py \
    --mode pretrain \
    --data-source csv \
    --csv-path data/selected_factors.csv \
    --deepspeed-config deepspeed_config.json
```

**Note**: When using DeepSpeed, batch size and learning rate are automatically adjusted per GPU. The configuration files use "auto" values that adapt to your settings.

### Custom Model Architecture

```bash
python scripts/train.py \
    --mode pretrain \
    --data-source csv \
    --csv-path data/selected_factors.csv \
    --hidden-size 768 \
    --num-layers 10 \
    --num-heads 12 \
    --intermediate-size 3072 \
    --input-token-len 96 \
    --output-token-lens 96 192
```

### Gradient Accumulation

For large batch size simulation:
```bash
# Use smaller batch size with gradient accumulation
--batch-size 4  # Effective batch size = 4 * accumulation_steps
```

### Mixed Precision Training

The framework automatically uses mixed precision if available (PyTorch 1.6+). DeepSpeed also provides automatic FP16 support.

### Resume Training

```bash
# Continue from checkpoint
--pretrained-model outputs/pretrain/best_model
```

---

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `--batch-size`
   - Use smaller `--model-structure`
   - Reduce `--context-length`

2. **Slow Data Loading**
   - Use `--use-cache` to enable caching
   - Check if data is already processed

3. **CUDA Not Available**
   - Framework automatically falls back to CPU
   - Training will be slower

4. **Large CSV Files**
   - Use `--max-variates` to limit number of columns
   - Process data in chunks

### Getting Help

```bash
# Show help for training script
python scripts/train.py --help

# Show help for evaluation script
python scripts/evaluate.py --help
```

---

## 📚 References

- [UTSD Dataset](https://huggingface.co/datasets/thuml/UTSD)
- [Timer Model](https://huggingface.co/thuml/timer-base-84m)

### Mirror Configuration

The framework automatically uses hf-mirror.com mirror. To switch:

```bash
export HF_ENDPOINT=https://hf-mirror.com  # Use mirror
export HF_ENDPOINT=https://huggingface.co  # Use official
```

---

## 📄 License

Please refer to the LICENSE file.

---

## 📊 Output Files

After training completes, the output directory contains:

- `best_model/`: Best model on validation set
  - `model.pt`: Model weights
  - `config.json`: Model configuration
  - `optimizer.pt`: Optimizer state
- `final_model/`: Model from last epoch
- `training_history.json`: Training history data
- `training_curves.png`: Training curves plot
