# TIMER Unified Training Framework

A complete Timer model training framework supporting both pretraining and fine-tuning, with support for multiple model structures and datasets.

## ✨ Features

- **Unified Training Entry**: One script supports both pretraining and fine-tuning
- **Multiple Model Structures**: Supports tiny/small/base/large model structures
- **Multiple Data Sources**: Supports local data and UTSD dataset
- **Mirror Support**: Automatically downloads models and datasets from hf-mirror.com
- **Flexible Configuration**: Supports command-line arguments and configuration files
- **Modular Design**: Clear code structure, easy to maintain and extend

## 📋 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. One-Click Run (Pretraining + Evaluation)

Use the provided scripts to complete pretraining and evaluation in one go:

**Linux/Mac (Bash):**
```bash
# Complete workflow: pretraining + evaluation
bash scripts/run_pretrain_and_eval.sh

# Only run pretraining
bash scripts/run_pretrain_and_eval.sh --skip-eval

# Only run evaluation (requires existing pretrained model)
bash scripts/run_pretrain_and_eval.sh --skip-pretrain

# Show help
bash scripts/run_pretrain_and_eval.sh --help
```

**Windows (Batch):**
```cmd
REM Complete workflow: pretraining + evaluation
scripts\run_pretrain_and_eval.bat

REM Only run pretraining
scripts\run_pretrain_and_eval.bat --skip-eval

REM Only run evaluation (requires existing pretrained model)
scripts\run_pretrain_and_eval.bat --skip-pretrain

REM Show help
scripts\run_pretrain_and_eval.bat --help
```

The scripts will automatically:
- ✅ Check Python environment and dependencies
- ✅ Check if CUDA is available
- ✅ Create necessary directory structure
- ✅ Download UTSD dataset (S3 format)
- ✅ Perform pretraining (using paper-recommended hyperparameters)
- ✅ Evaluate on ETTH1, ECL, TRAFFIC, WEATHER, PEMS03, PEMS04
- ✅ Save results to `outputs/` directory
- ✅ Display evaluation results summary

### 3. Start Training

#### Pretraining with UTSD Dataset (Recommended: S3 Format)

```bash
# Use S3 format preprocessing (recommended, follows paper methodology)
python scripts/train.py \
    --mode pretrain \
    --data-source utsd-s3 \
    --utsd-subset UTSD-1G \
    --model-structure base \
    --context-length 512 \
    --batch-size 4 \
    --num-epochs 20 \
    --output-dir pretrain_outputs

# Or use original UTSD format
python scripts/train.py \
    --mode pretrain \
    --data-source utsd \
    --utsd-subset UTSD-1G \
    --model-structure base \
    --batch-size 4 \
    --num-epochs 20 \
    --output-dir pretrain_outputs
```

#### Fine-tuning with Standard Datasets (ETTH1, ECL, TRAFFIC, etc.)

```bash
# Single dataset
python scripts/train.py \
    --mode finetune \
    --data-source standard \
    --standard-dataset ETTH1 \
    --lookback 672 \
    --pred-len 96 \
    --batch-size 4 \
    --num-epochs 10 \
    --output-dir finetune_etth1

# Multiple datasets
python scripts/train.py \
    --mode finetune \
    --data-source standard \
    --standard-datasets ETTH1 ECL TRAFFIC WEATHER PEMS03 PEMS04 \
    --lookback 672 \
    --pred-len 96 \
    --batch-size 4 \
    --num-epochs 10 \
    --output-dir finetune_multiple
```

#### Fine-tuning with Local Data

```bash
# 1. Prepare data
python scripts/prepare_data.py --csv-path <your_data.csv> --output-dir data

# 2. Start fine-tuning
python scripts/train.py \
    --mode finetune \
    --data-source local \
    --data-dir data \
    --batch-size 4 \
    --num-epochs 10 \
    --output-dir finetune_outputs
```

#### Pretraining with CSV File (Cryptocurrency Dataset)

The framework supports pretraining directly on CSV files, which is particularly useful for cryptocurrency time series data. Each column in the CSV file is treated as an independent time series variable.

**Dataset Format**:
- CSV file with datetime column (optional, will be excluded automatically)
- Multiple numeric columns representing different features/factors
- Each column is processed as a separate univariate time series
- Example: `selected_factors.csv` contains cryptocurrency factor data with datetime and multiple factor columns

**Usage**:
```bash
# Pretrain on CSV file (e.g., selected_factors.csv - cryptocurrency factors)
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

# Or use the convenience script
scripts\pretrain_csv.bat  # Windows
```

**CSV Pretraining Process**:
1. Each numeric column is extracted as an independent time series
2. Each series is normalized using training set statistics (9:1 split)
3. Normalized sequences are merged into a single sequence pool
4. Fixed-length windows are sampled uniformly from the pool
5. The model is trained on these windows using S3 format

**Parameters**:
- `--csv-path`: Path to CSV file
- `--csv-date-col`: Date column name (default: 'datetime', will be excluded)
- `--csv-exclude-cols`: Additional columns to exclude (optional)
- `--max-variates`: Maximum number of columns to process (optional, for limiting data size)

## 📖 Detailed Usage

### Training Modes

- `--mode pretrain`: Pretrain from scratch
- `--mode finetune`: Fine-tune (from pretrained model or HuggingFace model)

### Data Sources

- `--data-source local`: Use local data (prepared via prepare_data.py)
- `--data-source utsd`: Use UTSD dataset (original format, auto-download)
- `--data-source utsd-s3`: Use UTSD dataset (S3 format, recommended for pretraining)
- `--data-source standard`: Use standard time series datasets (ETTH1, ECL, TRAFFIC, WEATHER, PEMS03, PEMS04, etc.)
- `--data-source csv`: Use CSV file for pretraining

**S3 Format Description**:
S3 (Single-Series Sequence) format is a preprocessing method proposed in the paper, suitable for pretraining:
- Each variable sequence is split 9:1, normalized using training set statistics
- Normalized sequences are merged into a single-variate sequence pool
- Fixed-length window sequences are uniformly sampled from the pool
- No time alignment required, suitable for a wide range of univariate and irregular time series

**Local Cache Functionality**:
- Processed data is automatically saved to `data/` directory
- Second run will automatically use cache, no need to re-download and process
- Use `--no-cache` to force reprocessing
- Cache files include:
  - `train_sequences.pkl`: Training sequences
  - `val_sequences.pkl`: Validation sequences
  - `data_config.pkl`: Data configuration

### Model Structures

- `--model-structure tiny`: Small model (256 hidden, 4 layers)
- `--model-structure small`: Medium-small model (512 hidden, 6 layers)
- `--model-structure base`: Base model (1024 hidden, 8 layers)
- `--model-structure large`: Large model (2048 hidden, 12 layers)

You can also override with custom parameters:
```bash
--hidden-size 512 --num-layers 6 --num-heads 8
```

### Training Hyperparameters (Paper Settings)

- **Optimizer**: AdamW (default)
- **Learning Rate Schedule**: Cosine Annealing (default)
  - Base learning rate: `5e-5` (paper default)
  - Final learning rate: `2e-6` (paper default)
  - Decay steps: Proportional to training steps for 10 epochs
- **Batch Size**: Paper uses 8192 (adjust based on GPU memory)
- **Pretraining Token Count**: N=15 (can be set via `--input-token-len`)

### Standard Time Series Datasets

Supports the following standard datasets (auto-download):
- `ETTH1`, `ETTH2`: Electric transformer temperature data
- `ETTM1`, `ETTM2`: Electric transformer temperature data (minute-level)
- `ECL`: Electricity consumption data
- `TRAFFIC`: Traffic flow data
- `WEATHER`: Weather data
- `PEMS03`, `PEMS04`, `PEMS07`, `PEMS08`: Traffic sensor data

**Default Settings**:
- Lookback length: 672
- Prediction length: 96

**Usage Examples**:
```bash
# Single dataset
--data-source standard --standard-dataset ETTH1

# Multiple datasets
--data-source standard --standard-datasets ETTH1 ECL TRAFFIC WEATHER PEMS03 PEMS04
```

### UTSD Dataset Subsets

- `UTSD-1G`: 1GB data subset (~68.7k samples)
- `UTSD-2G`: 2GB data subset (~75.4k samples)
- `UTSD-4G`: 4GB data subset
- `UTSD-12G`: 12GB data subset
- Not specified: Use full dataset (~434k samples)

### S3 Format Parameters

- `--context-length`: Context length for S3 format (default 512)
- `--s3-train-samples`: Number of training samples (None means use all available samples)
- `--s3-val-samples`: Number of validation samples

### CSV File Parameters (Cryptocurrency Dataset)

- `--csv-path`: Path to CSV file (required for `--data-source csv`)
- `--csv-date-col`: Date column name to exclude (default: 'datetime')
- `--csv-exclude-cols`: Additional columns to exclude (optional)
- `--max-variates`: Maximum number of columns to process (optional, for limiting data size)

**Cryptocurrency Dataset Example**:
The `selected_factors.csv` file contains cryptocurrency factor data with:
- `datetime`: Timestamp column (automatically excluded)
- Multiple factor columns: Various technical indicators and features (e.g., alpha_volumeBS_2MA, alpha_opint_volume, caspar_hf_factor, etc.)
- Each factor column is treated as an independent time series
- The framework automatically normalizes and processes each column using S3 format
- Suitable for pretraining on multi-factor cryptocurrency time series data

### Complete Parameter List

```bash
python scripts/train.py --help
```

## 🔧 Project Structure

```
timer_finetune/
├── models/              # Model modules
│   ├── __init__.py
│   ├── timer_config.py  # Timer model configuration
│   └── timer_model.py   # Timer model implementation
│
├── data_processing/     # Data processing modules (code)
│   ├── __init__.py
│   ├── dataset.py       # Time series dataset classes
│   ├── data_loader.py   # Data loaders
│   ├── utsd_dataset.py  # UTSD dataset support
│   ├── s3_preprocessor.py  # S3 format preprocessing
│   └── standard_datasets.py  # Standard dataset support
│
├── data/                # Data directory (actual data files)
│   ├── utsd/            # UTSD dataset cache
│   ├── s3/              # S3 format data
│   └── standard_datasets/  # Standard datasets
│
├── training/            # Training modules
│   ├── __init__.py
│   ├── trainer.py       # Pretraining trainer
│   └── finetune_trainer.py  # Fine-tuning trainer
│
├── utils/               # Utility function modules
│   ├── __init__.py
│   └── model_utils.py   # Model utility functions
│
├── scripts/             # Script modules
│   ├── __init__.py
│   ├── train.py         # Unified training entry
│   ├── evaluate.py      # Model evaluation script
│   ├── run_pretrain_and_eval.sh  # One-click run script (Linux/Mac)
│   ├── run_pretrain_and_eval.bat # One-click run script (Windows)
│   ├── pretrain_csv.bat # CSV pretraining script
│   └── prepare_data.py  # Data preparation script
│
├── outputs/             # Output directory (models and results)
│
├── README.md            # This document
├── requirements.txt     # Dependencies
└── LICENSE              # License
```

**Note**:
- `data_processing/` folder contains data processing related **code modules** (to avoid conflicts with HuggingFace's datasets library)
- `data/` folder stores **actual data files** (downloaded datasets, preprocessed data, etc.)
- `outputs/` folder stores training outputs (model weights, training history, evaluation results, etc.)

## 🚀 Usage Examples

### Pretrain Small Model (S3 Format, Paper Hyperparameters)

```bash
python scripts/train.py \
    --mode pretrain \
    --data-source utsd-s3 \
    --utsd-subset UTSD-1G \
    --model-structure small \
    --context-length 512 \
    --batch-size 8 \
    --num-epochs 10 \
    --learning-rate 5e-5 \
    --min-learning-rate 2e-6 \
    --scheduler-type cosine \
    --output-dir pretrain_small
```

**Note**: The paper uses batch size=8192, but adjust based on GPU memory. You can use gradient accumulation to simulate large batch size.

### Fine-tune Large Model

```bash
python scripts/train.py \
    --mode finetune \
    --data-source local \
    --data-dir data \
    --model-structure large \
    --pretrained-model pretrain_outputs/best_model \
    --batch-size 2 \
    --num-epochs 20 \
    --learning-rate 1e-5 \
    --output-dir finetune_large
```

### Download UTSD Dataset

```bash
python scripts/download_utsd.py --subset UTSD-1G
```

## 📊 Output Files

After training completes, the output directory contains:

- `best_model/`: Best model on validation set
  - `model.pt`: Model weights
  - `config.json`: Model configuration
  - `optimizer.pt`: Optimizer state
- `final_model/`: Model from last epoch
- `training_history.json`: Training history data
- `training_curves.png`: Training curves plot

## 🔄 Workflow

### Complete Pretraining Workflow

1. **Download UTSD Dataset** (optional)
   ```bash
   python scripts/download_utsd.py --subset UTSD-1G
   ```

2. **Start Pretraining**
   ```bash
   python scripts/train.py --mode pretrain --data-source utsd --utsd-subset UTSD-1G
   ```

3. **Fine-tune with Pretrained Model**
   ```bash
   python scripts/train.py --mode finetune --pretrained-model pretrain_outputs/best_model
   ```

### Fine-tuning Workflow

1. **Prepare Local Data**
   ```bash
   python scripts/prepare_data.py --csv-path <path> --output-dir data
   ```

2. **Fine-tune from HuggingFace Model**
   ```bash
   python scripts/train.py --mode finetune --data-source local --data-dir data
   ```

## 🌐 Mirror Support

The framework automatically uses hf-mirror.com mirror, no additional configuration needed. To switch:

```bash
export HF_ENDPOINT=https://hf-mirror.com  # Use mirror
export HF_ENDPOINT=https://huggingface.co  # Use official
```

## 📝 Notes

1. **Memory Usage**: Adjust batch_size based on GPU memory
2. **Training Time**: Pretraining takes a long time, GPU recommended
3. **Data Download**: UTSD dataset is large, first download takes time
4. **Model Saving**: Models are automatically saved as best and final versions

## 🤝 Getting Help

```bash
python scripts/train.py --help
```

## 📚 Related Resources

- [UTSD Dataset](https://huggingface.co/datasets/thuml/UTSD)
- [Timer Model](https://huggingface.co/thuml/timer-base-84m)

## 📊 Model Evaluation

### Evaluate on Standard Datasets

```bash
# Evaluate with pretrained model
python scripts/evaluate.py \
    --model-path pretrain_outputs/best_model \
    --datasets ETTH1 ECL TRAFFIC WEATHER PEMS03 PEMS04 \
    --lookback 672 \
    --pred-len 96 \
    --batch-size 32 \
    --output-dir evaluation_results

# Evaluate with HuggingFace model
python scripts/evaluate.py \
    --huggingface-model thuml/timer-base-84m \
    --datasets ETTH1 ECL TRAFFIC WEATHER PEMS03 PEMS04 \
    --lookback 672 \
    --pred-len 96 \
    --output-dir evaluation_results

# Note: Must use --datasets (two dashes), not datasets
```

### Evaluation Metrics

- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **Direction Acc**: Direction Accuracy (whether prediction direction is correct)

Evaluation results are saved as JSON file and printed as summary table.

## 📄 License

Please refer to the LICENSE file.
