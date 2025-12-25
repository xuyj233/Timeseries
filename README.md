# TIMER 统一训练框架

支持预训练和微调的完整Timer模型训练框架，支持多种模型结构和数据集。

## ✨ 特性

- **统一训练入口**: 一个脚本支持预训练和微调
- **多种模型结构**: 支持tiny/small/base/large模型结构
- **多种数据源**: 支持本地数据和UTSD数据集
- **镜像支持**: 自动从hf-mirror.com下载模型和数据集
- **灵活配置**: 支持命令行参数和配置文件
- **模块化设计**: 清晰的代码结构，易于维护和扩展

## 📋 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 一键运行（预训练 + 评测）

使用提供的脚本一键完成预训练和评测：

**Linux/Mac (Bash):**
```bash
# 完整流程：预训练 + 评测
bash scripts/run_pretrain_and_eval.sh

# 只运行预训练
bash scripts/run_pretrain_and_eval.sh --skip-eval

# 只运行评测（需要已有预训练模型）
bash scripts/run_pretrain_and_eval.sh --skip-pretrain

# 查看帮助
bash scripts/run_pretrain_and_eval.sh --help
```

**Windows (批处理):**
```cmd
REM 完整流程：预训练 + 评测
scripts\run_pretrain_and_eval.bat

REM 只运行预训练
scripts\run_pretrain_and_eval.bat --skip-eval

REM 只运行评测（需要已有预训练模型）
scripts\run_pretrain_and_eval.bat --skip-pretrain

REM 查看帮助
scripts\run_pretrain_and_eval.bat --help
```

脚本会自动：
- ✅ 检查Python环境和依赖
- ✅ 检查CUDA是否可用
- ✅ 创建必要的目录结构
- ✅ 下载UTSD数据集（S3格式）
- ✅ 进行预训练（使用论文推荐的超参数）
- ✅ 在ETTH1, ECL, TRAFFIC, WEATHER, PEMS03, PEMS04上评测
- ✅ 保存结果到 `outputs/` 目录
- ✅ 显示评测结果摘要

### 2. 开始训练

#### 使用UTSD数据集预训练（推荐使用S3格式）

```bash
# 使用S3格式预处理（推荐，符合论文方法）
python scripts/train.py \
    --mode pretrain \
    --data-source utsd-s3 \
    --utsd-subset UTSD-1G \
    --model-structure base \
    --context-length 512 \
    --batch-size 4 \
    --num-epochs 20 \
    --output-dir pretrain_outputs

# 或使用原始UTSD格式
python scripts/train.py \
    --mode pretrain \
    --data-source utsd \
    --utsd-subset UTSD-1G \
    --model-structure base \
    --batch-size 4 \
    --num-epochs 20 \
    --output-dir pretrain_outputs
```

#### 使用标准数据集微调（ETTH1, ECL, TRAFFIC等）

```bash
# 单个数据集
python scripts/train.py \
    --mode finetune \
    --data-source standard \
    --standard-dataset ETTH1 \
    --lookback 672 \
    --pred-len 96 \
    --batch-size 4 \
    --num-epochs 10 \
    --output-dir finetune_etth1

# 多个数据集
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

#### 使用本地数据微调

```bash
# 1. 准备数据
python scripts/prepare_data.py --csv-path <your_data.csv> --output-dir data

# 2. 开始微调
python scripts/train.py \
    --mode finetune \
    --data-source local \
    --data-dir data \
    --batch-size 4 \
    --num-epochs 10 \
    --output-dir finetune_outputs
```

## 📖 详细使用说明

### 训练模式

- `--mode pretrain`: 从头预训练
- `--mode finetune`: 微调（从预训练模型或HuggingFace模型）

### 数据源

- `--data-source local`: 使用本地数据（通过prepare_data.py准备）
- `--data-source utsd`: 使用UTSD数据集（原始格式，自动下载）
- `--data-source utsd-s3`: 使用UTSD数据集（S3格式，推荐用于预训练）
- `--data-source standard`: 使用标准时间序列数据集（ETTH1, ECL, TRAFFIC, WEATHER, PEMS03, PEMS04等）

**S3格式说明**：
S3（Single-Series Sequence）格式是论文中提出的预处理方法，适用于预训练：
- 每个变量序列按9:1分割，使用训练集统计量归一化
- 归一化后的序列合并成单变量序列池
- 从池中均匀采样固定长度的窗口序列
- 不需要时间对齐，适用于广泛的单变量和不规则时间序列

**本地缓存功能**：
- 处理后的数据会自动保存到 `data_cache/` 目录
- 第二次运行时会自动使用缓存，无需重新下载和处理
- 使用 `--no-cache` 强制重新处理数据
- 缓存文件包括：
  - `train_sequences.pkl`: 训练序列
  - `val_sequences.pkl`: 验证序列
  - `data_config.pkl`: 数据配置

### 模型结构

- `--model-structure tiny`: 小模型（256 hidden, 4 layers）
- `--model-structure small`: 中小模型（512 hidden, 6 layers）
- `--model-structure base`: 基础模型（1024 hidden, 8 layers）
- `--model-structure large`: 大模型（2048 hidden, 12 layers）

也可以使用自定义参数覆盖：
```bash
--hidden-size 512 --num-layers 6 --num-heads 8
```

### 训练超参数（论文设置）

- **优化器**: AdamW（默认）
- **学习率调度**: Cosine Annealing（默认）
  - 基础学习率: `5e-5`（论文默认）
  - 最终学习率: `2e-6`（论文默认）
  - 衰减步数: 与10个epoch的训练步数成比例
- **Batch Size**: 论文中使用8192（根据GPU内存调整）
- **预训练Token数**: N=15（可通过`--input-token-len`设置）

### 标准时间序列数据集

支持以下标准数据集（自动下载）：
- `ETTH1`, `ETTH2`: 电力变压器温度数据
- `ETTM1`, `ETTM2`: 电力变压器温度数据（分钟级）
- `ECL`: 电力消耗数据
- `TRAFFIC`: 交通流量数据
- `WEATHER`: 天气数据
- `PEMS03`, `PEMS04`, `PEMS07`, `PEMS08`: 交通传感器数据

**默认设置**：
- Lookback length: 672
- Prediction length: 96

**使用示例**：
```bash
# 单个数据集
--data-source standard --standard-dataset ETTH1

# 多个数据集
--data-source standard --standard-datasets ETTH1 ECL TRAFFIC WEATHER PEMS03 PEMS04
```

### UTSD数据集子集

- `UTSD-1G`: 1GB数据子集（约68.7k样本）
- `UTSD-2G`: 2GB数据子集（约75.4k样本）
- `UTSD-4G`: 4GB数据子集
- `UTSD-12G`: 12GB数据子集
- 不指定: 使用完整数据集（约434k样本）

### S3格式参数

- `--context-length`: S3格式的上下文长度（默认512）
- `--s3-train-samples`: 训练样本数量（None表示使用所有可用样本）
- `--s3-val-samples`: 验证样本数量

### 完整参数列表

```bash
python scripts/train.py --help
```

## 🔧 项目结构

```
timer_finetune/
├── models/              # 模型模块
│   ├── __init__.py
│   ├── timer_config.py  # Timer模型配置
│   └── timer_model.py   # Timer模型实现
│
├── data_processing/     # 数据处理模块（代码）
│   ├── __init__.py
│   ├── dataset.py       # 时间序列数据集类
│   ├── data_loader.py   # 数据加载器
│   ├── utsd_dataset.py  # UTSD数据集支持
│   ├── s3_preprocessor.py  # S3格式预处理
│   └── standard_datasets.py  # 标准数据集支持
│
├── data_cache/          # 数据缓存目录（实际数据文件）
│   ├── utsd/            # UTSD数据集缓存
│   ├── s3/              # S3格式数据
│   └── standard_datasets/  # 标准数据集
│
├── training/            # 训练模块
│   ├── __init__.py
│   ├── trainer.py       # 预训练训练器
│   └── finetune_trainer.py  # 微调训练器
│
├── utils/               # 工具函数模块
│   ├── __init__.py
│   └── model_utils.py   # 模型工具函数
│
├── scripts/             # 脚本模块
│   ├── __init__.py
│   ├── train.py         # 统一训练入口
│   ├── evaluate.py      # 模型评测脚本
│   ├── run_pretrain_and_eval.sh  # 一键运行脚本（Linux/Mac）
│   ├── run_pretrain_and_eval.bat # 一键运行脚本（Windows）
│   └── prepare_data.py  # 数据准备脚本
│
├── outputs/             # 输出目录（模型和结果）
│
├── README.md            # 本文档
├── requirements.txt     # 依赖
└── LICENSE              # 许可证
```

**注意**：
- `data_processing/` 文件夹包含数据处理相关的**代码模块**（避免与HuggingFace的datasets库冲突）
- `data_cache/` 文件夹存储**实际的数据文件**（下载的数据集、预处理后的数据等）
- `outputs/` 文件夹存储训练输出（模型权重、训练历史、评测结果等）

## 🚀 使用示例

### 预训练小模型（使用S3格式，论文超参数）

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

**注意**: 论文中使用batch size=8192，但需要根据GPU内存调整。可以使用梯度累积来模拟大batch size。

### 微调大模型

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

### 下载UTSD数据集

```bash
python scripts/download_utsd.py --subset UTSD-1G
```

## 📊 输出文件

训练完成后，输出目录包含：

- `best_model/`: 验证集上表现最好的模型
  - `model.pt`: 模型权重
  - `config.json`: 模型配置
  - `optimizer.pt`: 优化器状态
- `final_model/`: 最后一轮的模型
- `training_history.json`: 训练历史数据
- `training_curves.png`: 训练曲线图

## 🔄 工作流程

### 完整预训练流程

1. **下载UTSD数据集**（可选）
   ```bash
   python scripts/download_utsd.py --subset UTSD-1G
   ```

2. **开始预训练**
   ```bash
   python scripts/train.py --mode pretrain --data-source utsd --utsd-subset UTSD-1G
   ```

3. **使用预训练模型进行微调**
   ```bash
   python scripts/train.py --mode finetune --pretrained-model pretrain_outputs/best_model
   ```

### 微调流程

1. **准备本地数据**
   ```bash
   python scripts/prepare_data.py --csv-path <path> --output-dir data
   ```

2. **从HuggingFace模型微调**
   ```bash
   python scripts/train.py --mode finetune --data-source local --data-dir data
   ```

## 🌐 镜像支持

框架自动使用hf-mirror.com镜像，无需额外配置。如果需要切换：

```bash
export HF_ENDPOINT=https://hf-mirror.com  # 使用镜像
export HF_ENDPOINT=https://huggingface.co  # 使用官方
```

## 📝 注意事项

1. **内存使用**: 根据GPU内存调整batch_size
2. **训练时间**: 预训练需要较长时间，建议使用GPU
3. **数据下载**: UTSD数据集较大，首次下载需要时间
4. **模型保存**: 模型会自动保存最佳和最终版本

## 🤝 获取帮助

```bash
python scripts/train.py --help
```

## 📚 相关资源

- [UTSD数据集](https://huggingface.co/datasets/thuml/UTSD)
- [Timer模型](https://huggingface.co/thuml/timer-base-84m)

## 📊 模型评测

### 在标准数据集上评测

```bash
# 使用预训练模型评测
python scripts/evaluate.py \
    --model-path pretrain_outputs/best_model \
    --datasets ETTH1 ECL TRAFFIC WEATHER PEMS03 PEMS04 \
    --lookback 672 \
    --pred-len 96 \
    --batch-size 32 \
    --output-dir evaluation_results

# 使用HuggingFace模型评测
python scripts/evaluate.py \
    --huggingface-model thuml/timer-base-84m \
    --datasets ETTH1 ECL TRAFFIC WEATHER PEMS03 PEMS04 \
    --lookback 672 \
    --pred-len 96 \
    --output-dir evaluation_results

# 注意：必须使用 --datasets（两个短横线），不是 datasets
```

### 评测指标

- **MSE**: 均方误差
- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **MAPE**: 平均绝对百分比误差
- **Direction Acc**: 方向准确率（预测方向是否正确）

评测结果会保存为JSON文件，并打印汇总表格。

## 📄 许可证

请参考LICENSE文件。
