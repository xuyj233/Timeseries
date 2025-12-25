#!/bin/bash
# ============================================================================
# TIMER 预训练和评测完整流程脚本
# ============================================================================
# 用法: 
#   bash scripts/run_pretrain_and_eval.sh              # 运行完整流程
#   bash scripts/run_pretrain_and_eval.sh --skip-pretrain  # 只运行评测
#   bash scripts/run_pretrain_and_eval.sh --skip-eval      # 只运行预训练
# ============================================================================

set -e  # 遇到错误立即退出

# ============================================================================
# 配置参数（可根据需要修改）
# ============================================================================
MODEL_STRUCTURE="base"        # 模型结构: tiny, small, base, large
BATCH_SIZE=4                  # 批次大小（根据GPU内存调整）
NUM_EPOCHS=10                 # 训练轮数
LEARNING_RATE=5e-5            # 基础学习率（论文默认）
MIN_LEARNING_RATE=2e-6        # 最小学习率（论文默认）
UTSD_SUBSET="UTSD-1G"         # UTSD子集: UTSD-1G, UTSD-2G, UTSD-4G, UTSD-12G
LOOKBACK=672                  # 历史窗口长度
PRED_LEN=96                   # 预测长度
CONTEXT_LENGTH=512            # S3格式上下文长度

# 评测数据集列表
EVAL_DATASETS=("ETTH1" "ECL" "TRAFFIC" "WEATHER" "PEMS03" "PEMS04")

# 输出目录
PRETRAIN_OUTPUT_DIR="outputs/pretrain_${MODEL_STRUCTURE}"
EVAL_OUTPUT_DIR="outputs/evaluation"

# ============================================================================
# 颜色输出函数
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${CYAN}============================================================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}============================================================================${NC}"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# 环境检查函数
# ============================================================================
check_environment() {
    print_info "检查Python环境..."
    
    if ! command -v python &> /dev/null; then
        print_error "未找到Python。请安装Python 3.7+。"
        exit 1
    fi
    
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    print_info "Python版本: ${PYTHON_VERSION}"
    
    # 检查必要的包
    print_info "检查依赖包..."
    python -c "import torch; import transformers; import datasets; import pandas; import numpy; import matplotlib" 2>/dev/null
    
    if [ $? -ne 0 ]; then
        print_warning "某些依赖包缺失。正在安装..."
        pip install -q -r requirements.txt
        if [ $? -ne 0 ]; then
            print_error "依赖安装失败！"
            exit 1
        fi
    fi
    
    # 检查CUDA（如果可用）
    python -c "import torch; print('CUDA可用' if torch.cuda.is_available() else 'CUDA不可用，将使用CPU')" 2>/dev/null
    
    print_success "环境检查通过！"
}

# ============================================================================
# 创建必要的目录
# ============================================================================
create_directories() {
    print_info "创建输出目录..."
    
    mkdir -p "${PRETRAIN_OUTPUT_DIR}"
    mkdir -p "${EVAL_OUTPUT_DIR}"
    mkdir -p data_cache
    mkdir -p data_cache/utsd
    mkdir -p data_cache/s3
    mkdir -p data_cache/standard_datasets
    
    print_success "目录创建完成"
}

# ============================================================================
# 预训练函数
# ============================================================================
pretrain() {
    print_header "步骤 1: 预训练 (Pretraining)"
    
    print_info "预训练配置:"
    print_info "  模型结构: ${MODEL_STRUCTURE}"
    print_info "  批次大小: ${BATCH_SIZE}"
    print_info "  训练轮数: ${NUM_EPOCHS}"
    print_info "  学习率: ${LEARNING_RATE} -> ${MIN_LEARNING_RATE}"
    print_info "  数据集: ${UTSD_SUBSET}"
    print_info "  上下文长度: ${CONTEXT_LENGTH}"
    print_info "  输出目录: ${PRETRAIN_OUTPUT_DIR}"
    echo ""
    
    print_info "开始预训练..."
    
    python scripts/train.py \
        --mode pretrain \
        --data-source utsd-s3 \
        --utsd-subset "${UTSD_SUBSET}" \
        --model-structure "${MODEL_STRUCTURE}" \
        --context-length "${CONTEXT_LENGTH}" \
        --batch-size "${BATCH_SIZE}" \
        --num-epochs "${NUM_EPOCHS}" \
        --learning-rate "${LEARNING_RATE}" \
        --min-learning-rate "${MIN_LEARNING_RATE}" \
        --scheduler-type cosine \
        --output-dir "${PRETRAIN_OUTPUT_DIR}" \
        --data-dir data_cache
    
    if [ $? -eq 0 ]; then
        print_success "预训练完成！"
        print_info "最佳模型保存在: ${PRETRAIN_OUTPUT_DIR}/best_model"
        print_info "最终模型保存在: ${PRETRAIN_OUTPUT_DIR}/final_model"
        echo ""
    else
        print_error "预训练失败！"
        exit 1
    fi
}

# ============================================================================
# 评测函数
# ============================================================================
evaluate() {
    print_header "步骤 2: 模型评测 (Evaluation)"
    
    MODEL_PATH="${PRETRAIN_OUTPUT_DIR}/best_model"
    
    # 检查模型是否存在
    if [ ! -d "${MODEL_PATH}" ]; then
        print_error "未找到模型文件: ${MODEL_PATH}"
        print_error "请先运行预训练或指定正确的模型路径"
        exit 1
    fi
    
    print_info "评测配置:"
    print_info "  模型路径: ${MODEL_PATH}"
    print_info "  评测数据集: ${EVAL_DATASETS[*]}"
    print_info "  历史窗口: ${LOOKBACK}"
    print_info "  预测长度: ${PRED_LEN}"
    print_info "  输出目录: ${EVAL_OUTPUT_DIR}"
    echo ""
    
    print_info "开始评测..."
    
    python scripts/evaluate.py \
        --model-path "${MODEL_PATH}" \
        --datasets ${EVAL_DATASETS[@]} \
        --lookback ${LOOKBACK} \
        --pred-len ${PRED_LEN} \
        --batch-size 32 \
        --output-dir "${EVAL_OUTPUT_DIR}" \
        --data-dir data_cache/standard_datasets
    
    if [ $? -eq 0 ]; then
        print_success "评测完成！"
        print_info "评测结果保存在: ${EVAL_OUTPUT_DIR}/evaluation_results.json"
        echo ""
        
        # 显示评测结果摘要
        if [ -f "${EVAL_OUTPUT_DIR}/evaluation_results.json" ]; then
            print_info "评测结果摘要:"
            python -c "
import json
try:
    with open('${EVAL_OUTPUT_DIR}/evaluation_results.json', 'r') as f:
        results = json.load(f)
    for dataset, metrics in results.items():
        if 'error' not in metrics:
            print(f'  {dataset}: MSE={metrics[\"MSE\"]:.6f}, MAE={metrics[\"MAE\"]:.6f}')
        else:
            print(f'  {dataset}: Error')
except:
    pass
"
        fi
    else
        print_error "评测失败！"
        exit 1
    fi
}

# ============================================================================
# 主函数
# ============================================================================
main() {
    print_header "TIMER 预训练和评测完整流程"
    echo ""
    
    # 解析命令行参数
    SKIP_PRETRAIN=false
    SKIP_EVAL=false
    
    for arg in "$@"; do
        case $arg in
            --skip-pretrain)
                SKIP_PRETRAIN=true
                shift
                ;;
            --skip-eval)
                SKIP_EVAL=true
                shift
                ;;
            --help|-h)
                echo "用法: bash $0 [选项]"
                echo ""
                echo "选项:"
                echo "  --skip-pretrain    跳过预训练步骤"
                echo "  --skip-eval        跳过评测步骤"
                echo "  --help, -h         显示此帮助信息"
                echo ""
                echo "示例:"
                echo "  bash $0                    # 运行完整流程"
                echo "  bash $0 --skip-pretrain    # 只运行评测"
                echo "  bash $0 --skip-eval        # 只运行预训练"
                exit 0
                ;;
            *)
                ;;
        esac
    done
    
    # 记录开始时间
    START_TIME=$(date +%s)
    
    # 环境检查
    check_environment
    echo ""
    
    # 创建目录
    create_directories
    echo ""
    
    # 执行预训练
    if [ "$SKIP_PRETRAIN" = false ]; then
        pretrain
    else
        print_warning "跳过预训练步骤 (--skip-pretrain)"
        echo ""
    fi
    
    # 执行评测
    if [ "$SKIP_EVAL" = false ]; then
        evaluate
    else
        print_warning "跳过评测步骤 (--skip-eval)"
        echo ""
    fi
    
    # 计算总用时
    END_TIME=$(date +%s)
    ELAPSED_TIME=$((END_TIME - START_TIME))
    HOURS=$((ELAPSED_TIME / 3600))
    MINUTES=$(((ELAPSED_TIME % 3600) / 60))
    SECONDS=$((ELAPSED_TIME % 60))
    
    # 最终总结
    print_header "所有步骤完成！"
    print_success "总用时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
    echo ""
    
    if [ "$SKIP_PRETRAIN" = false ]; then
        print_info "预训练模型: ${PRETRAIN_OUTPUT_DIR}/best_model"
    fi
    
    if [ "$SKIP_EVAL" = false ]; then
        print_info "评测结果: ${EVAL_OUTPUT_DIR}/evaluation_results.json"
    fi
    
    echo ""
    print_success "流程执行完毕！"
}

# ============================================================================
# 运行主函数
# ============================================================================
main "$@"
