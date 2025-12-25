#!/bin/bash
# ============================================================================
# TIMER Pretraining and Evaluation Complete Workflow Script
# ============================================================================
# Usage: 
#   bash scripts/run_pretrain_and_eval.sh              # Run complete workflow
#   bash scripts/run_pretrain_and_eval.sh --skip-pretrain  # Only run evaluation
#   bash scripts/run_pretrain_and_eval.sh --skip-eval      # Only run pretraining
# ============================================================================

set -e  # Exit immediately if a command exits with a non-zero status

# ============================================================================
# Configuration Parameters (modify as needed)
# ============================================================================
MODEL_STRUCTURE="base"        # Model structure: tiny, small, base, large
BATCH_SIZE=4                  # Batch size (adjust based on GPU memory)
NUM_EPOCHS=10                 # Number of training epochs
LEARNING_RATE=5e-5            # Base learning rate (paper default)
MIN_LEARNING_RATE=2e-6        # Minimum learning rate (paper default)
UTSD_SUBSET="UTSD-1G"         # UTSD subset: UTSD-1G, UTSD-2G, UTSD-4G, UTSD-12G
LOOKBACK=672                  # Lookback window length
PRED_LEN=96                   # Prediction length
CONTEXT_LENGTH=512            # S3 format context length

# Evaluation dataset list
EVAL_DATASETS=("ETTH1" "ECL" "TRAFFIC" "WEATHER" "PEMS03" "PEMS04")

# Output directories
PRETRAIN_OUTPUT_DIR="outputs/pretrain_${MODEL_STRUCTURE}"
EVAL_OUTPUT_DIR="outputs/evaluation"

# ============================================================================
# Color output functions
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
# Environment check function
# ============================================================================
check_environment() {
    print_info "Checking Python environment..."
    
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please install Python 3.7+."
        exit 1
    fi
    
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    print_info "Python version: ${PYTHON_VERSION}"
    
    # Check required packages
    print_info "Checking dependencies..."
    python -c "import torch; import transformers; import datasets; import pandas; import numpy; import matplotlib" 2>/dev/null
    
    if [ $? -ne 0 ]; then
        print_warning "Some dependencies are missing. Installing..."
        pip install -q -r requirements.txt
        if [ $? -ne 0 ]; then
            print_error "Dependency installation failed!"
            exit 1
        fi
    fi
    
    # Check CUDA (if available)
    python -c "import torch; print('CUDA available' if torch.cuda.is_available() else 'CUDA not available, will use CPU')" 2>/dev/null
    
    print_success "Environment check passed!"
}

# ============================================================================
# Create necessary directories
# ============================================================================
create_directories() {
    print_info "Creating output directories..."
    
    mkdir -p "${PRETRAIN_OUTPUT_DIR}"
    mkdir -p "${EVAL_OUTPUT_DIR}"
    mkdir -p data
    mkdir -p data/utsd
    mkdir -p data/s3
    mkdir -p data/standard_datasets
    
    print_success "Directories created"
}

# ============================================================================
# Pretraining function
# ============================================================================
pretrain() {
    print_header "Step 1: Pretraining"
    
    print_info "Pretraining configuration:"
    print_info "  Model structure: ${MODEL_STRUCTURE}"
    print_info "  Batch size: ${BATCH_SIZE}"
    print_info "  Number of epochs: ${NUM_EPOCHS}"
    print_info "  Learning rate: ${LEARNING_RATE} -> ${MIN_LEARNING_RATE}"
    print_info "  Dataset: ${UTSD_SUBSET}"
    print_info "  Context length: ${CONTEXT_LENGTH}"
    print_info "  Output directory: ${PRETRAIN_OUTPUT_DIR}"
    echo ""
    
    print_info "Starting pretraining..."
    
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
        --data-dir data
    
    if [ $? -eq 0 ]; then
        print_success "Pretraining completed!"
        print_info "Best model saved at: ${PRETRAIN_OUTPUT_DIR}/best_model"
        print_info "Final model saved at: ${PRETRAIN_OUTPUT_DIR}/final_model"
        echo ""
    else
        print_error "Pretraining failed!"
        exit 1
    fi
}

# ============================================================================
# Evaluation function
# ============================================================================
evaluate() {
    print_header "Step 2: Model Evaluation"
    
    MODEL_PATH="${PRETRAIN_OUTPUT_DIR}/best_model"
    
    # Check if model exists
    if [ ! -d "${MODEL_PATH}" ]; then
        print_error "Model file not found: ${MODEL_PATH}"
        print_error "Please run pretraining first or specify correct model path"
        exit 1
    fi
    
    print_info "Evaluation configuration:"
    print_info "  Model path: ${MODEL_PATH}"
    print_info "  Evaluation datasets: ${EVAL_DATASETS[*]}"
    print_info "  Lookback window: ${LOOKBACK}"
    print_info "  Prediction length: ${PRED_LEN}"
    print_info "  Output directory: ${EVAL_OUTPUT_DIR}"
    echo ""
    
    print_info "Starting evaluation..."
    
    python scripts/evaluate.py \
        --model-path "${MODEL_PATH}" \
        --datasets ${EVAL_DATASETS[@]} \
        --lookback ${LOOKBACK} \
        --pred-len ${PRED_LEN} \
        --batch-size 32 \
        --output-dir "${EVAL_OUTPUT_DIR}" \
        --data-dir data/standard_datasets
    
    if [ $? -eq 0 ]; then
        print_success "Evaluation completed!"
        print_info "Evaluation results saved at: ${EVAL_OUTPUT_DIR}/evaluation_results.json"
        echo ""
        
        # Display evaluation results summary
        if [ -f "${EVAL_OUTPUT_DIR}/evaluation_results.json" ]; then
            print_info "Evaluation results summary:"
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
        print_error "Evaluation failed!"
        exit 1
    fi
}

# ============================================================================
# Main function
# ============================================================================
main() {
    print_header "TIMER Pretraining and Evaluation Complete Workflow"
    echo ""
    
    # Parse command line arguments
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
                echo "Usage: bash $0 [options]"
                echo ""
                echo "Options:"
                echo "  --skip-pretrain    Skip pretraining step"
                echo "  --skip-eval        Skip evaluation step"
                echo "  --help, -h         Show this help message"
                echo ""
                echo "Examples:"
                echo "  bash $0                    # Run complete workflow"
                echo "  bash $0 --skip-pretrain    # Only run evaluation"
                echo "  bash $0 --skip-eval        # Only run pretraining"
                exit 0
                ;;
            *)
                ;;
        esac
    done
    
    # Record start time
    START_TIME=$(date +%s)
    
    # Environment check
    check_environment
    echo ""
    
    # Create directories
    create_directories
    echo ""
    
    # Execute pretraining
    if [ "$SKIP_PRETRAIN" = false ]; then
        pretrain
    else
        print_warning "Skipping pretraining step (--skip-pretrain)"
        echo ""
    fi
    
    # Execute evaluation
    if [ "$SKIP_EVAL" = false ]; then
        evaluate
    else
        print_warning "Skipping evaluation step (--skip-eval)"
        echo ""
    fi
    
    # Calculate total time
    END_TIME=$(date +%s)
    ELAPSED_TIME=$((END_TIME - START_TIME))
    HOURS=$((ELAPSED_TIME / 3600))
    MINUTES=$(((ELAPSED_TIME % 3600) / 60))
    SECONDS=$((ELAPSED_TIME % 60))
    
    # Final summary
    print_header "All steps completed!"
    print_success "Total time: ${HOURS} hours ${MINUTES} minutes ${SECONDS} seconds"
    echo ""
    
    if [ "$SKIP_PRETRAIN" = false ]; then
        print_info "Pretrained model: ${PRETRAIN_OUTPUT_DIR}/best_model"
    fi
    
    if [ "$SKIP_EVAL" = false ]; then
        print_info "Evaluation results: ${EVAL_OUTPUT_DIR}/evaluation_results.json"
    fi
    
    echo ""
    print_success "Workflow execution completed!"
}

# ============================================================================
# Run main function
# ============================================================================
main "$@"
