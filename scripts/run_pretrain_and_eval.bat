@echo off
REM ============================================================================
REM TIMER Pretraining and Evaluation Complete Workflow Script (Windows Version)
REM ============================================================================
REM Usage: 
REM   scripts\run_pretrain_and_eval.bat              # Run complete workflow
REM   scripts\run_pretrain_and_eval.bat --skip-pretrain  # Only run evaluation
REM   scripts\run_pretrain_and_eval.bat --skip-eval      # Only run pretraining
REM ============================================================================

setlocal enabledelayedexpansion

REM ============================================================================
REM Configuration Parameters (modify as needed)
REM ============================================================================
set MODEL_STRUCTURE=base
set BATCH_SIZE=4
set NUM_EPOCHS=10
set LEARNING_RATE=5e-5
set MIN_LEARNING_RATE=2e-6
set UTSD_SUBSET=UTSD-1G
set LOOKBACK=672
set PRED_LEN=96
set CONTEXT_LENGTH=512

REM Evaluation dataset list
set EVAL_DATASETS=ETTH1 ECL TRAFFIC WEATHER PEMS03 PEMS04

REM Output directories
set PRETRAIN_OUTPUT_DIR=outputs\pretrain_%MODEL_STRUCTURE%
set EVAL_OUTPUT_DIR=outputs\evaluation

REM ============================================================================
REM Parse command line arguments
REM ============================================================================
set SKIP_PRETRAIN=0
set SKIP_EVAL=0

:parse_args
if "%1"=="" goto end_parse
if /i "%1"=="--skip-pretrain" set SKIP_PRETRAIN=1
if /i "%1"=="--skip-eval" set SKIP_EVAL=1
if /i "%1"=="--help" goto show_help
if /i "%1"=="-h" goto show_help
shift
goto parse_args
:end_parse

REM ============================================================================
REM Show help information
REM ============================================================================
:show_help
echo Usage: %0 [options]
echo.
echo Options:
echo   --skip-pretrain    Skip pretraining step
echo   --skip-eval        Skip evaluation step
echo   --help, -h         Show this help message
echo.
echo Examples:
echo   %0                    # Run complete workflow
echo   %0 --skip-pretrain    # Only run evaluation
echo   %0 --skip-eval        # Only run pretraining
exit /b 0

REM ============================================================================
REM Main workflow starts
REM ============================================================================
echo ============================================================================
echo TIMER Pretraining and Evaluation Complete Workflow
echo ============================================================================
echo.

REM Record start time
set START_TIME=%TIME%

REM Environment check
echo [INFO] Checking Python environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.7+.
    exit /b 1
)

python --version
echo.

REM Check dependencies
echo [INFO] Checking dependencies...
python -c "import torch; import transformers; import datasets; import pandas; import numpy; import matplotlib" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Some dependencies are missing. Installing...
    pip install -q -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed!
        exit /b 1
    )
)

REM Check CUDA
python -c "import torch; print('CUDA available' if torch.cuda.is_available() else 'CUDA not available, will use CPU')"
echo.

echo [SUCCESS] Environment check passed!
echo.

REM Create directories
echo [INFO] Creating output directories...
if not exist "%PRETRAIN_OUTPUT_DIR%" mkdir "%PRETRAIN_OUTPUT_DIR%"
if not exist "%EVAL_OUTPUT_DIR%" mkdir "%EVAL_OUTPUT_DIR%"
if not exist data mkdir data
if not exist data\utsd mkdir data\utsd
if not exist data\s3 mkdir data\s3
if not exist data\standard_datasets mkdir data\standard_datasets
echo [SUCCESS] Directories created
echo.

REM Pretraining
if %SKIP_PRETRAIN%==1 (
    echo [WARNING] Skipping pretraining step (--skip-pretrain)
    echo.
) else (
    echo ============================================================================
    echo Step 1: Pretraining
    echo ============================================================================
    echo.
    echo [INFO] Pretraining configuration:
    echo   Model structure: %MODEL_STRUCTURE%
    echo   Batch size: %BATCH_SIZE%
    echo   Number of epochs: %NUM_EPOCHS%
    echo   Learning rate: %LEARNING_RATE% -^> %MIN_LEARNING_RATE%
    echo   Dataset: %UTSD_SUBSET%
    echo   Context length: %CONTEXT_LENGTH%
    echo   Output directory: %PRETRAIN_OUTPUT_DIR%
    echo.
    
    echo [INFO] Starting pretraining...
    python scripts\train.py ^
        --mode pretrain ^
        --data-source utsd-s3 ^
        --utsd-subset %UTSD_SUBSET% ^
        --model-structure %MODEL_STRUCTURE% ^
        --context-length %CONTEXT_LENGTH% ^
        --batch-size %BATCH_SIZE% ^
        --num-epochs %NUM_EPOCHS% ^
        --learning-rate %LEARNING_RATE% ^
        --min-learning-rate %MIN_LEARNING_RATE% ^
        --scheduler-type cosine ^
        --output-dir %PRETRAIN_OUTPUT_DIR% ^
        --data-dir data
    
    if errorlevel 1 (
        echo [ERROR] Pretraining failed!
        exit /b 1
    )
    
    echo.
    echo [SUCCESS] Pretraining completed!
    echo [INFO] Best model saved at: %PRETRAIN_OUTPUT_DIR%\best_model
    echo [INFO] Final model saved at: %PRETRAIN_OUTPUT_DIR%\final_model
    echo.
)

REM Evaluation
if %SKIP_EVAL%==1 (
    echo [WARNING] Skipping evaluation step (--skip-eval)
    echo.
) else (
    echo ============================================================================
    echo Step 2: Model Evaluation
    echo ============================================================================
    echo.
    
    set MODEL_PATH=%PRETRAIN_OUTPUT_DIR%\best_model
    
    if not exist "!MODEL_PATH!" (
        echo [ERROR] Model file not found: !MODEL_PATH!
        echo [ERROR] Please run pretraining first or specify correct model path
        exit /b 1
    )
    
    echo [INFO] Evaluation configuration:
    echo   Model path: !MODEL_PATH!
    echo   Evaluation datasets: %EVAL_DATASETS%
    echo   Lookback window: %LOOKBACK%
    echo   Prediction length: %PRED_LEN%
    echo   Output directory: %EVAL_OUTPUT_DIR%
    echo.
    
    echo [INFO] Starting evaluation...
    python scripts\evaluate.py ^
        --model-path "!MODEL_PATH!" ^
        --datasets %EVAL_DATASETS% ^
        --lookback %LOOKBACK% ^
        --pred-len %PRED_LEN% ^
        --batch-size 32 ^
        --output-dir %EVAL_OUTPUT_DIR% ^
        --data-dir data\standard_datasets
    
    if errorlevel 1 (
        echo [ERROR] Evaluation failed!
        exit /b 1
    )
    
    echo.
    echo [SUCCESS] Evaluation completed!
    echo [INFO] Evaluation results saved at: %EVAL_OUTPUT_DIR%\evaluation_results.json
    echo.
)

REM Final summary
echo ============================================================================
echo All steps completed!
echo ============================================================================
echo.

if %SKIP_PRETRAIN%==0 (
    echo [INFO] Pretrained model: %PRETRAIN_OUTPUT_DIR%\best_model
)

if %SKIP_EVAL%==0 (
    echo [INFO] Evaluation results: %EVAL_OUTPUT_DIR%\evaluation_results.json
)

echo.
echo [SUCCESS] Workflow execution completed!

endlocal
