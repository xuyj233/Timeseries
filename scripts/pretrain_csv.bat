@echo off
REM ============================================================================
REM Run pretraining on selected_factors.csv
REM ============================================================================

setlocal enabledelayedexpansion

REM Configuration parameters
set CSV_PATH=data\selected_factors.csv
set MODEL_STRUCTURE=base
set BATCH_SIZE=4
set NUM_EPOCHS=10
set LEARNING_RATE=5e-5
set MIN_LEARNING_RATE=2e-6
set CONTEXT_LENGTH=512
set OUTPUT_DIR=outputs\pretrain_csv

echo ============================================================================
echo Running pretraining on selected_factors.csv (Cryptocurrency Dataset)
echo ============================================================================
echo.

REM Check if CSV file exists
if not exist "%CSV_PATH%" (
    echo [ERROR] CSV file not found: %CSV_PATH%
    exit /b 1
)

echo [INFO] CSV file: %CSV_PATH%
echo [INFO] Model structure: %MODEL_STRUCTURE%
echo [INFO] Batch size: %BATCH_SIZE%
echo [INFO] Number of epochs: %NUM_EPOCHS%
echo [INFO] Learning rate: %LEARNING_RATE% -^> %MIN_LEARNING_RATE%
echo [INFO] Context length: %CONTEXT_LENGTH%
echo [INFO] Output directory: %OUTPUT_DIR%
echo.

REM Create output directories
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist data mkdir data

echo [INFO] Starting pretraining...
python scripts\train.py ^
    --mode pretrain ^
    --data-source csv ^
    --csv-path %CSV_PATH% ^
    --csv-date-col datetime ^
    --model-structure %MODEL_STRUCTURE% ^
    --context-length %CONTEXT_LENGTH% ^
    --batch-size %BATCH_SIZE% ^
    --num-epochs %NUM_EPOCHS% ^
    --learning-rate %LEARNING_RATE% ^
    --min-learning-rate %MIN_LEARNING_RATE% ^
    --scheduler-type cosine ^
    --output-dir %OUTPUT_DIR% ^
    --data-dir data

if errorlevel 1 (
    echo [ERROR] Pretraining failed!
    exit /b 1
)

echo.
echo [SUCCESS] Pretraining completed!
echo [INFO] Best model saved at: %OUTPUT_DIR%\best_model
echo [INFO] Final model saved at: %OUTPUT_DIR%\final_model
echo.

endlocal

