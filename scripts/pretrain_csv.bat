@echo off
REM ============================================================================
REM 在 selected_factors.csv 上运行预训练
REM ============================================================================

setlocal enabledelayedexpansion

REM 配置参数
set CSV_PATH=selected_factors.csv
set MODEL_STRUCTURE=base
set BATCH_SIZE=4
set NUM_EPOCHS=10
set LEARNING_RATE=5e-5
set MIN_LEARNING_RATE=2e-6
set CONTEXT_LENGTH=512
set OUTPUT_DIR=outputs\pretrain_csv

echo ============================================================================
echo 在 selected_factors.csv 上运行预训练
echo ============================================================================
echo.

REM 检查CSV文件是否存在
if not exist "%CSV_PATH%" (
    echo [ERROR] CSV文件不存在: %CSV_PATH%
    exit /b 1
)

echo [INFO] CSV文件: %CSV_PATH%
echo [INFO] 模型结构: %MODEL_STRUCTURE%
echo [INFO] 批次大小: %BATCH_SIZE%
echo [INFO] 训练轮数: %NUM_EPOCHS%
echo [INFO] 学习率: %LEARNING_RATE% -^> %MIN_LEARNING_RATE%
echo [INFO] 上下文长度: %CONTEXT_LENGTH%
echo [INFO] 输出目录: %OUTPUT_DIR%
echo.

REM 创建输出目录
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist data_cache mkdir data_cache

echo [INFO] 开始预训练...
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
    --data-dir data_cache

if errorlevel 1 (
    echo [ERROR] 预训练失败！
    exit /b 1
)

echo.
echo [SUCCESS] 预训练完成！
echo [INFO] 最佳模型保存在: %OUTPUT_DIR%\best_model
echo [INFO] 最终模型保存在: %OUTPUT_DIR%\final_model
echo.

endlocal

