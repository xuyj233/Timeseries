@echo off
REM ============================================================================
REM TIMER 预训练和评测完整流程脚本 (Windows版本)
REM ============================================================================
REM 用法: 
REM   scripts\run_pretrain_and_eval.bat              # 运行完整流程
REM   scripts\run_pretrain_and_eval.bat --skip-pretrain  # 只运行评测
REM   scripts\run_pretrain_and_eval.bat --skip-eval      # 只运行预训练
REM ============================================================================

setlocal enabledelayedexpansion

REM ============================================================================
REM 配置参数（可根据需要修改）
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

REM 评测数据集列表
set EVAL_DATASETS=ETTH1 ECL TRAFFIC WEATHER PEMS03 PEMS04

REM 输出目录
set PRETRAIN_OUTPUT_DIR=outputs\pretrain_%MODEL_STRUCTURE%
set EVAL_OUTPUT_DIR=outputs\evaluation

REM ============================================================================
REM 解析命令行参数
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
REM 显示帮助信息
REM ============================================================================
:show_help
echo 用法: %0 [选项]
echo.
echo 选项:
echo   --skip-pretrain    跳过预训练步骤
echo   --skip-eval        跳过评测步骤
echo   --help, -h         显示此帮助信息
echo.
echo 示例:
echo   %0                    # 运行完整流程
echo   %0 --skip-pretrain    # 只运行评测
echo   %0 --skip-eval        # 只运行预训练
exit /b 0

REM ============================================================================
REM 主流程开始
REM ============================================================================
echo ============================================================================
echo TIMER 预训练和评测完整流程
echo ============================================================================
echo.

REM 记录开始时间
set START_TIME=%TIME%

REM 环境检查
echo [INFO] 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 未找到Python。请安装Python 3.7+。
    exit /b 1
)

python --version
echo.

REM 检查依赖包
echo [INFO] 检查依赖包...
python -c "import torch; import transformers; import datasets; import pandas; import numpy; import matplotlib" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] 某些依赖包缺失。正在安装...
    pip install -q -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] 依赖安装失败！
        exit /b 1
    )
)

REM 检查CUDA
python -c "import torch; print('CUDA可用' if torch.cuda.is_available() else 'CUDA不可用，将使用CPU')"
echo.

echo [SUCCESS] 环境检查通过！
echo.

REM 创建目录
echo [INFO] 创建输出目录...
if not exist "%PRETRAIN_OUTPUT_DIR%" mkdir "%PRETRAIN_OUTPUT_DIR%"
if not exist "%EVAL_OUTPUT_DIR%" mkdir "%EVAL_OUTPUT_DIR%"
if not exist data_cache mkdir data_cache
if not exist data_cache\utsd mkdir data_cache\utsd
if not exist data_cache\s3 mkdir data_cache\s3
if not exist data_cache\standard_datasets mkdir data_cache\standard_datasets
echo [SUCCESS] 目录创建完成
echo.

REM 预训练
if %SKIP_PRETRAIN%==1 (
    echo [WARNING] 跳过预训练步骤 (--skip-pretrain)
    echo.
) else (
    echo ============================================================================
    echo 步骤 1: 预训练 (Pretraining)
    echo ============================================================================
    echo.
    echo [INFO] 预训练配置:
    echo   模型结构: %MODEL_STRUCTURE%
    echo   批次大小: %BATCH_SIZE%
    echo   训练轮数: %NUM_EPOCHS%
    echo   学习率: %LEARNING_RATE% -^> %MIN_LEARNING_RATE%
    echo   数据集: %UTSD_SUBSET%
    echo   上下文长度: %CONTEXT_LENGTH%
    echo   输出目录: %PRETRAIN_OUTPUT_DIR%
    echo.
    
    echo [INFO] 开始预训练...
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
        --data-dir data_cache
    
    if errorlevel 1 (
        echo [ERROR] 预训练失败！
        exit /b 1
    )
    
    echo.
    echo [SUCCESS] 预训练完成！
    echo [INFO] 最佳模型保存在: %PRETRAIN_OUTPUT_DIR%\best_model
    echo [INFO] 最终模型保存在: %PRETRAIN_OUTPUT_DIR%\final_model
    echo.
)

REM 评测
if %SKIP_EVAL%==1 (
    echo [WARNING] 跳过评测步骤 (--skip-eval)
    echo.
) else (
    echo ============================================================================
    echo 步骤 2: 模型评测 (Evaluation)
    echo ============================================================================
    echo.
    
    set MODEL_PATH=%PRETRAIN_OUTPUT_DIR%\best_model
    
    if not exist "!MODEL_PATH!" (
        echo [ERROR] 未找到模型文件: !MODEL_PATH!
        echo [ERROR] 请先运行预训练或指定正确的模型路径
        exit /b 1
    )
    
    echo [INFO] 评测配置:
    echo   模型路径: !MODEL_PATH!
    echo   评测数据集: %EVAL_DATASETS%
    echo   历史窗口: %LOOKBACK%
    echo   预测长度: %PRED_LEN%
    echo   输出目录: %EVAL_OUTPUT_DIR%
    echo.
    
    echo [INFO] 开始评测...
    python scripts\evaluate.py ^
        --model-path "!MODEL_PATH!" ^
        --datasets %EVAL_DATASETS% ^
        --lookback %LOOKBACK% ^
        --pred-len %PRED_LEN% ^
        --batch-size 32 ^
        --output-dir %EVAL_OUTPUT_DIR% ^
        --data-dir data_cache\standard_datasets
    
    if errorlevel 1 (
        echo [ERROR] 评测失败！
        exit /b 1
    )
    
    echo.
    echo [SUCCESS] 评测完成！
    echo [INFO] 评测结果保存在: %EVAL_OUTPUT_DIR%\evaluation_results.json
    echo.
)

REM 最终总结
echo ============================================================================
echo 所有步骤完成！
echo ============================================================================
echo.

if %SKIP_PRETRAIN%==0 (
    echo [INFO] 预训练模型: %PRETRAIN_OUTPUT_DIR%\best_model
)

if %SKIP_EVAL%==0 (
    echo [INFO] 评测结果: %EVAL_OUTPUT_DIR%\evaluation_results.json
)

echo.
echo [SUCCESS] 流程执行完毕！

endlocal




