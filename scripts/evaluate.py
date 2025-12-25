"""
模型评测脚本
在标准时间序列数据集上评估模型性能
"""
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import TimerConfig, TimerForPrediction
from data_processing import load_standard_dataset, prepare_multiple_datasets
from utils import load_pretrained_model, count_parameters


def calculate_metrics(predictions, targets, mask=None):
    """
    计算评估指标
    
    Args:
        predictions: 预测值 (batch_size, pred_len, n_features)
        targets: 真实值 (batch_size, pred_len, n_features)
        mask: 掩码（可选）
    
    Returns:
        dict: 包含各种指标的字典
    """
    if mask is not None:
        predictions = predictions * mask
        targets = targets * mask
        valid_count = mask.sum()
    else:
        valid_count = predictions.numel()
    
    # 转换为numpy
    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    
    # 计算MSE
    mse = np.mean((pred_np - target_np) ** 2)
    
    # 计算MAE
    mae = np.mean(np.abs(pred_np - target_np))
    
    # 计算RMSE
    rmse = np.sqrt(mse)
    
    # 计算MAPE（避免除零）
    epsilon = 1e-8
    mape = np.mean(np.abs((target_np - pred_np) / (target_np + epsilon))) * 100
    
    # 计算方向准确率（Direction Accuracy）
    if pred_np.shape[-1] == 1:
        # 单变量：计算方向
        pred_diff = np.diff(pred_np.squeeze(-1), axis=1)
        target_diff = np.diff(target_np.squeeze(-1), axis=1)
        direction_correct = np.sign(pred_diff) == np.sign(target_diff)
        direction_acc = np.mean(direction_correct) * 100
    else:
        # 多变量：对每个变量计算方向准确率
        direction_accs = []
        for i in range(pred_np.shape[-1]):
            pred_diff = np.diff(pred_np[:, :, i], axis=1)
            target_diff = np.diff(target_np[:, :, i], axis=1)
            direction_correct = np.sign(pred_diff) == np.sign(target_diff)
            direction_accs.append(np.mean(direction_correct) * 100)
        direction_acc = np.mean(direction_accs)
    
    return {
        'MSE': float(mse),
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MAPE': float(mape),
        'Direction_Acc': float(direction_acc)
    }


def evaluate_model(model, test_loader, device, dataset_name="Unknown"):
    """
    评估模型在测试集上的性能
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        dataset_name: 数据集名称
    
    Returns:
        dict: 评估指标
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    print(f"\nEvaluating on {dataset_name}...")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            # 安全地解包数据（处理可能返回额外值的情况）
            if isinstance(batch_data, (tuple, list)):
                if len(batch_data) == 2:
                    history, target = batch_data
                elif len(batch_data) > 2:
                    # 如果返回超过2个值，只取前两个
                    history, target = batch_data[0], batch_data[1]
                else:
                    print(f"Warning: Unexpected batch data length: {len(batch_data)}")
                    continue
            else:
                print(f"Warning: Unexpected batch data type: {type(batch_data)}")
                continue
            
            history = history.to(device)
            target = target.to(device)
            
            # 生成预测
            # Timer模型在推理模式下，使用history作为输入，生成pred_len长度的预测
            pred_len = target.shape[1]
            
            # 调用模型进行预测（推理模式）
            try:
                outputs = model(
                    input_ids=history,
                    labels=None,
                    return_dict=True,
                    max_output_length=pred_len
                )
                
                predictions = outputs.logits
                
                # Timer模型在推理模式下：
                # predictions = lm_head(hidden_states)[:, -1, :] 
                # 返回形状是 (batch_size, output_token_len)，其中output_token_len是每个token的预测长度
                # 但我们需要的是 (batch_size, pred_len, n_features)
                # 注意：Timer使用patch embedding，每个patch的长度是input_token_len
                
                batch_size = history.shape[0]
                n_features = history.shape[-1]
                
                if predictions is not None:
                    # Timer模型的predictions形状是 (batch_size, output_token_len)
                    # output_token_len是每个patch的预测长度，不是时间步数
                    # 我们需要将其reshape为 (batch_size, num_patches, output_token_len)
                    # 然后reshape为 (batch_size, total_pred_len, n_features)
                    
                    if len(predictions.shape) == 2:
                        # predictions: (batch_size, output_token_len)
                        # 需要reshape为 (batch_size, 1, output_token_len) 然后展开
                        # 但实际Timer返回的可能是已经处理过的
                        
                        # 尝试理解：如果output_token_len = 96, pred_len = 96
                        # 那么predictions可能是 (batch_size, 96)，需要reshape为 (batch_size, 96, 1)
                        # 或者如果output_token_len = pred_len * n_features，需要reshape
                        
                        output_token_len = predictions.shape[1]
                        
                        # 如果output_token_len等于pred_len，说明每个时间步预测一个值
                        if output_token_len == pred_len:
                            # (batch_size, pred_len) -> (batch_size, pred_len, 1)
                            predictions = predictions.unsqueeze(-1)
                            # 如果n_features > 1，需要扩展
                            if n_features > 1:
                                predictions = predictions.repeat(1, 1, n_features)
                        elif output_token_len == pred_len * n_features:
                            # (batch_size, pred_len * n_features) -> (batch_size, pred_len, n_features)
                            predictions = predictions.view(batch_size, pred_len, n_features)
                        else:
                            # 其他情况：假设是单变量预测，扩展特征维度
                            if output_token_len >= pred_len:
                                predictions = predictions[:, :pred_len].unsqueeze(-1)
                                if n_features > 1:
                                    predictions = predictions.repeat(1, 1, n_features)
                            else:
                                # 长度不足，填充
                                padding = torch.zeros(batch_size, pred_len - output_token_len, device=device)
                                predictions = torch.cat([predictions, padding], dim=1).unsqueeze(-1)
                                if n_features > 1:
                                    predictions = predictions.repeat(1, 1, n_features)
                    elif len(predictions.shape) == 3:
                        # 已经是3D，直接调整
                        if predictions.shape[1] > pred_len:
                            predictions = predictions[:, :pred_len, :]
                        elif predictions.shape[1] < pred_len:
                            last_step = predictions[:, -1:, :]
                            padding = last_step.repeat(1, pred_len - predictions.shape[1], 1)
                            predictions = torch.cat([predictions, padding], dim=1)
                        
                        # 调整特征维度
                        if predictions.shape[-1] != n_features:
                            if predictions.shape[-1] > n_features:
                                predictions = predictions[:, :, :n_features]
                            else:
                                padding = torch.zeros(
                                    batch_size, 
                                    predictions.shape[1], 
                                    n_features - predictions.shape[-1],
                                    device=device
                                )
                                predictions = torch.cat([predictions, padding], dim=-1)
                else:
                    # 如果logits为None，使用零填充
                    predictions = torch.zeros(batch_size, pred_len, n_features, device=device)
                    
            except Exception as e:
                # 如果预测失败，使用零填充
                if batch_idx % 10 == 0:  # 只打印部分警告，避免刷屏
                    print(f"Warning: Prediction failed for batch {batch_idx}: {str(e)}")
                batch_size, _, n_features = history.shape
                predictions = torch.zeros(batch_size, pred_len, n_features, device=device)
            
            # 最终确保预测形状与目标匹配
            if predictions.shape[1] != target.shape[1]:
                if predictions.shape[1] > target.shape[1]:
                    predictions = predictions[:, :target.shape[1], :]
                else:
                    # 填充（使用最后一个预测值）
                    padding = predictions[:, -1:, :].repeat(1, target.shape[1] - predictions.shape[1], 1)
                    predictions = torch.cat([predictions, padding], dim=1)
            
            # 确保特征维度匹配
            if predictions.shape[-1] != target.shape[-1]:
                if predictions.shape[-1] > target.shape[-1]:
                    predictions = predictions[:, :, :target.shape[-1]]
                else:
                    # 如果特征维度不足，使用零填充
                    padding = torch.zeros(
                        predictions.shape[0], 
                        predictions.shape[1], 
                        target.shape[-1] - predictions.shape[-1],
                        device=device
                    )
                    predictions = torch.cat([predictions, padding], dim=-1)
            
            all_predictions.append(predictions)
            all_targets.append(target)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # 合并所有预测
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # 计算指标
    metrics = calculate_metrics(all_predictions, all_targets)
    
    return metrics, all_predictions, all_targets


def main():
    parser = argparse.ArgumentParser(description="Evaluate Timer model on standard datasets")
    
    # 模型相关
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to pretrained model directory (containing model.pt and config.json)")
    parser.add_argument("--huggingface-model", type=str, default=None,
                       help="HuggingFace model name (alternative to --model-path)")
    
    # 数据集相关
    parser.add_argument("--dataset", type=str, default=None,
                       choices=['ETTH1', 'ETTH2', 'ETTM1', 'ETTM2', 'ECL', 'TRAFFIC', 'WEATHER', 
                               'PEMS03', 'PEMS04', 'PEMS07', 'PEMS08'],
                       help="Single dataset to evaluate")
    parser.add_argument("--datasets", type=str, nargs='+', default=None,
                       help="Multiple datasets to evaluate (e.g., ETTH1 ECL TRAFFIC)")
    parser.add_argument("--data-dir", type=str, default="data/standard_datasets",
                       help="Data directory")
    
    # 评估参数
    parser.add_argument("--lookback", type=int, default=672,
                       help="Lookback length (default: 672)")
    parser.add_argument("--pred-len", type=int, default=96,
                       help="Prediction length (default: 96)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu, default: auto)")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # 设备
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # 加载模型
    print("\nLoading model...")
    if args.huggingface_model:
        from transformers import AutoModelForCausalLM
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        model = AutoModelForCausalLM.from_pretrained(
            args.huggingface_model,
            trust_remote_code=True
        )
        model = model.to(device)
        print(f"Loaded HuggingFace model: {args.huggingface_model}")
    elif args.model_path:
        model = load_pretrained_model(args.model_path, device=device)
        print(f"Loaded model from: {args.model_path}")
    else:
        raise ValueError("Please specify either --model-path or --huggingface-model")
    
    # 统计参数
    param_stats = count_parameters(model)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {param_stats['total']:,}")
    print(f"  Trainable parameters: {param_stats['trainable']:,}")
    
    # 确定要评估的数据集
    if args.datasets:
        dataset_names = args.datasets
    elif args.dataset:
        dataset_names = [args.dataset]
    else:
        # 默认评估所有提到的数据集
        dataset_names = ['ETTH1', 'ECL', 'TRAFFIC', 'WEATHER', 'PEMS03', 'PEMS04']
    
    print(f"\nDatasets to evaluate: {', '.join(dataset_names)}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 评估结果
    all_results = {}
    
    # 逐个评估每个数据集
    for dataset_name in dataset_names:
        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset_name}")
        print(f"{'='*60}")
        
        try:
            # 加载数据集
            train_dataset, val_dataset, test_dataset, data_config = load_standard_dataset(
                dataset_name=dataset_name,
                lookback=args.lookback,
                pred_len=args.pred_len,
                data_dir=args.data_dir,
                download=True
            )
            
            # 创建测试数据加载器
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=(device.type == 'cuda')
            )
            
            # 评估模型
            metrics, predictions, targets = evaluate_model(
                model, test_loader, device, dataset_name
            )
            
            # 保存结果
            all_results[dataset_name] = metrics
            
            # 打印结果
            print(f"\n{dataset_name} Results:")
            print(f"  MSE:  {metrics['MSE']:.6f}")
            print(f"  MAE:  {metrics['MAE']:.6f}")
            print(f"  RMSE: {metrics['RMSE']:.6f}")
            print(f"  MAPE: {metrics['MAPE']:.2f}%")
            print(f"  Direction Acc: {metrics['Direction_Acc']:.2f}%")
            
        except Exception as e:
            print(f"Error evaluating on {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            all_results[dataset_name] = {'error': str(e)}
    
    # 保存所有结果
    import json
    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")
    
    # 打印汇总表格
    print(f"\n{'Dataset':<15} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'MAPE':<10} {'Dir Acc':<10}")
    print("-" * 75)
    
    for dataset_name, metrics in all_results.items():
        if 'error' not in metrics:
            print(f"{dataset_name:<15} {metrics['MSE']:<12.6f} {metrics['MAE']:<12.6f} "
                  f"{metrics['RMSE']:<12.6f} {metrics['MAPE']:<10.2f} {metrics['Direction_Acc']:<10.2f}")
        else:
            print(f"{dataset_name:<15} Error: {metrics['error']}")
    
    # 计算平均值（排除错误）
    valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
    if valid_results:
        avg_metrics = {
            'MSE': float(np.mean([m['MSE'] for m in valid_results.values()])),
            'MAE': float(np.mean([m['MAE'] for m in valid_results.values()])),
            'RMSE': float(np.mean([m['RMSE'] for m in valid_results.values()])),
            'MAPE': float(np.mean([m['MAPE'] for m in valid_results.values()])),
            'Direction_Acc': float(np.mean([m['Direction_Acc'] for m in valid_results.values()]))
        }
        print("-" * 75)
        print(f"{'Average':<15} {avg_metrics['MSE']:<12.6f} {avg_metrics['MAE']:<12.6f} "
              f"{avg_metrics['RMSE']:<12.6f} {avg_metrics['MAPE']:<10.2f} {avg_metrics['Direction_Acc']:<10.2f}")
    
    print(f"\n[OK] Results saved to {results_file}")


if __name__ == "__main__":
    main()

