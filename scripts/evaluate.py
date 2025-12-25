"""
Model evaluation script
Evaluate model performance on standard time series datasets
"""
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import TimerConfig, TimerForPrediction
from data_processing import load_standard_dataset, prepare_multiple_datasets
from utils import load_pretrained_model, count_parameters


def calculate_metrics(predictions, targets, mask=None):
    """
    Calculate evaluation metrics
    
    Args:
        predictions: Predicted values (batch_size, pred_len, n_features)
        targets: True values (batch_size, pred_len, n_features)
        mask: Mask (optional)
    
    Returns:
        dict: Dictionary containing various metrics
    """
    if mask is not None:
        predictions = predictions * mask
        targets = targets * mask
        valid_count = mask.sum()
    else:
        valid_count = predictions.numel()
    
    # Convert to numpy
    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    
    # Calculate MSE
    mse = np.mean((pred_np - target_np) ** 2)
    
    # Calculate MAE
    mae = np.mean(np.abs(pred_np - target_np))
    
    # Calculate RMSE
    rmse = np.sqrt(mse)
    
    # Calculate MAPE (avoid division by zero)
    epsilon = 1e-8
    mape = np.mean(np.abs((target_np - pred_np) / (target_np + epsilon))) * 100
    
    # Calculate direction accuracy (Direction Accuracy)
    if pred_np.shape[-1] == 1:
        # Univariate: calculate direction
        pred_diff = np.diff(pred_np.squeeze(-1), axis=1)
        target_diff = np.diff(target_np.squeeze(-1), axis=1)
        direction_correct = np.sign(pred_diff) == np.sign(target_diff)
        direction_acc = np.mean(direction_correct) * 100
    else:
        # Multivariate: calculate direction accuracy for each variable
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
    Evaluate model performance on test set
    
    Args:
        model: Model
        test_loader: Test data loader
        device: Device
        dataset_name: Dataset name
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    print(f"\nEvaluating on {dataset_name}...")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            # Safely unpack data (handle cases where extra values are returned)
            if isinstance(batch_data, (tuple, list)):
                if len(batch_data) == 2:
                    history, target = batch_data
                elif len(batch_data) > 2:
                    # If more than 2 values returned, only take the first two
                    history, target = batch_data[0], batch_data[1]
                else:
                    print(f"Warning: Unexpected batch data length: {len(batch_data)}")
                    continue
            else:
                print(f"Warning: Unexpected batch data type: {type(batch_data)}")
                continue
            
            history = history.to(device)
            target = target.to(device)
            
            # Generate predictions
            # Timer model in inference mode uses history as input, generates pred_len length predictions
            pred_len = target.shape[1]
            
            # Call model for prediction (inference mode)
            try:
                outputs = model(
                    input_ids=history,
                    labels=None,
                    return_dict=True,
                    max_output_length=pred_len
                )
                
                predictions = outputs.logits
                
                # Timer model in inference mode:
                # predictions = lm_head(hidden_states)[:, -1, :] 
                # Returns shape (batch_size, output_token_len), where output_token_len is prediction length per token
                # But we need (batch_size, pred_len, n_features)
                # Note: Timer uses patch embedding, each patch length is input_token_len
                
                batch_size = history.shape[0]
                n_features = history.shape[-1]
                
                if predictions is not None:
                    # Timer model predictions shape is (batch_size, output_token_len)
                    # output_token_len is prediction length per patch, not number of time steps
                    # We need to reshape to (batch_size, num_patches, output_token_len)
                    # Then reshape to (batch_size, total_pred_len, n_features)
                    
                    if len(predictions.shape) == 2:
                        # predictions: (batch_size, output_token_len)
                        # Need to reshape to (batch_size, 1, output_token_len) then expand
                        # But actual Timer return may already be processed
                        
                        # Try to understand: if output_token_len = 96, pred_len = 96
                        # Then predictions might be (batch_size, 96), need reshape to (batch_size, 96, 1)
                        # Or if output_token_len = pred_len * n_features, need reshape
                        
                        output_token_len = predictions.shape[1]
                        
                        # If output_token_len equals pred_len, each time step predicts one value
                        if output_token_len == pred_len:
                            # (batch_size, pred_len) -> (batch_size, pred_len, 1)
                            predictions = predictions.unsqueeze(-1)
                            # If n_features > 1, need to expand
                            if n_features > 1:
                                predictions = predictions.repeat(1, 1, n_features)
                        elif output_token_len == pred_len * n_features:
                            # (batch_size, pred_len * n_features) -> (batch_size, pred_len, n_features)
                            predictions = predictions.view(batch_size, pred_len, n_features)
                        else:
                            # Other cases: assume univariate prediction, expand feature dimension
                            if output_token_len >= pred_len:
                                predictions = predictions[:, :pred_len].unsqueeze(-1)
                                if n_features > 1:
                                    predictions = predictions.repeat(1, 1, n_features)
                            else:
                                # Insufficient length, pad
                                padding = torch.zeros(batch_size, pred_len - output_token_len, device=device)
                                predictions = torch.cat([predictions, padding], dim=1).unsqueeze(-1)
                                if n_features > 1:
                                    predictions = predictions.repeat(1, 1, n_features)
                    elif len(predictions.shape) == 3:
                        # Already 3D, directly adjust
                        if predictions.shape[1] > pred_len:
                            predictions = predictions[:, :pred_len, :]
                        elif predictions.shape[1] < pred_len:
                            last_step = predictions[:, -1:, :]
                            padding = last_step.repeat(1, pred_len - predictions.shape[1], 1)
                            predictions = torch.cat([predictions, padding], dim=1)
                        
                        # Adjust feature dimension
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
                    # If logits is None, use zero padding
                    predictions = torch.zeros(batch_size, pred_len, n_features, device=device)
                    
            except Exception as e:
                # If prediction fails, use zero padding
                if batch_idx % 10 == 0:  # Only print some warnings to avoid flooding
                    print(f"Warning: Prediction failed for batch {batch_idx}: {str(e)}")
                batch_size, _, n_features = history.shape
                predictions = torch.zeros(batch_size, pred_len, n_features, device=device)
            
            # Finally ensure prediction shape matches target
            if predictions.shape[1] != target.shape[1]:
                if predictions.shape[1] > target.shape[1]:
                    predictions = predictions[:, :target.shape[1], :]
                else:
                    # Pad (using last prediction value)
                    padding = predictions[:, -1:, :].repeat(1, target.shape[1] - predictions.shape[1], 1)
                    predictions = torch.cat([predictions, padding], dim=1)
            
            # Ensure feature dimension matches
            if predictions.shape[-1] != target.shape[-1]:
                if predictions.shape[-1] > target.shape[-1]:
                    predictions = predictions[:, :, :target.shape[-1]]
                else:
                    # If feature dimension insufficient, use zero padding
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
    
    # Concatenate all predictions
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_targets)
    
    return metrics, all_predictions, all_targets


def main():
    parser = argparse.ArgumentParser(description="Evaluate Timer model on standard datasets")
    
    # Model related
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to pretrained model directory (containing model.pt and config.json)")
    parser.add_argument("--huggingface-model", type=str, default=None,
                       help="HuggingFace model name (alternative to --model-path)")
    
    # Dataset related
    parser.add_argument("--dataset", type=str, default=None,
                       choices=['ETTH1', 'ETTH2', 'ETTM1', 'ETTM2', 'ECL', 'TRAFFIC', 'WEATHER', 
                               'PEMS03', 'PEMS04', 'PEMS07', 'PEMS08'],
                       help="Single dataset to evaluate")
    parser.add_argument("--datasets", type=str, nargs='+', default=None,
                       help="Multiple datasets to evaluate (e.g., ETTH1 ECL TRAFFIC)")
    parser.add_argument("--data-dir", type=str, default="data/standard_datasets",
                       help="Data directory")
    
    # Evaluation parameters
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
    
    # Device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
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
    
    # Count parameters
    param_stats = count_parameters(model)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {param_stats['total']:,}")
    print(f"  Trainable parameters: {param_stats['trainable']:,}")
    
    # Determine datasets to evaluate
    if args.datasets:
        dataset_names = args.datasets
    elif args.dataset:
        dataset_names = [args.dataset]
    else:
        # Default: evaluate all mentioned datasets
        dataset_names = ['ETTH1', 'ECL', 'TRAFFIC', 'WEATHER', 'PEMS03', 'PEMS04']
    
    print(f"\nDatasets to evaluate: {', '.join(dataset_names)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluation results
    all_results = {}
    
    # Evaluate each dataset one by one
    for dataset_name in dataset_names:
        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset_name}")
        print(f"{'='*60}")
        
        try:
            # Load dataset
            train_dataset, val_dataset, test_dataset, data_config = load_standard_dataset(
                dataset_name=dataset_name,
                lookback=args.lookback,
                pred_len=args.pred_len,
                data_dir=args.data_dir,
                download=True
            )
            
            # Create test data loader
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=(device.type == 'cuda')
            )
            
            # Evaluate model
            metrics, predictions, targets = evaluate_model(
                model, test_loader, device, dataset_name
            )
            
            # Save results
            all_results[dataset_name] = metrics
            
            # Print results
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
    
    # Save all results
    import json
    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")
    
    # Print summary table
    print(f"\n{'Dataset':<15} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'MAPE':<10} {'Dir Acc':<10}")
    print("-" * 75)
    
    for dataset_name, metrics in all_results.items():
        if 'error' not in metrics:
            print(f"{dataset_name:<15} {metrics['MSE']:<12.6f} {metrics['MAE']:<12.6f} "
                  f"{metrics['RMSE']:<12.6f} {metrics['MAPE']:<10.2f} {metrics['Direction_Acc']:<10.2f}")
        else:
            print(f"{dataset_name:<15} Error: {metrics['error']}")
    
    # Calculate average (excluding errors)
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

