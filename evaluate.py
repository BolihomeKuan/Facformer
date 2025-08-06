#!/usr/bin/env python3
"""
Facformer Model Evaluation Script

This script provides comprehensive evaluation of the Facformer model,
including performance metrics, visualization, and comparison analysis.

Usage:
    python evaluate.py --model_path ./checkpoints/ --test_data ./data/test/
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from models.Facformer import Model
from data_provider.data_loader import get_data
from utils.metrics import metric

def load_results(results_path):
    """
    Load pre-computed results from model testing
    
    Args:
        results_path: Path to results directory
        
    Returns:
        predictions, ground truth, and metrics
    """
    pred_path = os.path.join(results_path, 'pred.npy')
    true_path = os.path.join(results_path, 'true.npy')
    metrics_path = os.path.join(results_path, 'metrics.npy')
    
    if not all(os.path.exists(p) for p in [pred_path, true_path, metrics_path]):
        raise FileNotFoundError("Result files not found. Please run model testing first.")
    
    predictions = np.load(pred_path)
    ground_truth = np.load(true_path)
    metrics = np.load(metrics_path)
    
    return predictions, ground_truth, metrics

def calculate_detailed_metrics(predictions, ground_truth):
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        predictions: Model predictions
        ground_truth: Ground truth values
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Flatten arrays for overall metrics
    pred_flat = predictions.reshape(-1, predictions.shape[-1])
    true_flat = ground_truth.reshape(-1, ground_truth.shape[-1])
    
    # Calculate metrics for each feature
    feature_names = ['Quality_Index_1', 'Quality_Index_2', 'Quality_Index_3', 'Quality_Index_4']
    for i, feature in enumerate(feature_names):
        if i < pred_flat.shape[1]:  # Check if feature exists in predictions
            metrics[f'{feature}_MSE'] = mean_squared_error(true_flat[:, i], pred_flat[:, i])
            metrics[f'{feature}_MAE'] = mean_absolute_error(true_flat[:, i], pred_flat[:, i])
            metrics[f'{feature}_RMSE'] = np.sqrt(metrics[f'{feature}_MSE'])
            metrics[f'{feature}_R2'] = r2_score(true_flat[:, i], pred_flat[:, i])
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((true_flat[:, i] - pred_flat[:, i]) / (true_flat[:, i] + 1e-8))) * 100
            metrics[f'{feature}_MAPE'] = mape
    
    # Overall metrics
    metrics['Overall_MSE'] = mean_squared_error(true_flat, pred_flat)
    metrics['Overall_MAE'] = mean_absolute_error(true_flat, pred_flat)
    metrics['Overall_RMSE'] = np.sqrt(metrics['Overall_MSE'])
    
    return metrics

def create_evaluation_plots(predictions, ground_truth, save_dir="./evaluation_plots/"):
    """
    Create comprehensive evaluation plots
    
    Args:
        predictions: Model predictions
        ground_truth: Ground truth values
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    feature_names = ['Quality Index 1', 'Quality Index 2', 'Quality Index 3', 'Quality Index 4']
    
    # 1. Prediction vs Ground Truth Scatter Plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (ax, feature) in enumerate(zip(axes, feature_names)):
        pred_flat = predictions[:, :, i].flatten()
        true_flat = ground_truth[:, :, i].flatten()
        
        ax.scatter(true_flat, pred_flat, alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(true_flat.min(), pred_flat.min())
        max_val = max(true_flat.max(), pred_flat.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel(f'Ground Truth {feature}')
        ax.set_ylabel(f'Predicted {feature}')
        ax.set_title(f'{feature}: Prediction vs Ground Truth')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calculate and display R²
        r2 = r2_score(true_flat, pred_flat)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_vs_truth.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Time Series Prediction Examples
    n_samples = min(3, predictions.shape[0])  # Reduce samples to fit 4 features
    fig, axes = plt.subplots(n_samples, 4, figsize=(20, 3*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for sample in range(n_samples):
        for i, feature in enumerate(feature_names):
            ax = axes[sample, i] if n_samples > 1 else axes[i]
            
            time_steps = range(predictions.shape[1])
            ax.plot(time_steps, ground_truth[sample, :, i], 'b-', 
                   label='Ground Truth', linewidth=2)
            ax.plot(time_steps, predictions[sample, :, i], 'r--', 
                   label='Prediction', linewidth=2)
            
            ax.set_xlabel('Time Step')
            ax.set_ylabel(feature)
            ax.set_title(f'Sample {sample+1}: {feature} Time Series')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'time_series_examples.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Error Distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (ax, feature) in enumerate(zip(axes, feature_names)):
        errors = (predictions[:, :, i] - ground_truth[:, :, i]).flatten()
        
        ax.hist(errors, bins=50, alpha=0.7, density=True)
        ax.axvline(errors.mean(), color='red', linestyle='--', 
                  label=f'Mean Error: {errors.mean():.3f}')
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Density')
        ax.set_title(f'{feature}: Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()

def generate_report(metrics, save_path="evaluation_report.txt"):
    """
    Generate comprehensive evaluation report
    
    Args:
        metrics: Dictionary of calculated metrics
        save_path: Path to save the report
    """
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Facformer Model Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Feature-wise Performance:\n")
        f.write("-" * 30 + "\n")
        
        features = ['Quality_Index_1', 'Quality_Index_2', 'Quality_Index_3', 'Quality_Index_4']
        for feature in features:
            f.write(f"\n{feature}:\n")
            f.write(f"  MSE:  {metrics[f'{feature}_MSE']:.6f}\n")
            f.write(f"  MAE:  {metrics[f'{feature}_MAE']:.6f}\n")
            f.write(f"  RMSE: {metrics[f'{feature}_RMSE']:.6f}\n")
            f.write(f"  R²:   {metrics[f'{feature}_R2']:.6f}\n")
            f.write(f"  MAPE: {metrics[f'{feature}_MAPE']:.3f}%\n")
        
        f.write(f"\nOverall Performance:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Overall MSE:  {metrics['Overall_MSE']:.6f}\n")
        f.write(f"Overall MAE:  {metrics['Overall_MAE']:.6f}\n")
        f.write(f"Overall RMSE: {metrics['Overall_RMSE']:.6f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Report generated by Facformer evaluation script\n")
    
    print(f"Evaluation report saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Facformer Model Evaluation')
    parser.add_argument('--results_path', type=str, 
                       		default='./results/Food-SPV_Final_Exp_Facformer_Food-SPV_sl16_ll8_pl8_dm64_nh8_el2_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0/',
                       help='Path to results directory')
    parser.add_argument('--output_dir', type=str, default='./evaluation_output/',
                       help='Directory to save evaluation outputs')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Facformer Model Evaluation")
    print("=" * 60)
    
    try:
        # Load results
        print("Loading evaluation results...")
        predictions, ground_truth, saved_metrics = load_results(args.results_path)
        print("✓ Results loaded successfully")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Ground truth shape: {ground_truth.shape}")
        
        # Calculate detailed metrics
        print("\nCalculating detailed metrics...")
        detailed_metrics = calculate_detailed_metrics(predictions, ground_truth)
        print("✓ Metrics calculated")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate plots
        print("\nGenerating evaluation plots...")
        plot_dir = os.path.join(args.output_dir, "plots")
        create_evaluation_plots(predictions, ground_truth, plot_dir)
        print("✓ Plots generated")
        
        # Generate report
        print("\nGenerating evaluation report...")
        report_path = os.path.join(args.output_dir, "evaluation_report.txt")
        generate_report(detailed_metrics, report_path)
        print("✓ Report generated")
        
        # Display summary
        print("\nEvaluation Summary:")
        print("-" * 40)
        for feature in ['Quality_Index_1', 'Quality_Index_2', 'Quality_Index_3', 'Quality_Index_4']:
            if f'{feature}_R2' in detailed_metrics:
                print(f"{feature} R²: {detailed_metrics[f'{feature}_R2']:.4f}")
        print(f"Overall RMSE: {detailed_metrics['Overall_RMSE']:.6f}")
        print(f"Overall MAE: {detailed_metrics['Overall_MAE']:.6f}")
        
        print(f"\n✓ Evaluation completed! Results saved to {args.output_dir}")
        
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())