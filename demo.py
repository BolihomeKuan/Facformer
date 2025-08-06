#!/usr/bin/env python3
"""
Facformer Demo Script

This script demonstrates how to use the Facformer model for shelf-life prediction
of aquatic products under variable temperature conditions.

Usage:
    python demo.py --data_path ./data/Food-SPV/ --model_path ./checkpoints/
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import warnings
warnings.filterwarnings('ignore')

from models.Facformer import Model
from data_provider.data_loader import get_data
from utils.metrics import metric
from utils.tools import visual

def load_model(model_path, configs):
    """
    Load pre-trained Facformer model
    
    Args:
        model_path: Path to the model checkpoint
        configs: Model configuration
        
    Returns:
        Loaded model
    """
    model = Model(configs)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def prepare_demo_data(data_path, seq_len=16, pred_len=8):
    """
    Prepare demonstration data for prediction
    
    Args:
        data_path: Path to data directory
        seq_len: Input sequence length
        pred_len: Prediction length
        
    Returns:
        Demo data tensor and metadata
    """
    # Load sample data (replace with your actual data loading logic)
    print("Loading demo data...")
    
    # Example: Load a single sample for demonstration
    sample_data = pd.read_csv(f"{data_path}/test/T22-Sample1_微波.csv")
    
    # Extract features: typically [Day, Temperature, Quality_Indicator_1, Quality_Indicator_2]
    # Adjust based on your actual data format
    features = sample_data[['Day', 'Temperature', 'Bacterial_Count', 'TVB_N']].values
    
    # Prepare input sequences
    x_enc = features[:seq_len, 2:]  # Quality indicators
    x_enc_mark = features[:seq_len, :2]  # Day and Temperature
    
    # Convert to tensors
    x_enc = torch.FloatTensor(x_enc).unsqueeze(0)
    x_enc_mark = torch.FloatTensor(x_enc_mark).unsqueeze(0)
    
    # Prepare decoder input (typically zeros for prediction)
    x_dec = torch.zeros(1, pred_len, x_enc.shape[-1])
    x_dec_mark = torch.zeros(1, pred_len, x_enc_mark.shape[-1])
    
    return x_enc, x_enc_mark, x_dec, x_dec_mark, features

def run_prediction(model, x_enc, x_enc_mark, x_dec, x_dec_mark):
    """
    Run prediction using Facformer model
    
    Args:
        model: Facformer model
        x_enc: Encoder input
        x_enc_mark: Encoder mark input
        x_dec: Decoder input
        x_dec_mark: Decoder mark input
        
    Returns:
        Predictions
    """
    with torch.no_grad():
        predictions = model(x_enc, x_enc_mark, x_dec, x_dec_mark)
    return predictions

def visualize_results(historical_data, predictions, save_path="demo_results.png"):
    """
    Visualize prediction results
    
    Args:
        historical_data: Historical data array
        predictions: Model predictions
        save_path: Path to save visualization
    """
    plt.figure(figsize=(15, 10))
    
    # Plot historical data
    seq_len = len(historical_data)
    time_historical = range(seq_len)
    time_future = range(seq_len, seq_len + len(predictions))
    
    plt.subplot(2, 2, 1)
    plt.plot(time_historical, historical_data[:, 2], 'b-', label='Historical Quality Index 1', linewidth=2)
    plt.plot(time_future, predictions[:, 0], 'r--', label='Predicted Quality Index 1', linewidth=2)
    plt.xlabel('Time (Days)')
    plt.ylabel('Quality Index 1')
    plt.title('Facformer Prediction: Quality Index 1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(time_historical, historical_data[:, 3], 'b-', label='Historical Quality Index 2', linewidth=2)
    plt.plot(time_future, predictions[:, 1], 'r--', label='Predicted Quality Index 2', linewidth=2)
    plt.xlabel('Time (Days)')
    plt.ylabel('Quality Index 2')
    plt.title('Facformer Prediction: Quality Index 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(time_historical, historical_data[:, 2], 'b-', label='Historical Data', linewidth=2)  
    plt.plot(time_future, predictions[:, 2], 'r--', label='Predicted Quality Index 3', linewidth=2)
    plt.xlabel('Time (Days)')
    plt.ylabel('Quality Index 3')
    plt.title('Facformer Prediction: Quality Index 3')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(time_historical, historical_data[:, 3], 'b-', label='Historical Data', linewidth=2)
    plt.plot(time_future, predictions[:, 3], 'r--', label='Predicted Quality Index 4', linewidth=2)
    plt.xlabel('Time (Days)')
    plt.ylabel('Quality Index 4')
    plt.title('Facformer Prediction: Quality Index 4')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Facformer Demo Script')
    parser.add_argument('--data_path', type=str, default='./data/Food-SPV/', 
                       help='Path to data directory')
    parser.add_argument('--model_path', type=str, 
                       default='./checkpoints/Food-SPV_Final_Exp_Facformer_Food-SPV_sl16_ll8_pl8_dm64_nh8_el2_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--seq_len', type=int, default=16, help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=8, help='Prediction length')
    parser.add_argument('--output_path', type=str, default='./demo_results.png', 
                       help='Path to save results')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Facformer Model Demonstration")
    print("=" * 60)
    
    # Model configuration (matched with actual checkpoint)
    class Config:
        def __init__(self):
            self.pred_len = args.pred_len
            self.output_attention = False
            self.enc_in = 2
            self.dec_in = 2
            self.d_model = 64
            self.embed = 'timeF'
            self.dropout = 0.1
            self.factor = 1
            self.n_heads = 8
            self.e_layers = 2
            self.d_layers = 1
            self.d_ff = 256
            self.activation = 'relu'
            self.c_out = 4
    
    configs = Config()
    
    try:
        # Load model
        print("Loading Facformer model...")
        model = load_model(args.model_path, configs)
        print("✓ Model loaded successfully")
        
        # Prepare demo data
        print("Preparing demo data...")
        x_enc, x_enc_mark, x_dec, x_dec_mark, historical_data = prepare_demo_data(
            args.data_path, args.seq_len, args.pred_len
        )
        print("✓ Demo data prepared")
        
        # Run prediction
        print("Running prediction...")
        predictions = run_prediction(model, x_enc, x_enc_mark, x_dec, x_dec_mark)
        predictions = predictions.squeeze().numpy()
        print("✓ Prediction completed")
        
        # Display results
        print("\nPrediction Results:")
        print("-" * 40)
        for i, pred in enumerate(predictions):
            print(f"Day {args.seq_len + i + 1}: Quality Index 1 = {pred[0]:.3f}, Quality Index 2 = {pred[1]:.3f}, Quality Index 3 = {pred[2]:.3f}, Quality Index 4 = {pred[3]:.3f}")
        
        # Visualize results
        print("\nGenerating visualization...")
        visualize_results(historical_data, predictions, args.output_path)
        
        print("\n✓ Demo completed successfully!")
        
    except FileNotFoundError as e:
        print(f"✗ Error: File not found - {e}")
        print("Please check the file paths and ensure the model and data exist.")
    except Exception as e:
        print(f"✗ Error during demo: {e}")
        print("Please check your model configuration and data format.")

if __name__ == "__main__":
    main()