import numpy as np
import random
import torch
import os
import argparse
import warnings
warnings.filterwarnings('ignore')
from exp.exp_main import Exp_Main
# from exp.exp_main_ml import Exp_Main_ML  # Removed for Facformer-focused implementation

def main(args):
	"""
	Main training and testing function for Facformer
	
	Args:
		args: Configuration arguments
	"""
	# Set random seeds for reproducibility
	fix_seed = 2023
	random.seed(fix_seed)
	torch.manual_seed(fix_seed)
	np.random.seed(fix_seed)



	if args.is_training:
		for iteration in range(args.itr):
			setting = '{}_{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
				args.model_id,
				args.model,
				args.data,
				args.seq_len,
				args.label_len,
				args.pred_len,
				args.d_model,
				args.n_heads,
				args.e_layers,
				args.d_layers,
				args.d_ff,
				args.factor,
				args.embed,
				args.distil,
				args.des,
				iteration)
			# Initialize experiment - focused on deep learning models
			if args.model in ['Facformer', 'Autoformer', 'Informer', 'Transformer', 'LSTM', 'RNN']:
				exp = Exp_Main(args)
			else:
				raise ValueError(f"Model {args.model} is not supported. Please use one of: Facformer, Autoformer, Transformer, LSTM, RNN")
			
			print(f'>>>>>>>Start training: {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>>')
			
			# Train model
			if args.use_mark:
				print('Using marked data (temperature and time information)')
				exp.train_mark(setting)
			else:
				exp.train(setting)

			print(f'>>>>>>>Testing: {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
			
			# Test model
			if args.use_mark:
				exp.test_mark(setting, load=True)
			else:
				exp.test(setting, load=True)
			print(f'Completed iteration {iteration + 1}/{args.itr} for {setting}')





if __name__ == '__main__':
	"""
	Main entry point for Facformer training and evaluation
	"""
	parser = argparse.ArgumentParser(description='Facformer: Deep Learning for Aquatic Product Shelf-Life Prediction')
	
	# Model selection
	parser.add_argument('--is_training', type=int, default=1, required=False, 
	                   help='Training status (1 for training, 0 for testing only)')
	parser.add_argument('--model_id', type=str, default='test', required=False, 
	                   help='Model identifier for distinguishing experiments')
	parser.add_argument('--model', type=str, default='Facformer', required=False,
	                   choices=['Facformer', 'Autoformer', 'Transformer', 'LSTM', 'RNN'],
	                   help='Model type to use (focused on deep learning models)')
	# Data configuration
	parser.add_argument('--data', type=str, default='Food-SPV', 
	                   help='Dataset name to use')
	parser.add_argument('--data_path', type=str, default='./data/Food-SPV/', 
	                   help='Path to the dataset directory')
	parser.add_argument('--use_constant', type=bool, default=False, 
	                   help='Whether to use constant temperature data')
	parser.add_argument('--use_mark', type=bool, default=False, 
	                   help='Whether to use mark embeddings for temperature and time')
	parser.add_argument('--scale', type=bool, default=True, 
	                   help='Whether to apply data normalization')
	parser.add_argument('--embed', type=str, default='timeF', 
	                   choices=['fixed', 'learned', 'embed_T', 'timeF'],
	                   help='Time features encoding method')
	# Model checkpoints
	parser.add_argument('--checkpoints', type=str, default='./checkpoints/', 
	                   help='Directory to save model checkpoints')
	
	# Training configuration
	parser.add_argument('--train_mode', type=str, default='generation', choices=['generation', 'once'],
	                   help='Training mode: generation (autoregressive) or once (teacher forcing)')
	parser.add_argument('--fine_tuning', type=bool, default=False, 
	                   help='Whether to support fine-tuning with new data')


	# Model architecture parameters
	parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
	parser.add_argument('--seq_len', type=int, default=16, help='Input sequence length')
	parser.add_argument('--label_len', type=int, default=8, help='Start token length for decoder')
	parser.add_argument('--pred_len', type=int, default=8, help='Prediction sequence length')
	
	# Transformer architecture parameters
	parser.add_argument('--enc_in', type=int, default=2, help='Encoder input feature size')
	parser.add_argument('--dec_in', type=int, default=2, help='Decoder input feature size')
	parser.add_argument('--enc_mark_in', type=int, default=2, help='Encoder mark input size')
	parser.add_argument('--dec_mark_in', type=int, default=2, help='Decoder mark input size')
	parser.add_argument('--c_out', type=int, default=4, help='Output feature dimension')
	parser.add_argument('--d_model', type=int, default=64, help='Model dimension')
	parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
	parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
	parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers')
	parser.add_argument('--d_layers', type=int, default=1, help='Number of decoder layers')
	parser.add_argument('--d_ff', type=int, default=256, help='Feed-forward network dimension')
	parser.add_argument('--activation', type=str, default='relu', 
	                   choices=['relu', 'gelu'], help='Activation function')
	parser.add_argument('--factor', type=int, default=1, help='Attention factor')
	

	parser.add_argument('--distil', action='store_false', default=True,
	                   help='Whether to use distilling in encoder (use --distil to disable)')
	
	# Training efficiency parameters
	parser.add_argument('--num_workers', type=int, default=0, 
	                   help='Number of workers for data loading')
	
	# Training process parameters
	parser.add_argument('--itr', type=int, default=1, help='Number of experiment iterations')
	parser.add_argument('--patience', type=int, default=2, help='Early stopping patience')
	parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
	parser.add_argument('--train_epochs', type=int, default=3, help='Number of training epochs')
	parser.add_argument('--lradj', type=str, default='type1', help='Learning rate adjustment strategy')

	# Model output and optimization
	parser.add_argument('--output_attention', action='store_true', default=False,
	                   help='Whether to output attention weights')
	parser.add_argument('--des', type=str, default='test', help='Experiment description')
	parser.add_argument('--use_amp', action='store_true', default=False,
	                   help='Use automatic mixed precision training (requires PyTorch 1.6+)')
	
	# GPU configuration
	parser.add_argument('--use_gpu', type=bool, default=False, help='Use GPU for training')
	parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
	parser.add_argument('--use_multi_gpu', action='store_true', default=False,
	                   help='Use multiple GPUs')
	parser.add_argument('--devices', type=str, default='0', 
	                   help='Device IDs for multiple GPUs (comma-separated)')

	# Model-specific parameters
	parser.add_argument('--moving_avg', type=int, default=7, 
	                   help='Moving average window size (for Autoformer)')
	parser.add_argument('--hidden_size', type=int, default=32, 
	                   help='Hidden size for RNN-based models')
	parser.add_argument('--n_layers', type=int, default=1, 
	                   help='Number of layers for RNN-based models')

	








	
	

	


	# Parse arguments and configure model-specific settings
	args = parser.parse_args()
	
	# Configure Facformer-specific settings
	if args.model == 'Facformer':
		args.use_mark = True  # Facformer requires mark embeddings
		args.enc_in = 2       # Quality indicators (Bacterial_Count, TVB_N)
		args.dec_in = 2       # Same as encoder input
	
	# GPU configuration
	if args.use_gpu and args.use_multi_gpu:
		args.devices = args.devices.replace(' ', '')
		device_ids = args.devices.split(',')
		args.device_ids = [int(id_) for id_ in device_ids]
		args.gpu = args.device_ids[0]

	main(args)