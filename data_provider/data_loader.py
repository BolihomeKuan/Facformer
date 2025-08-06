from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import numpy as np
import joblib
"""
Data Loading Module for Aquatic Products Shelf-life Prediction

Variables:
- B: Bacterial count (quality indicator)
- N: TVB-N level (quality indicator) 
- T: Temperature (environmental factor)
- D: Days (time in days)

Two data design patterns:
(1) With mark embeddings: seq_x, seq_y, seq_x_mark, seq_y_mark
    - seq_x: B and N (dependent variables - quality indicators)
    - x_mark: T and D (independent variables - environmental factors)

(2) Standard format: seq_x, seq_y only

Normalization: Column-wise standardization on training set, 
applied to validation and test sets (following Autoformer approach)
"""
class Dataset_dT(Dataset):
	def __init__(self,datas,size=None,train_mode=None,use_mark=None,use_constant=None):
		assert train_mode in ['generation','once']
		self.train_mode = train_mode  # Generation mode: sequential vs one-shot prediction
		self.use_mark = use_mark  # Whether to use separate mark embeddings for time and temperature			
		self.use_constant = use_constant  # Whether to use constant temperature data

		if size is None:
			# Default: use 8 input points to predict 8 points (4 overlap + 4 new)
			self.seq_len = 8
			self.label_len = 4
			self.pred_len = 4
		else:
			# size = [seq_len, label_len, pred_len]
			self.seq_len = size[0]
			self.label_len = size[1]
			self.pred_len = size[2]
		self.__read_data__(datas)

	def __read_data__(self,datas):
		self.data_x = []
		self.data_y = []
		self.data_x_mark = []
		self.data_y_mark = []
		if self.use_constant:
			pass  # Constant temperature handling not implemented yet
		else:
			for ii in range(datas.shape[0]):
				data = datas[ii,:,:]
				if self.train_mode == 'generation':
					if self.use_mark:
						for index in range(len(data)-self.seq_len-self.pred_len+1):
							s_begin = index#0
							s_end = s_begin + self.seq_len#30
							r_begin = s_end - self.label_len#15
							r_end = r_begin + self.label_len + self.pred_len#60
							seq_x = data[s_begin:s_end,2:4]  # Quality indicators (B, N)
							seq_y = data[r_begin:r_end,2:4]  # Target quality indicators
							seq_x_mark = data[s_begin:s_end,0:2]  # Environmental marks (D, T)
							seq_y_mark = data[r_begin:r_end,0:2]  # Target environmental marks
							self.data_x.append(seq_x)
							self.data_y.append(seq_y)
							self.data_x_mark.append(seq_x_mark)
							self.data_y_mark.append(seq_y_mark)
					else:
						for index in range(len(data)-self.seq_len-self.pred_len+1):
							s_begin = index#0
							s_end = s_begin + self.seq_len#8
							r_begin = s_end - self.label_len#8-4
							r_end = r_begin + self.label_len + self.pred_len
							seq_x = data[s_begin:s_end]  # Input sequence
							seq_y = data[r_begin:r_end]  # Target sequence
							self.data_x.append(seq_x)
							self.data_y.append(seq_y)

				elif self.train_mode == 'once':
					# One-shot prediction mode (alternative training approach)
					pass
		# print(len(self.data_x))

	def __getitem__(self,index):
		if self.use_mark:
			seq_x = self.data_x[index]  # Input sequence
			seq_y = self.data_y[index]  # Target sequence
			seq_x_mark = self.data_x_mark[index]  # Input marks
			seq_y_mark = self.data_y_mark[index]  # Target marks
			return seq_x,seq_y,seq_x_mark,seq_y_mark
		else:
			seq_x = self.data_x[index]  # Input sequence
			seq_y = self.data_y[index]  # Target sequence
			return seq_x,seq_y

	def __len__(self):
		return len(self.data_x)


	def inverse_transform(self,data):
		"""Inverse standardization using sklearn's built-in function"""
		return self.scaler.inverse_transform(data)



def data_provider(args,flag,datas):
	data_dict = {
	'Food-SPV':Dataset_dT,
	'Food-SPV_mark':Dataset_dT	
	}
	Data = data_dict[args.data]


	if flag == 'test':
		shuffle_flag = False
		drop_last = False
		batch_size = args.batch_size

	elif flag == 'pred':
		shuffle_flag = False
		drop_last = False
		batch_size = 1
		Data = data_dict[args.data]
	else:
		shuffle_flag = True  # Shuffle for train and validation sets
		drop_last = False  # Keep incomplete final batch
		batch_size = args.batch_size
	
	data_set = Data(
		datas = datas,
		size = [args.seq_len,args.label_len,args.pred_len],
		train_mode = args.train_mode,
		use_mark = args.use_mark,
		use_constant = args.use_constant)

	print(f'flag:{flag},data_length:{len(data_set)}')
	data_loader = DataLoader(
		data_set,
		batch_size=batch_size,
		shuffle = shuffle_flag,
		num_workers = args.num_workers,
		drop_last = drop_last)
	return data_set,data_loader

def processing_data(args):
	scaler = StandardScaler()
	data_path = args.data_path
	scale = args.scale
	use_constant =args.use_constant
	datas = {}
	for _flag in ['train','valid','test']:
		if use_constant:
			# Constant temperature data handling (not implemented - variable lengths/temperatures)
			path = os.path.join(data_path,'constant_temperature',_flag)
		else:
			path = os.path.join(data_path,'variable_temperature',_flag)
			files = os.listdir(path)
			files.sort(key=lambda x:int(x.split('-')[0].replace('T','')))  # Sort by temperature if not shuffling

			datas_list = []
			df_len = None
			col_len = None

			for file in files:
				path_tmp =os.path.join(path,file)
				df_tmp = pd.read_csv(path_tmp)
				if df_len is None:
					df_len = len(df_tmp)
				else:
					if df_len != len(df_tmp):
						raise('data no same length')
				if col_len is None:
					col_len = len(df_tmp.columns)  # Expected columns: ['D', 'T', 'B', 'N']
				else:
					if col_len != len(df_tmp.columns):
						raise('columns no same length')
				datas_list.append(df_tmp.values)
			datas[_flag] = np.concatenate(datas_list,axis=0)  # Combine files for standardization
			
			if scale:
				# Standardization: fit on training data, apply to all sets
				if _flag == 'train':
					scaler.fit(datas[_flag])
					joblib.dump(scaler,'scaler.model')
					datas[_flag] = scaler.transform(datas[_flag])
				else:
					datas[_flag] = scaler.transform(datas[_flag])
			datas[_flag] = datas[_flag].reshape(-1,df_len,col_len)
	return datas['train'],datas['valid'],datas['test']


def get_data(args):
	"""Get data loaders following Autoformer approach: fit scaler on training set, apply to all sets"""
	train_datas,valid_datas, test_datas = processing_data(args)
	# print('train_datas.shape',train_datas.shape)
	# print('valid_datas.shape',valid_datas.shape)
	# print('test_datas.shape',test_datas.shape)
	train_data,train_loader = data_provider(args,'train',train_datas)
	valid_data,valid_loader = data_provider(args,'valid',valid_datas)
	test_data,test_loader = data_provider(args,'test',test_datas)
	return train_data,valid_data,test_data,train_loader,valid_loader,test_loader



if __name__ == '__main__':
	# Command line arguments for shelf-life prediction
	import argparse
	parser = argparse.ArgumentParser(description='Data loader for aquatic products shelf-life prediction')
	# Basic data configuration
	parser.add_argument('--data',type=str,default='Food-SPV',help='choose a data')
	parser.add_argument('--data_path',type=str,default='./data/Food-SPV/',help='path of the ori data')
	parser.add_argument('--use_constant',type=bool,default=False,help='whether to use constant temperature data')

	# Model training configuration
	parser.add_argument('--train_mode',type=str,default='generation',choices=['generation','once'],help='Training mode: generation (sequential) vs once (one-shot)')
	parser.add_argument('--fine_tuning',type=bool,default=False,help='whether support fine tuning')
	parser.add_argument('--use_mark',type=bool,default=False,help='whether use mark embeddings for temperature and time')
	parser.add_argument('--scale',type=bool,default=True,help='whether to apply data standardization')

	# Model parameters
	parser.add_argument('--batch_size',type=int,default=6,help='batch size')
	parser.add_argument('--seq_len',type=int,default=8,help='input sequence length')
	parser.add_argument('--label_len',type=int,default=4,help='start token length')
	parser.add_argument('--pred_len',type=int,default=4,help='prediction sequence length')
	
	# Performance parameters
	parser.add_argument('--num_workers',type=int,default=4,help='data loader num workers')
	args = parser.parse_args()
	
	train_data,valid_data,test_data,train_loader,valid_loader,test_loader = get_data(args)
	if args.use_mark:
		for i,(batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
			print(batch_x)
			print(batch_y)
			print(batch_x_mark)
			print(batch_y_mark)			
			break
	else:
		for i,(batch_x,batch_y) in enumerate(train_loader):
			print(batch_x)
			print(batch_y)
			break

	#data_provider(args,'train')
	# data_provider(args,'test')
	# data_provider(args,'valid')