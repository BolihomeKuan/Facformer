import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math

"""
Mark Feature Embedding Module

Each mark feature requires separate embedding processing:
- D (Days): Time-based embeddings with range encoding
- T (Temperature): Temperature-based embeddings with range encoding

Three embedding streams are generated:
- x_enc: Basic embedding from input features
- xm_enc: Mark embedding from environmental features  
- xxm_enc: Combined embedding from all features
"""
def compared_version(ver1,ver2):
	"""
	Compare two version strings
	Returns: -1 if ver1 < ver2, 1 if ver1 > ver2, True/False for equality comparison
	"""
	list1 = str(ver1).split(".")
	list2 = str(ver2).split(".")

	for i in range(len(list1)) if len(list1) < len(list2) else range(len(list2)):
		# Compare shorter version length
		if int(list1[i]) == int(list2[i]):
			pass
		elif int(list1[i]) < int(list2[i]):
			return -1
		else:
			return 1

	if len(list1) == len(list2):
		return True
	elif len(list1) < len(list2):
		return False
	else:
		return True

class PositionalEmbedding(nn.Module):
	"""Sinusoidal positional embedding for sequence position encoding"""
	def __init__(self,d_model,max_len=5000):
		super(PositionalEmbedding,self).__init__()
		pe = torch.zeros(max_len,d_model).float()
		pe.requires_grad = False
		position = torch.arange(0,max_len).float().unsqueeze(1)  # Expand on dimension 1
		div_term = (torch.arange(0,d_model,2).float() * -(math.log(10000.0)/d_model)).exp()
		pe[:,0::2] = torch.sin(position*div_term)
		pe[:,1::2] = torch.cos(position*div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe',pe)
	def forward(self,x):
		x = self.pe[:,:x.size(1)]
		return x



class TokenEmbedding(nn.Module):
	"""Token embedding using 1D convolution to transform input features"""
	def __init__(self,c_in,d_model):
		# c_in: input feature dimension
		super(TokenEmbedding,self).__init__()
		padding = 1 if compared_version(torch.__version__,'1.5.0') else 2
		self.tokenConv = nn.Conv1d(in_channels=c_in,out_channels=d_model,
									kernel_size=3,padding=padding,padding_mode='circular',bias=False)  # Dimension transformation
		for m in self.modules():
			if isinstance(m,nn.Conv1d):
				nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')
				# Kaiming initialization for ReLU-based activations

	def forward(self,x):
		x = self.tokenConv(x.permute(0,2,1)).transpose(1,2)
		return x


class FixedEmbedding(nn.Module):
	"""Fixed embedding with non-trainable weights using positional encoding"""
	def __init__(self,c_in,d_model):
		super(FixedEmbedding,self).__init__()
		w = torch.zeros(c_in,d_model).float()
		w.requires_grad = False
		position = torch.arange(0, c_in).float().unsqueeze(1)
		div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
		w[:,0::2] = torch.sin(position * div_term)
		if d_model > 1:
			w[:,1::2] = torch.cos(position * div_term[:w[:,1::2].shape[1]])

		self.emb = nn.Embedding(c_in ,d_model)
		self.emb.weight = nn.Parameter(w,requires_grad=False)

	def forward(self,x):
		return self.emb(x).detach()

class T_Embedding(nn.Module):
	def __init__(self,d_model,embed_type=None):
		super(T_Embedding,self).__init__()
		temperature_size = 60#[-19,+40]
		Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
		self.t_embed = Embed(temperature_size,d_model)

	def forward(self,x):
		x = x.long()
		x = self.t_embed(x[:,:,1])  # Extract temperature embedding ['D', 'T', 'B', 'N'] -> temperature only
		return x



class DataEmbedding(nn.Module):
	def __init__(self,c_in,d_model,embed_type=None,use_mark=None,dropout=0.1):
		super(DataEmbedding,self).__init__()

		self.value_embedding = TokenEmbedding(c_in=c_in,d_model=d_model)
		self.position_embedding = PositionalEmbedding(d_model=d_model)
		self.temporal_embedding = T_Embedding(d_model,embed_type)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self,x):
		# Combine value and positional embeddings
		x0 = self.value_embedding(x)
		x1 = self.position_embedding(x)
		x= x0 + x1  # Position embedding broadcasts across batch dimension
		return self.dropout(x)

class DataEmbedding_wo_pos(nn.Module):
	def __init__(self,c_in,d_model,embed_type='fixed',use_mark='h',dropout=0.1):
		super(DataEmbedding_wo_pos,self).__init__()

		self.value_embedding = TokenEmbedding(c_in=c_in,d_model=d_model)
		self.position_embedding = PositionalEmbedding(d_model=d_model)
		if embed_type != 'embed_T':
			self.temporal_embedding = T_Embedding(d_model=d_model,embed_type=embed_type)
		
		self.dropout = nn.Dropout(p=dropout)

	def forward(self,x,x_mark):
		x = self.value_embedding(x) + self.temporal_embedding(x_mark)
		return self.dropout(x)



class PositionalEmbedding_mark(nn.Module):
	def __init__(self,d_model,max_len=5000):
		super(PositionalEmbedding_mark,self).__init__()
		pe = torch.zeros(max_len,d_model).float()
		pe.requires_grad = False
		position = torch.arange(0,max_len).float().unsqueeze(1)  # Expand on dimension 1
		div_term = (torch.arange(0,d_model,2).float() * -(math.log(10000.0)/d_model)).exp()
		pe[:,0::2] = torch.sin(position*div_term)
		pe[:,1::2] = torch.cos(position*div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe',pe)
	def forward(self,x):
		return self.pe[:,:x.size(1)]


class TokenEmbedding_mark(nn.Module):
	"""Token embedding for mark features using 1D convolution"""
	def __init__(self,c_in,d_model):
		# c_in: input feature dimension
		super(TokenEmbedding_mark,self).__init__()
		padding = 1 if compared_version(torch.__version__,'1.5.0') else 2
		self.tokenConv = nn.Conv1d(in_channels=c_in,out_channels=d_model,
			kernel_size=3,padding=padding,padding_mode='circular',bias=False)  # Dimension transformation
		for m in self.modules():
			if isinstance(m,nn.Conv1d):
				nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')
				# Kaiming initialization for ReLU-based activations

	def forward(self,x):
		x = self.tokenConv(x.permute(0,2,1)).transpose(1,2)
		return x

class FixedEmbedding_mark(nn.Module):
	"""Fixed embedding for mark features with non-trainable positional weights"""
	def __init__(self,c_in,d_model):
		super(FixedEmbedding_mark,self).__init__()
		w = torch.zeros(c_in,d_model).float()
		w.requires_grad = False
		position = torch.arange(0, c_in).float().unsqueeze(1)
		div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
		w[:,0::2] = torch.sin(position * div_term)
		if d_model > 1:
			w[:,1::2] = torch.cos(position * div_term[:w[:,1::2].shape[1]])

		self.emb = nn.Embedding(c_in ,d_model)
		self.emb.weight = nn.Parameter(w,requires_grad=False)

	def forward(self,x):
		return self.emb(x).detach()

class T_Embedding_mark(nn.Module):
	def __init__(self,d_model,embed_type=None):
		super(T_Embedding_mark,self).__init__()
		temperature_size = 60  # Temperature range [-19,+40]
		Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
		self.t_embed = Embed(temperature_size,d_model)

	def forward(self,x):
		x = x.long()
		x = self.t_embed(x[:,:,1])  # Extract temperature embedding from mark tensor
		return x



class DataEmbedding_mark(nn.Module):
	def __init__(self,c_in,d_model,embed_type=None,use_mark=None,dropout=0.1):
		super(DataEmbedding_mark,self).__init__()

		self.value_embedding = TokenEmbedding(c_in=c_in,d_model=d_model)
		self.position_embedding = PositionalEmbedding(d_model=d_model)
		if embed_type == 'embed_T':
			self.temporal_embedding = T_Embedding(d_model=d_model,embed_type=embed_type)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self,x,x_mark):
		x= self.value_embedding(x) + self.temporal_embedding(x_mark)+self.position_embedding(x)
		return self.dropout(x)

class DataEmbedding_wo_pos_mark(nn.Module):
	def __init__(self,c_in,d_model,embed_type='fixed',use_mark='h',dropout=0.1):
		super(DataEmbedding_wo_pos_mark,self).__init__()

		self.value_embedding = TokenEmbedding(c_in=c_in,d_model=d_model)
		self.position_embedding = PositionalEmbedding(d_model=d_model)
		if embed_type != 'embed_T':
			self.temporal_embedding = T_Embedding(d_model=d_model,embed_type=embed_type)
		
		self.dropout = nn.Dropout(p=dropout)

	def forward(self,x,x_mark):
		x = self.value_embedding(x) + self.temporal_embedding(x_mark)
		return self.dropout(x)