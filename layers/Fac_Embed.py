import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math

"""
Fac-Embed: Mark Embedding Module for Facformer

This module implements the Mark Embedding functionality, which is a key innovation
of the Facformer architecture. It enables the model to encode environmental context
(temperature and time) separately from quality indicators.

Main Approaches:
A. Mark-based Embedding:
   1. Extract positional (day) and temperature information as separate marks
   2. Fixed or learnable positional and temporal embeddings
   3. Temperature and day information support future known values
   4. Requires integer encoding for embedding layers

B. Direct Feature Embedding:
   1. Process all features together through transformer
   2. Apply only positional encoding

The Mark Embedding approach enables better environmental awareness and
temperature-sensitive predictions for cold-chain applications.
"""
def compared_version(ver1, ver2):
	"""
	Compare two version strings
	
	Args:
		ver1: First version string
		ver2: Second version string
		
	Returns:
		-1 if ver1 < ver2, 1 if ver1 > ver2, True if ver1 == ver2, False otherwise
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
	"""
	Sinusoidal Positional Embedding
	
	Generates fixed sinusoidal embeddings to encode positional information
	in the sequence. This helps the model understand temporal order.
	
	Args:
		d_model: Model dimension
		max_len: Maximum sequence length
	"""
	def __init__(self, d_model, max_len=5000):
		super(PositionalEmbedding, self).__init__()
		pe = torch.zeros(max_len, d_model).float()
		pe.requires_grad = False
		position = torch.arange(0, max_len).float().unsqueeze(1)  # Expand on dimension 1
		div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0)/d_model)).exp()
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)
		
	def forward(self, x):
		"""
		Args:
			x: Input tensor [batch_size, seq_len, features]
		Returns:
			Positional embeddings [batch_size, seq_len, d_model]
		"""
		return self.pe[:, :x.size(1)]



class TokenEmbedding(nn.Module):
	"""
	Token Embedding using 1D Convolution
	
	Transforms input features into higher-dimensional embeddings using
	1D convolution. This serves as the value embedding in the architecture.
	
	Args:
		c_in: Input feature dimension
		d_model: Model dimension (output dimension)
	"""
	def __init__(self, c_in, d_model):
		super(TokenEmbedding, self).__init__()
		padding = 1 if compared_version(torch.__version__, '1.5.0') else 2
		self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
									kernel_size=3, padding=padding, padding_mode='circular', bias=False)
		
		# Initialize weights using Kaiming initialization
		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

	def forward(self, x):
		"""
		Args:
			x: Input tensor [batch_size, seq_len, c_in]
		Returns:
			Token embeddings [batch_size, seq_len, d_model]
		"""
		x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
		return x


class FixedEmbedding(nn.Module):
	"""Fixed embedding layer with non-trainable weights using positional encoding"""
	def __init__(self,c_in,d_model):
		super(FixedEmbedding,self).__init__()
		w = torch.zeros(c_in,d_model).float()
		w.required_grad = False
		w[:,0::2] = torch.sin(position * div_term)
		w[:,1::2] = torch.cos(position * div_term)

		self.emb = nn.Embedding(c_in ,d_model)
		self.emb.weight = nn.Parameter(w,required_grad=False)

	def forward(self,x):
		return self.emb(x).detach()

class T_Embedding(nn.Module):
	def __init__(self,d_model,embed_type=None):
		super(T_Embedding,self).__init__()
		temperature_size = 60#[-19,+40]
		Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
		self.t_embed = Embed(temperature_size,d_model)

	def forward(self,x_mark):
		x_mark = x_mark.long()[:,:,1:2]
		x_mark = self.t_embed(x_mark)  # Temperature embedding from mark tensor
		return x_mark

class T_scale_Embedding(nn.Module):
	"""
	Temperature Scale Embedding
	
	Embeds temperature information using 1D convolution. This is part of the
	Mark Embedding system that enables temperature-aware predictions.
	
	Args:
		c_in: Input channels (should be 1 for temperature)
		d_model: Model dimension
		embed_type: Embedding type (not used in this implementation)
	"""
	def __init__(self, c_in, d_model, embed_type=None):
		super(T_scale_Embedding, self).__init__()
		padding = 1 if compared_version(torch.__version__, '1.5.0') else 2
		self.t_embed = nn.Conv1d(in_channels=c_in, out_channels=d_model,
									kernel_size=3, padding=padding, padding_mode='circular', bias=False)
	
	def forward(self, x_mark):
		"""
		Args:
			x_mark: Mark tensor containing [Day, Temperature] [batch_size, seq_len, 2]
		Returns:
			Temperature embeddings [batch_size, seq_len, d_model]
		"""
		x_mark = x_mark[:, :, 1:2]  # Extract temperature column
		x_mark = self.t_embed(x_mark.permute(0, 2, 1)).transpose(1, 2)
		return x_mark

class D_Embedding(nn.Module):
	def __init__(self,d_model,embed_type=None):
		super(D_Embedding,self).__init__()
		temperature_size = 365#[-19,+40]
		Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
		self.t_embed = Embed(temperature_size,d_model)

	def forward(self,x_mark):
		x_mark = x_mark.long()[:,:,0:1]
		x_mark = self.t_embed(x_mark)#['D', 'T']#, 'B', 'N']#4维变3维，只取T相关的emb，4维数组变3维数组
		return x_mark

class D_scale_Embedding(nn.Module):
	def __init__(self,c_in,d_model,embed_type=None):
		super(D_scale_Embedding,self).__init__()
		padding = 1 if compared_version(torch.__version__,'1.5.0') else 2
		# Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
		self.t_embed = nn.Conv1d(in_channels=c_in,out_channels=d_model,
									kernel_size=3,padding=padding,padding_mode='circular',bias=False)#将c_in维升维到d_model维	def forward(self,x_mark):

	def forward(self,x_mark):
		x_mark = x_mark[:,:,0:1]
		x_mark = self.t_embed(x_mark.permute(0,2,1)).transpose(1,2)#['D', 'T']#, 'B', 'N']#4维变3维，只取T相关的emb，4维数组变3维数组
		return x_mark


class DataEmbedding(nn.Module):
	"""
	Main Data Embedding Module for Facformer
	
	Combines multiple embedding types to create the three embedding streams:
	1. Basic Embedding: Value + Positional embeddings
	2. Mark Embedding: Temperature + Date embeddings  
	3. Total Embedding: All embeddings combined
	
	This tri-embedding approach enables the model to process quality indicators
	and environmental factors both separately and jointly.
	
	Args:
		c_in: Input feature dimension
		d_model: Model dimension
		embed_type: Embedding type (not used in current implementation)
		use_mark: Whether to use mark embeddings (not used in current implementation)
		dropout: Dropout rate
	"""
	def __init__(self, c_in, d_model, embed_type=None, use_mark=None, dropout=0.1):
		super(DataEmbedding, self).__init__()
		self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
		self.position_embedding = PositionalEmbedding(d_model=d_model)
		self.temporal_embedding = T_scale_Embedding(1, d_model)
		self.date_embedding = D_scale_Embedding(1, d_model)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x, x_mark):
		"""
		Generate three types of embeddings for Facformer
		
		Args:
			x: Input features [batch_size, seq_len, features] - quality indicators
			x_mark: Mark features [batch_size, seq_len, 2] - [Day, Temperature]
			
		Returns:
			basic_embed: Basic embedding (value + position)
			mark_embed: Mark embedding (temperature + date)  
			total_embed: Total embedding (all combined)
		"""
		# Generate individual embeddings
		value_embed = self.value_embedding(x)           # Quality indicator embeddings
		position_embed = self.position_embedding(x)     # Positional embeddings
		temp_embed = self.temporal_embedding(x_mark)     # Temperature embeddings
		date_embed = self.date_embedding(x_mark)         # Date embeddings

		# Combine embeddings into three streams
		basic_embed = value_embed + position_embed       # Basic: quality + position
		mark_embed = temp_embed + date_embed             # Mark: temperature + date
		total_embed = value_embed + position_embed + mark_embed  # Total: all combined
		
		return self.dropout(basic_embed), self.dropout(mark_embed), self.dropout(total_embed)

class DataEmbedding_wo_pos(nn.Module):
	def __init__(self,c_in,d_model,embed_type='fixed',use_mark='h',dropout=0.1):
		super(DataEmbedding_wo_pos,self).__init__()

		self.value_embedding = TokenEmbedding(c_in=c_in,d_model=d_model)
		self.position_embedding = PositionalEmbedding(d_model=d_model)
		if embed_type != 'embed_T':
			self.temporal_embedding = T_Embedding(d_model=d_model,embed_type=embed_type,freq=freq)
		
		self.dropout = nn.Dropout(p=dropout)

	def forward(self,x,x_mark):
		x = self.value_embedding(x) + self.temporal_embedding(x_mark)
		return self.dropout(x)



class PositionalEmbedding_mark(nn.Module):
	def __init__(self,d_model,max_len=5000):
		super(PositionalEmbedding,self).__init__()
		pe = torch.zeros(max_len,d_model).float()
		re.required_grad= False
		position = torch.arange(0,max_len).float().unsqueeze(1)#在第1维上拓展
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

class FixedEmbedding_mark(nn.Module):
	"""Fixed embedding for mark features with non-trainable positional weights"""
	def __init__(self,c_in,d_model):
		super(FixedEmbedding,self).__init__()
		w = torch.zeros(c_in,d_model).float()
		w.required_grad = False
		w[:,0::2] = torch.sin(position * div_term)
		w[:,1::2] = torch.cos(position * div_term)

		self.emb = nn.Embedding(c_in ,d_model)
		self.emb.weight = nn.Parameter(w,required_grad=False)

	def forward(self,x):
		return self.emb(x).detach()

class T_Embedding_mark(nn.Module):
	def __init__(self,d_model,embed_type=None):
		super(TemporalEmbedding,self).__init__()
		temperature_size = 60#[-19,+40]
		Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
		self.t_embed = Embed(minute_size,d_model)


	def forward(self,x):
		x = x.long()
		x = self.t_embed(x[:,:,1])  # Extract temperature embedding from mark tensor
		return x



class DataEmbedding_mark(nn.Module):
	def __init__(self,c_in,d_model,embed_type=None,use_mark=None,dropout=0.1):
		super(DataEmbedding,self).__init__()

		self.value_embedding = TokenEmbedding(c_in=c_in,d_model=d_model)
		self.position_embedding = PositionalEmbedding(d_model=d_model)
		if embed_type == 'embed_T':
			self.temporal_embedding = T_Embedding(d_model=d_model,embed_type=embed_type,freq=freq)
		self.dropout = nn.Dropput(p=dropout)

	def forward(self,x,x_mark):
		x= self.value_embedding(x) + self.temporal_embedding(x_mark)+self.position_embedding(x)
		return self.dropout(x)

class DataEmbedding_wo_pos_mark(nn.Module):
	def __init__(self,c_in,d_model,embed_type='fixed',use_mark='h',dropout=0.1):
		super(DataEmbedding_wo_pos,self).__init__()

		self.value_embedding = TokenEmbedding(c_in=c_in,d_model=d_model)
		self.position_embedding = PositionalEmbedding(d_model=d_model)
		if embed_type != 'embed_T':
			self.temporal_embedding = T_Embedding(d_model=d_model,embed_type=embed_type,freq=freq)
		
		self.dropout = nn.Dropput(p=dropout)

	def forward(self,x,x_mark):
		x = self.value_embedding(x) + self.temporal_embedding(x_mark)
		return self.dropout(x)