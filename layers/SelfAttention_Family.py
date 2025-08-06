import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import math
from math import sqrt

from utils.masking import TriangularCausalMask#, ProbMask
# from reformer_pytorch import LSHSelfAttention




class FullAttention(nn.Module):
	def __init__(self,mask_flag=True,factor=5,scale=None,attention_dropout=0.1,output_attention=False):
		super(FullAttention,self).__init__()
		self.scale = scale  # Normalization factor
		self.mask_flag = mask_flag
		self.output_attention = output_attention
		self.dropout = nn.Dropout(attention_dropout)

	def forward(self,queries,keys,values,attn_mask):
		B,L,H,E = queries.shape  # [batch_size, seq_len, n_heads, d_model//n_heads]
		_,S,_,D = values.shape
		scale = self.scale or 1./sqrt(E)

		scores = torch.einsum("blhe,bshe->bhls",queries,keys)  # Einstein summation for efficient computation

		if self.mask_flag:
			if attn_mask is None:
				attn_mask = TriangularCausalMask(B,L,device=queries.device)  # [B,1,L,L]
			scores.masked_fill_(attn_mask.mask,-np.inf)  # Apply causal mask: {B,H,L,L}

		A = self.dropout(torch.softmax(scale * scores,dim=-1))
		V = torch.einsum("bhls,bshd->blhd",A,values)

		if self.output_attention:
			return (V.contiguous(),A)
		else:
			return (V.contiguous(),None)


class AttentionLayer(nn.Module):
	def __init__(self,attention,d_model,n_heads,d_keys=None,d_values=None):
		super(AttentionLayer,self).__init__()
		d_keys = d_keys or (d_model//n_heads)  # Must be divisible
		d_values = d_values or (d_model//n_heads)
		
		self.inner_attention = attention
		self.query_projection = nn.Linear(d_model,d_keys*n_heads)
		self.key_projection = nn.Linear(d_model,d_keys*n_heads)
		self.value_projection = nn.Linear(d_model,d_values*n_heads)
		self.out_projection = nn.Linear(d_values*n_heads,d_model)
		self.n_heads = n_heads

	def forward(self,queries,keys,values,attn_mask):
		# print('x==x==x:',queries==keys,keys==values)
		B, L, _ = queries.shape
		_, S, _ = keys.shape
		H = self.n_heads
		# print('attn_mask::',attn_mask)
		# print('queries.shape:before',queries.shape)
		# print('B,L,H',B,L,H)
		queries = self.query_projection(queries).view(B,L,H,-1)
		keys = self.key_projection(keys).view(B,S,H,-1)
		values = self.value_projection(values).view(B,S,H,-1)
		# print('queries.shape:after',queries.shape)
		out,attn = self.inner_attention(queries,keys,values,attn_mask)
		# print('out.shape',out.shape)
		# print('attn.shape::',attn.shape)
		out = out.view(B,L,-1)
		# print('out.shape::',out.shape)
		return self.out_projection(out),attn




