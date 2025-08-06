import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import math
from math import sqrt

from utils.masking import TriangularCausalMask


class FullAttention(nn.Module):
	"""
	Full Attention Mechanism for Facformer
	
	Implements scaled dot-product attention with optional masking for causal modeling.
	This is the core attention mechanism used in both encoder and decoder layers.
	
	Args:
		mask_flag: Whether to apply causal masking
		factor: Attention factor (not used in full attention)
		scale: Custom scaling factor (if None, uses 1/sqrt(d_k))
		attention_dropout: Dropout rate for attention weights
		output_attention: Whether to return attention weights
	"""
	def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
		super(FullAttention, self).__init__()
		self.scale = scale  # Normalization factor
		self.mask_flag = mask_flag
		self.output_attention = output_attention
		self.dropout = nn.Dropout(attention_dropout)

	def forward(self, queries, keys, values, attn_mask):
		"""
		Forward pass of Full Attention
		
		Args:
			queries: Query matrix [batch_size, seq_len, n_heads, d_k]
			keys: Key matrix [batch_size, seq_len, n_heads, d_k]
			values: Value matrix [batch_size, seq_len, n_heads, d_v]
			attn_mask: Attention mask
			
		Returns:
			Attention output and optionally attention weights
		"""
		B, L, H, E = queries.shape  # [batch_size, seq_len, n_heads, d_model//n_heads]
		_, S, _, D = values.shape
		scale = self.scale or 1./sqrt(E)
		
		# Compute attention scores using Einstein summation
		scores = torch.einsum("blhe,bshe->bhls", queries, keys)

		if self.mask_flag:
			if attn_mask is None:
				attn_mask = TriangularCausalMask(B, L, device=queries.device)  # [B, 1, L, L]
			scores.masked_fill_(attn_mask.mask, -np.inf)  # scores: {B, H, L, L}

		# Apply softmax and dropout to attention weights
		A = self.dropout(torch.softmax(scale * scores, dim=-1))
		
		# Apply attention weights to values
		V = torch.einsum("bhls,bshd->blhd", A, values)

		if self.output_attention:
			return (V.contiguous(), A)
		else:
			return (V.contiguous(), None)


class FacAttention(nn.Module):
	"""
	Fac-Attention Mechanism for Forward-Looking Environmental Perception
	
	This is the core innovation of Facformer, enabling the model to incorporate
	future environmental conditions (temperature forecasts) for robust prediction
	under variable cold-chain conditions.
	
	Key Features:
	- No causal masking for Mark Embedding branch (forward-looking)
	- Standard causal masking for Basic and Total Embedding branches
	- Temperature-aware cross-attention mechanism
	
	Args:
		mask_flag: Whether to apply causal masking (False for Mark branch)
		factor: Attention factor (not used in full attention)
		scale: Custom scaling factor
		attention_dropout: Dropout rate for attention weights
		output_attention: Whether to return attention weights
	"""
	def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
		super(FacAttention, self).__init__()
		self.scale = scale
		self.mask_flag = mask_flag  # Mark embedding uses mask_flag=False for forward-looking
		self.output_attention = output_attention
		self.dropout = nn.Dropout(attention_dropout)

	def forward(self, queries, keys, values, attn_mask):
		"""
		Forward pass of Fac-Attention
		
		Args:
			queries: Query matrix from Mark Embedding [batch_size, seq_len, n_heads, d_k]
			keys: Key matrix from encoder Mark stream [batch_size, seq_len, n_heads, d_k]
			values: Value matrix from encoder Basic stream [batch_size, seq_len, n_heads, d_v]
			attn_mask: Attention mask (ignored for Mark branch)
			
		Returns:
			Temperature-aware attention output and optionally attention weights
		"""
		B, L, H, E = queries.shape
		_, S, _, D = values.shape
		scale = self.scale or 1./sqrt(E)
		
		# Compute attention scores using Einstein summation
		scores = torch.einsum("blhe,bshe->bhls", queries, keys)
		
		# Critical: Mark Embedding branch does NOT use causal masking
		# This enables forward-looking perception of environmental conditions
		if self.mask_flag and attn_mask is not None:
			# Only apply mask if explicitly required (not for Mark branch)
			scores.masked_fill_(attn_mask.mask, -np.inf)
		
		# Apply softmax and dropout to attention weights
		A = self.dropout(torch.softmax(scale * scores, dim=-1))
		
		# Apply attention weights to values (cross-attention with Basic stream)
		V = torch.einsum("bhls,bshd->blhd", A, values)
		
		if self.output_attention:
			return (V.contiguous(), A)
		else:
			return (V.contiguous(), None)


class AttentionLayer(nn.Module):
	"""
	Multi-Head Attention Layer for Facformer
	
	Implements multi-head attention with linear projections for queries, keys, and values.
	This layer is used throughout the encoder and decoder architectures.
	
	Args:
		attention: Attention mechanism (e.g., FullAttention, FacAttention)
		d_model: Model dimension
		n_heads: Number of attention heads
		d_keys: Dimension of keys (defaults to d_model//n_heads)
		d_values: Dimension of values (defaults to d_model//n_heads)
	"""
	def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
		super(AttentionLayer, self).__init__()
		d_keys = d_keys or (d_model//n_heads)  # Must be divisible
		d_values = d_values or (d_model//n_heads)
		
		self.inner_attention = attention
		self.query_projection = nn.Linear(d_model, d_keys*n_heads)
		self.key_projection = nn.Linear(d_model, d_keys*n_heads)
		self.value_projection = nn.Linear(d_model, d_values*n_heads)
		self.out_projection = nn.Linear(d_values*n_heads, d_model)
		self.n_heads = n_heads

	def forward(self, queries, keys, values, attn_mask):
		"""
		Forward pass of Multi-Head Attention
		
		Args:
			queries: Query tensor [batch_size, seq_len, d_model]
			keys: Key tensor [batch_size, seq_len, d_model]
			values: Value tensor [batch_size, seq_len, d_model]
			attn_mask: Attention mask
			
		Returns:
			Output tensor and attention weights
		"""
		B, L, _ = queries.shape
		_, S, _ = keys.shape
		H = self.n_heads
		
		# Project and reshape for multi-head attention
		queries = self.query_projection(queries).view(B, L, H, -1)
		keys = self.key_projection(keys).view(B, S, H, -1)
		values = self.value_projection(values).view(B, S, H, -1)
		
		# Apply attention mechanism
		out, attn = self.inner_attention(queries, keys, values, attn_mask)
		
		# Reshape and project output
		out = out.view(B, L, -1)
		return self.out_projection(out), attn




