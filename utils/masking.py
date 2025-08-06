"""
Attention Masking Utilities for Facformer

This module provides masking utilities for attention mechanisms in the Facformer model,
including causal masking for decoder self-attention and probabilistic masking for
efficient attention computation.
"""

import torch

class TriangularCausalMask():
	"""
	Triangular causal mask for autoregressive attention
	
	Creates a upper triangular mask to prevent attention to future positions
	in sequence generation tasks.
	
	Args:
		B: Batch size
		L: Sequence length  
		device: Device to place the mask tensor
	"""
	def __init__(self,B,L,device='cpu'):
		mask_shape = [B,1,L,L]
		with torch.no_grad():
			self._mask = torch.triu(torch.ones(mask_shape,dtype=torch.bool),diagonal=1).to(device)

	@property
	def mask(self):
		return self._mask
	
class ProbMask():
	"""
	Probabilistic mask for efficient attention computation
	
	Used in models like Informer for sparse attention patterns.
	
	Args:
		B: Batch size
		H: Number of attention heads
		L: Sequence length
		index: Index tensor for mask selection
		scores: Attention scores tensor
		device: Device to place the mask tensor
	"""
	def __init__(self,B,H,L,index,scores,device='cpu'):
		_mask = torch.ones(L,scores.shape[-1],dtype=torch.bool).to(device).triu(1)
		_mask_ex = _mask[None,None,:].expand(B,H,L,scores.shape[-1])
		indicator = _mask_ex[torch.arange(B)[:,None,None],
							torch.arange(H)[None,:,None],
							index,:].to(device)
		self._mask = indicator.view(scores.shape).to(device)

	@property
	def mask(self):
		return self._mask