"""
Facformer: Future Annotation Collaborative Transformer

This model implements a novel architecture for time series prediction under variable 
environmental conditions. Key innovations include:

1. Fac-Attention Mechanism: Enables forward-looking perception of environmental conditions
2. Mark Embedding: Encodes dynamic temperature patterns into feature representations  
3. Multi-stream Integration: Combines multiple prediction streams for robust forecasting

Architecture Overview:
- Basic Embedding: Standard feature encoding with positional information
- Mark Embedding: Environmental context encoding (temperature, time)
- Total Embedding: Combined representation for comprehensive modeling

The model is specifically designed for shelf-life prediction of aquatic products
under variable temperature storage conditions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers.Fac_Embed import DataEmbedding

from layers.Facformer_EncDec import Decoder,DecoderLayer,Encoder,EncoderLayer,ConvLayer
from layers.Fac_Attention import FullAttention,FacAttention,AttentionLayer


class Model(nn.Module):
	"""
	Facformer Model Implementation
	
	A Transformer-based architecture specifically designed for shelf-life prediction
	of aquatic products under variable temperature conditions.
	
	Args:
		configs: Configuration object containing model hyperparameters
	"""
	def __init__(self, configs):
		super(Model, self).__init__()
		self.pred_len = configs.pred_len
		self.output_attention = configs.output_attention  # Whether to output attention weights

		# Embedding layers for encoder and decoder
		self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.dropout)
		self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.dropout)

		# Tri-parallel encoder architecture
		# Each encoder branch handles different information streams:
		# 1. Quality-specific features (Basic Embedding)
		# 2. Temperature dynamics (Mark Embedding)  
		# 3. Comprehensive representation (Total Embedding)
		self.encoder = Encoder(
			[EncoderLayer(
				AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
						output_attention=configs.output_attention), configs.d_model, configs.n_heads),
				AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
						output_attention=configs.output_attention), configs.d_model, configs.n_heads),
				AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
						output_attention=configs.output_attention), configs.d_model, configs.n_heads),
				configs.d_model,
				configs.d_ff,
				dropout=configs.dropout,
				activation=configs.activation
				) for l in range(configs.e_layers)],
			norm_layer=torch.nn.LayerNorm(configs.d_model))

		# Decoder with Fac-Attention mechanism
		# Implements differential masking strategy:
		# - Basic/Total streams: Use causal masking for historical data only
		# - Mark stream: No masking for forward-looking environmental perception
		self.decoder = Decoder(
			[DecoderLayer(
				# Basic stream: Masked self-attention for historical quality data
				AttentionLayer(FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
					configs.d_model, configs.n_heads),
				# Basic stream: Cross-attention with encoder
				AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
					configs.d_model, configs.n_heads),
				# Total stream: Masked self-attention for comprehensive representation
				AttentionLayer(FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
					configs.d_model, configs.n_heads),
				# Total stream: Cross-attention with encoder
				AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
					configs.d_model, configs.n_heads),
				# Mark stream: Self-attention WITHOUT masking for forward-looking perception
				AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
					configs.d_model, configs.n_heads),
				# Mark stream: Cross-attention with encoder mark stream
				AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
					configs.d_model, configs.n_heads),
				# Fac-Attention: Temperature-aware cross-attention (Mark→Basic)
				AttentionLayer(FacAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
					configs.d_model, configs.n_heads),
				configs.d_model,
				configs.d_ff,
				dropout=configs.dropout,
				activation=configs.activation,
				) for l in range(configs.d_layers)],
			norm_layer=torch.nn.LayerNorm(configs.d_model),
			projection1=nn.Linear(configs.d_model, configs.d_model*4, bias=True),
			projection2=nn.Linear(configs.d_model*4, configs.c_out, bias=True)
			)


	def forward(self, x_enc, x_enc_mark, x_dec, x_dec_mark,
		enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
		"""
		Forward pass of Facformer model
		
		Implements differential masking strategy for temperature-aware prediction:
		- Basic/Total streams: Use causal masking for historical data patterns
		- Mark stream: No masking for forward-looking environmental perception
		
		This design enables the model to incorporate future temperature information
		(e.g., forecasted cold-chain conditions) while maintaining causal constraints
		for quality indicator predictions.
		
		Args:
			x_enc: Encoder input features [batch_size, seq_len, features]
			x_enc_mark: Encoder mark features (temperature, time) [batch_size, seq_len, mark_features]
			x_dec: Decoder input features [batch_size, seq_len, features]
			x_dec_mark: Decoder mark features [batch_size, seq_len, mark_features]
			enc_self_mask: Encoder self-attention mask (not used for Mark streams)
			dec_self_mask: Decoder self-attention mask (applied selectively)
			dec_enc_mask: Decoder-encoder attention mask (applied selectively)
			
		Returns:
			Predictions for the next pred_len time steps [batch_size, pred_len, features]
		"""
		# Generate embeddings for encoder inputs
		# Returns Basic, Mark, and Total embeddings
		x_enc_out, xm_enc_out, xxm_enc_out = self.enc_embedding(x_enc, x_enc_mark)
		
		# Generate embeddings for decoder inputs
		x_dec_out, xm_dec_out, xxm_dec_out = self.dec_embedding(x_dec, x_dec_mark)
		
		# Pass through tri-parallel encoder
		enc_out, xm_enc_out, xxm_enc_out = self.encoder(x_enc_out, xm_enc_out, xxm_enc_out)
		
		# Decode with Fac-Attention mechanism
		dec_out = self.decoder(x_dec_out, xm_dec_out, xxm_dec_out, enc_out, xm_enc_out, xxm_enc_out, 
							  x_mask=dec_self_mask, cross_mask=dec_self_mask)
		
		# Return predictions for the prediction horizon
		return dec_out[:, -self.pred_len:, :]  # [batch_size, pred_len, features]



