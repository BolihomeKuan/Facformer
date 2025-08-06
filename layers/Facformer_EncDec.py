import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvLayer(nn.Module):
	def __init__(self,c_in):
		super(ConvLayer,self).__init__()
		self.downConv = nn.Conv1d(
			in_channels=c_in,
			out_channels=c_in,
			kernel_size=3,
			padding=2,
			padding_mode='circular')
		self.norm = nn.BatchNorm1d(c_in)
		self.activation = nn.ELU()
		self.maxPool = nn.MaxPool1d(kernel_size=3,stride=2,padding=1)

	def forward(self,x):
		x = self.downConv(x.permute(0,2,1))  # Transpose for 1D convolution
		x = self.norm(x)
		x = self.activation(x)
		x = self.maxPool(x)
		x = x.transpose(1,2)
		return x


class EncoderLayer(nn.Module):
	def __init__(self,attention1,attention2,mark_attention,d_model,d_ff=None,dropout=0.1,activation="relu"):
		super(EncoderLayer,self).__init__()
		d_ff = d_ff or 4 * d_model  # Hidden size for feed-forward layers
		self.attention1 = attention1
		self.attention2 = attention2
		self.mark_attention = mark_attention
		# Use 1D convolution to implement feed-forward layers
		self.conv1 = nn.Conv1d(in_channels=d_model,out_channels=d_ff,kernel_size=1)  # Point-wise convolution
		self.conv2 = nn.Conv1d(in_channels=d_ff,out_channels=d_model,kernel_size=1)
		
		self.norm1 = nn.LayerNorm(d_model)  # Layer normalization
		self.norm2 = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)
		self.activation = F.relu if activation == 'relu' else F.gelu

	def forward(self,x,xm,xxm,attn_mask=None):
		# Process three streams through their respective attention mechanisms
		new_x, attn = self.attention1(x,x,x,attn_mask=attn_mask)
		new_xm,attn = self.mark_attention(xm,xm,xm,attn_mask=attn_mask)
		new_xxm,attn= self.attention2(xxm,xxm,xxm,attn_mask=attn_mask)

		# Basic embedding stream processing
		x = x + self.dropout(new_x)
		y = x = self.norm1(x)
		y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))  # Transpose for 1D convolution
		y = self.dropout(self.conv2(y).transpose(-1,1))

		# Mark embedding stream processing
		xm = xm + self.dropout(new_xm)
		ym = xm = self.norm1(xm)
		ym = self.dropout(self.activation(self.conv1(ym.transpose(-1,1))))  # Transpose for 1D convolution
		ym = self.dropout(self.conv2(ym).transpose(-1,1))

		# Total embedding stream processing
		xxm = xxm + self.dropout(new_xxm)
		yxm = xxm = self.norm1(xxm)
		yxm = self.dropout(self.activation(self.conv1(yxm.transpose(-1,1))))  # Transpose for 1D convolution
		yxm = self.dropout(self.conv2(yxm).transpose(-1,1))

		return self.norm2(x+y),self.norm2(xm+ym),self.norm2(xxm+yxm)

class Encoder(nn.Module):
	def __init__(self,attn_layers,conv_layers=None,norm_layer=None):
		super(Encoder,self).__init__()
		self.attn_layers = nn.ModuleList(attn_layers)
		self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
		self.norm = norm_layer

	def forward(self,x,xm,xxm,attn_mask=None):
		# Process through tri-parallel encoder layers
		# x: [batch_size, seq_len, d_model] 
		for attn_layer in self.attn_layers:
			x,xm,xxm = attn_layer(x,xm,xxm,attn_mask=attn_mask)

		if self.norm is not None:
			x = self.norm(x)
			xm = self.norm(xm)
			xxm = self.norm(xxm)
		return x,xm,xxm


class DecoderLayer(nn.Module):
	def __init__(self,self_attention1,cross_attention1,self_attention2,cross_attention2,
				mark_attention,mark_cross_attention,fac_attention,
				d_model,d_ff=None,dropout=0.1,activation='relu'):
		super(DecoderLayer,self).__init__()
		d_ff = d_ff or 4*d_model
		self.self_attention1 = self_attention1
		self.cross_attention1 = cross_attention1
		self.self_attention2 = self_attention2
		self.cross_attention2 = cross_attention2

		self.mark_attention = mark_attention
		self.mark_cross_attention = mark_cross_attention
		self.fac_attention = fac_attention

		# Feed-forward networks for different streams
		self.conv1 = nn.Conv1d(in_channels=d_model,out_channels=d_ff,kernel_size=1)
		self.conv2 = nn.Conv1d(in_channels=d_ff,out_channels=d_model,kernel_size=1)

		self.conv3 = nn.Conv1d(in_channels=d_model,out_channels=d_ff,kernel_size=1)
		self.conv4 = nn.Conv1d(in_channels=d_ff,out_channels=d_model,kernel_size=1)

		# Four-stream integration components
		# Stream 1: Historical Quality Stream processing
		self.hist_projection = nn.Linear(d_model, d_model)
		
		# Stream 2: Masked Attention Stream processing  
		self.mask_projection = nn.Linear(d_model, d_model)
		
		# Stream 3: Convolutional Stream for local pattern extraction
		self.conv_stream = nn.Sequential(
			nn.Conv1d(in_channels=d_model, out_channels=d_ff//2, kernel_size=3, padding=1),
			nn.ELU(),
			nn.Conv1d(in_channels=d_ff//2, out_channels=d_model, kernel_size=3, padding=1),
			nn.Dropout(dropout)
		)
		
		# Stream 4: Environmental Stream processing
		self.env_projection = nn.Linear(d_model, d_model)
		
		# Stream fusion mechanism with learnable weights
		self.stream_weights = nn.Parameter(torch.ones(4) / 4)  # Initialize equal weights
		self.fusion_gate = nn.Sequential(
			nn.Linear(d_model * 4, d_ff),
			nn.GELU(),
			nn.Linear(d_ff, d_model),
			nn.Sigmoid()
		)
		
		# Multi-scale temporal modeling
		self.temporal_conv1 = nn.Conv1d(d_model, d_model, kernel_size=1, padding=0)
		self.temporal_conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1) 
		self.temporal_conv5 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)

		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.norm3 = nn.LayerNorm(d_model)
		self.norm4 = nn.LayerNorm(d_model)  # Additional normalization for stream fusion
		self.dropout = nn.Dropout(dropout)
		self.activation = F.relu if activation =='relu' else F.gelu

	def forward(self,xp,xpm,xxpm,x_cross,xm_cross,xxm_cross,x_mask=None,cross_mask=None):
		"""
		Four-stream Integration Mechanism for Enhanced Shelf-life Prediction
		
		This method implements the core four-stream integration described in the paper:
		1. Historical Quality Stream (XP): Direct quality indicator projections from historical data
		2. Masked Attention Stream (XP1): Temporal pattern predictions from masked self-attention  
		3. Convolutional Stream (XP2): Local pattern extraction through 1D convolutions
		4. Environmental Stream (XP3): Temperature-aware predictions from Fac-Attention
		"""
		batch_size, seq_len, d_model = xp.shape
		
		# ==================== STREAM 1: Historical Quality Stream ====================
		# Direct projection of historical quality indicators with residual connection
		xp_processed = xp + self.dropout(self.self_attention1(xp,xp,xp,attn_mask=x_mask)[0])
		xp_processed = self.norm1(xp_processed)
		
		# Historical Quality Stream: Direct feature mapping from quality indicators
		historical_stream = self.hist_projection(xp_processed)
		historical_stream = self.dropout(F.gelu(historical_stream))
		
		# ==================== STREAM 2: Masked Attention Stream ====================
		# Temporal pattern extraction through cross-attention with encoder representations
		masked_attention_output = self.dropout(self.cross_attention1(xp_processed,x_cross,x_cross,attn_mask=cross_mask)[0])
		
		# Enhanced masked attention processing with multi-scale temporal modeling
		masked_stream_raw = self.mask_projection(masked_attention_output)
		
		# Multi-scale temporal convolution for capturing different time dependencies
		temp_input = masked_stream_raw.transpose(-1,1)  # [batch, d_model, seq_len]
		temp_conv1 = self.temporal_conv1(temp_input)
		temp_conv3 = self.temporal_conv3(temp_input) 
		temp_conv5 = self.temporal_conv5(temp_input)
		
		# Combine multi-scale features
		multi_scale_temp = (temp_conv1 + temp_conv3 + temp_conv5) / 3.0
		masked_attention_stream = multi_scale_temp.transpose(-1,1)  # Back to [batch, seq_len, d_model]
		masked_attention_stream = self.dropout(F.gelu(masked_attention_stream))
		
		# ==================== STREAM 3: Convolutional Stream ====================
		# Local pattern extraction through dedicated convolutional layers
		# Process mark embeddings for environmental context
		xpm_processed = xpm + self.dropout(self.mark_attention(xpm,xpm,xpm,attn_mask=None)[0])  # No mask for forward-looking
		xpm_processed = self.norm1(xpm_processed)
		
		# Convolutional stream for local pattern capture
		conv_input = xpm_processed.transpose(-1,1)  # [batch, d_model, seq_len]
		convolutional_stream = self.conv_stream(conv_input)
		convolutional_stream = convolutional_stream.transpose(-1,1)  # Back to [batch, seq_len, d_model]
		
		# ==================== STREAM 4: Environmental Stream ====================
		# Temperature-aware predictions through Fac-Attention mechanism
		# Critical: Uses forward-looking capability for environmental conditions
		environmental_attention = self.dropout(self.fac_attention(xpm_processed,xm_cross,x_cross,attn_mask=None)[0])
		environmental_stream = self.env_projection(environmental_attention)
		environmental_stream = self.dropout(F.gelu(environmental_stream))
		
		# ==================== ADVANCED FOUR-STREAM FUSION ====================
		# Normalize stream weights using softmax for adaptive importance
		normalized_weights = F.softmax(self.stream_weights, dim=0)
		
		# Weighted combination of four streams
		weighted_historical = normalized_weights[0] * historical_stream
		weighted_masked = normalized_weights[1] * masked_attention_stream  
		weighted_conv = normalized_weights[2] * convolutional_stream
		weighted_env = normalized_weights[3] * environmental_stream
		
		# Concatenate streams for fusion gate processing
		concatenated_streams = torch.cat([
			weighted_historical, weighted_masked, 
			weighted_conv, weighted_env
		], dim=-1)  # [batch, seq_len, d_model*4]
		
		# Adaptive fusion gate for dynamic stream integration
		fusion_weights = self.fusion_gate(concatenated_streams)  # [batch, seq_len, d_model]
		
		# Residual integration with attention-based weighting
		primary_prediction = weighted_historical + weighted_masked + weighted_conv + weighted_env
		fusion_enhanced = primary_prediction * fusion_weights
		
		# Final stream integration with residual connection and normalization
		integrated_prediction = self.norm4(fusion_enhanced + primary_prediction)
		
		# Feed-forward processing for final prediction refinement
		yxp = self.dropout(self.activation(self.conv1(integrated_prediction.transpose(-1,1))))
		yxp = self.dropout(self.conv2(yxp).transpose(-1,1))
		final_prediction = self.norm2(integrated_prediction + yxp)

		# ==================== AUXILIARY STREAM PROCESSING ====================
		# Mark embedding stream processing (maintained for compatibility)
		xpm_final = xpm_processed + self.dropout(self.mark_cross_attention(xpm_processed,xm_cross,xm_cross,attn_mask=cross_mask)[0])
		yxpm = xpm_final = self.norm2(xpm_final)
		yxpm = self.dropout(self.activation(self.conv3(yxpm.transpose(-1,1))))
		yxpm = self.dropout(self.conv4(yxpm).transpose(-1,1))

		# Total embedding stream processing (maintained for compatibility)
		xxpm = xxpm + self.dropout(self.self_attention2(xxpm,xxpm,xxpm,attn_mask=x_mask)[0])
		xxpm = self.norm1(xxpm)
		xxpm = xxpm + self.dropout(self.cross_attention2(xxpm,xxm_cross,xxm_cross,attn_mask=cross_mask)[0])
		yxxpm = xxpm = self.norm2(xxpm)
		yxxpm = self.dropout(self.activation(self.conv1(yxxpm.transpose(-1,1))))
		yxxpm = self.dropout(self.conv2(yxxpm).transpose(-1,1))

		return self.norm3(final_prediction),self.norm3(xpm_final+yxpm),self.norm3(xxpm+yxxpm)

class Decoder(nn.Module):
	def __init__(self,layers,norm_layer=None,projection1=None,projection2=None):
		super(Decoder,self).__init__()
		self.layers = nn.ModuleList(layers)
		self.norm = norm_layer
		self.projection1 = projection1
		self.projection2 = projection2

	def forward(self,xp,xpm,xxpm,x,xm,xxm,x_mask=None,cross_mask=None):
		# Process through decoder layers
		for layer in self.layers:
			xp,xpm,xxpm = layer(xp,xpm,xxpm,x,xm,xxm,x_mask=x_mask,cross_mask=cross_mask)
		
		# Final normalization combining basic and total streams
		if self.norm is not None:
			xp = self.norm((xp+xxpm))  # Combine basic and total predictions
		
		# Final projection layers
		if self.projection1 is not None:
			xp = self.projection1(xp)
			xp = self.projection2(xp)
		
		return xp






