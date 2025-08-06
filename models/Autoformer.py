import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from layers.Embed import DataEmbedding,DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation,AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder,Decoder,EncoderLayer,DecoderLayer,my_Layernorm,series_decomp

class Model(nn.Module):
	#
	def __init__(self,configs):
		super(Model,self).__init__()
		self.seq_len = configs.seq_len
		self.label_len = configs.label_len
		self.pred_len = configs.pred_len
		self.output_attention = configs.output_attention

		#Decomp
		kernel_size = configs.moving_avg#default=25
		self.decomp = series_decomp(kernel_size)

		self.enc_embedding = DataEmbedding(configs.enc_in,configs.d_model,configs.embed,configs.dropout)
		self.dec_embedding = DataEmbedding(configs.enc_in,configs.d_model,configs.embed,configs.dropout)

		self.encoder = Encoder(
			[EncoderLayer(
				AutoCorrelationLayer(
					AutoCorrelation(False,configs.factor,attention_dropout=configs.dropout,output_attention= configs.output_attention),
							configs.d_model,configs.n_heads),
				configs.d_model,
				configs.d_ff,
				moving_avg = configs.moving_avg,
				dropout=configs.dropout,
				activation=configs.activation) for l in range(configs.e_layers)])

		self.decoder = Decoder(
			[
				DecoderLayer(
					AutoCorrelationLayer(
						AutoCorrelation(True,configs.factor,attention_dropout=configs.dropout,
									output_attention=False),
						configs.d_model,configs.n_heads),
					AutoCorrelationLayer(
						AutoCorrelation(False,configs.factor,attention_dropout=configs.dropout,
									output_attention=False),
						configs.d_model,configs.n_heads),
					configs.d_model,
					configs.c_out,
					configs.d_ff,
					moving_avg=configs.moving_avg,
					dropout=configs.dropout,
					activation=configs.activation,) for l in range(configs.d_layers)],
			norm_layer = my_Layernorm(configs.d_model),
			projection=nn.Linear(configs.d_model,configs.c_out,bias=True))

	def forward(self,x_enc,x_dec,
			enc_self_mask=None,dec_self_mask=None,dec_enc_mask=None):
		#decomp init
		mean = torch.mean(x_enc,dim=1).unsqueeze(1).repeat(1,self.pred_len,1)
		# print('mean:',mean.shape)
		zeros = torch.zeros([x_dec.shape[0],self.pred_len,x_dec.shape[2]],device=x_enc.device)
		# print('zeros:',zeros.shape)
		seasonal_init,trend_init = self.decomp(x_enc)

		trend_init = torch.cat([trend_init[:,-self.label_len:,:],mean],dim=1)#趋势部分
		seasonal_init = torch.cat([seasonal_init[:,-self.label_len:,:],zeros],dim=1)#震荡部分初始化
		# print('seasonal_init:',seasonal_init.shape)
		# print('trend_init:',trend_init.shape)
		enc_out = self.enc_embedding(x_enc)
		# print('enc_out:',enc_out.shape)
		enc_out,attns = self.encoder(enc_out,attn_mask = enc_self_mask)
		# print('enc_out:',enc_out.shape)	
		#dec

		dec_out = self.dec_embedding(seasonal_init)#embedding

		seasonal_part,trend_part = self.decoder(dec_out,enc_out,x_mask=dec_self_mask,cross_mask=dec_enc_mask,trend=trend_init)

		dec_out = trend_part+seasonal_part

		if self.output_attention:
			return dec_out[:,-self.pred_len:,:],attns
		else:
			return dec_out[:,-self.pred_len:,:]


