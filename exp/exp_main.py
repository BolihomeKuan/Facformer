import torch
import torch.nn as nn
from torch import optim
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from data_provider.data_loader import get_data
from exp.exp_basic import Exp_Basic
from models import Facformer,Transformer,Autoformer,LSTM,RNN
from utils.tools import EarlyStopping,adjust_learning_rate,visual
from utils.metrics import metric

class Exp_Main(Exp_Basic):
	def __init__(self,args):
		super(Exp_Main,self).__init__(args)

	def _build_model(self):
		model_dict = {
		'Facformer':Facformer,
		'Autoformer':Autoformer,
		'Transformer':Transformer,
		'LSTM':LSTM,
		'RNN':RNN,
		}
		model = model_dict[self.args.model].Model(self.args).float()

		if self.args.use_multi_gpu and self.args.use_gpu:
			model = nn.DataParallel(model,device_ids=self.args.device_ids)
		return model

	def _get_data(self,flag):
		train_data,valid_data,test_data,train_loader,valid_loader,test_loader = get_data(self.args)
		return train_data,valid_data,test_data,train_loader,valid_loader,test_loader

	def _select_optimizer(self):
		model_optim = optim.Adam(self.model.parameters(),lr = self.args.learning_rate)
		return model_optim

	def _select_criterion(self):
		criterion = nn.MSELoss()
		return criterion

	def vali(self,vali_data,vali_loader,criterion):
		total_loss = []
		self.model.eval()
		with torch.no_grad():
			for i,(batch_x,batch_y) in enumerate(vali_loader):
				batch_x = batch_x.float().to(self.device)
				batch_y = batch_y.float()
				#decoder input
				dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).float()
				# print(dec_inp.shape)
				dec_inp = torch.cat([batch_y[:,:self.args.label_len,:],dec_inp],dim=1).float().to(self.device)
				# print(dec_inp.shape)
				if self.args.use_amp:
					with torch.cuda.amp.autocast():
						if self.args.output_attention:
							if self.args.model in ['Transformer','Autoformer']:
								outputs = self.model(batch_x, dec_inp)[0]
							else:
								outputs = self.model(batch_x)
						else:
							if self.args.model in ['Transformer','Autoformer']:
								outputs = self.model(batch_x, dec_inp)
							else:
								outputs = self.model(batch_x)
				else:
					if self.args.output_attention:
						if self.args.model in ['Transformer','Autoformer']:
							outputs = self.model(batch_x, dec_inp)[0]
						else:
							outputs = self.model(batch_x)
					else:
						if self.args.model in ['Transformer','Autoformer']:
							outputs = self.model(batch_x, dec_inp)
						else:
							outputs = self.model(batch_x)
				f_dim = -2
				outputs = outputs[:,-self.args.pred_len:,f_dim:]
				batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

				pred = outputs.detach().cpu()
				true = batch_y.detach().cpu()
				loss = criterion(pred,true)
				total_loss.append(loss)

		total_loss = np.average(total_loss)
		self.model.train()
		return total_loss
	
	def train(self,setting):
		train_data,vali_data,test_data,train_loader,vali_loader,test_loader = self._get_data(self.args)


		path = os.path.join(self.args.checkpoints,setting)
		if not os.path.exists(path):
			os.makedirs(path)

		time_now = time.time()
		train_steps = len(train_loader)
		early_stopping = EarlyStopping(patience=self.args.patience,verbose=True)

		model_optim = self._select_optimizer()
		criterion = self._select_criterion()

		if self.args.use_amp:
			scaler = torch.cuda.amp.GradScaler()

		for epoch in range(self.args.train_epochs):
			iter_count = 0
			train_loss = []
			self.model.train()
			epoch_time = time.time()

			for i,(batch_x,batch_y,) in enumerate(train_loader):
				iter_count += 1
				model_optim.zero_grad()
				batch_x = batch_x.float().to(self.device)  # Input sequence
				batch_y = batch_y.float().to(self.device)  # Target sequence

				dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).float()
				dec_inp = torch.cat([batch_y[:,:self.args.label_len,:],dec_inp],dim=1).float().to(self.device)  # Decoder input: known labels + zero padding
				# print('batch_x.shape::',batch_x.shape)
				# print('dec_inp.shape::',dec_inp.shape)
				if self.args.use_amp:
					with torch.cuda.amp.autocast():
						if self.args.output_attention:
							if self.args.model in ['Transformer','Autoformer','Shelfformer']:
								outputs = self.model(batch_x, dec_inp)[0]
							else:
								outputs = self.model(batch_x)
						else:
							if self.args.model in ['Transformer','Autoformer','Shelfformer']:
								outputs = self.model(batch_x, dec_inp)
							else:
								outputs = self.model(batch_x)

						f_dim = -2  # Predict last two variables (quality indicators)
						outputs = outputs[:,-self.args.pred_len:,f_dim:]  # Extract prediction length outputs
						batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
						loss = criterion(outputs,batch_y)
						train_loss.append(loss.item())
				else:
					if self.args.output_attention:
						if self.args.model in ['Transformer','Autoformer']:
							outputs = self.model(batch_x, dec_inp)[0]
						else:
							outputs = self.model(batch_x)
					else:
						if self.args.model in ['Transformer','Autoformer']:
							outputs = self.model(batch_x, dec_inp)
						else:
							outputs = self.model(batch_x)

					f_dim = -2  # Predict last two variables (quality indicators)
					outputs = outputs[:, -self.args.pred_len:, f_dim:]
					batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
					loss = criterion(outputs, batch_y)
					train_loss.append(loss.item())

				if (i+1) % 100 ==0:
					print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
					speed = (time.time() - time_now) / iter_count
					left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
					print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
					iter_count = 0
					time_now = time.time()
				if self.args.use_amp:
					scaler.scale(loss).backward()
					scaler.step(model_optim)
					scaler.update()
				else:
					loss.backward()
					model_optim.step()
			print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
			train_loss = np.average(train_loss)
			vali_loss = self.vali(vali_data, vali_loader, criterion)
			test_loss = self.vali(test_data, test_loader, criterion)					

			print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
			    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
			early_stopping(vali_loss, self.model, path)
			if early_stopping.early_stop:
				print("Early stopping")
				break

			adjust_learning_rate(model_optim, epoch + 1, self.args)

		best_model_path = path + '/' + 'checkpoint.pth'
		self.model.load_state_dict(torch.load(best_model_path))

		return

	def test(self,setting,load=0):
		_,_,test_data,_,_,test_loader = self._get_data(self.args)
		if load:
			print('loading model')
			self.model.load_state_dict(torch.load(os.path.join('./checkpoints/'+setting,'checkpoint.pth')))
		preds = []
		trues = []
		folder_path = './test_results/'+setting+'/'
		if not os.path.exists(folder_path):
			os.makedirs(folder_path)

		self.model.eval()
		with torch.no_grad():
			for i,(batch_x,batch_y) in enumerate(test_loader):
				batch_x = batch_x.float().to(self.device)
				batch_y = batch_y.float().to(self.device)

				#decoder input

				dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).float()
				# print('dec_inp.shape',dec_inp.shape)
				# print('batch_y.shape',batch_y.shape)
				# print(batch_y[:,:self.args.label_len,:].shape)
				dec_inp = torch.cat([batch_y[:,:self.args.label_len,:],dec_inp],dim=1).float().to(self.device)#[6, 4, 4]+[6, 4, 4]=[6, 8, 4]

				#encoder - decoder
				if self.args.use_amp:
					with torch.cuda.amp.autocast():
						if self.args.output_attention:
							if self.args.model in ['Transformer','Autoformer']:
								outputs = self.model(batch_x, dec_inp)[0]
							else:
								outputs = self.model(batch_x)
						else:
							if self.args.model in ['Transformer','Autoformer']:
								outputs = self.model(batch_x, dec_inp)
							else:
								outputs = self.model(batch_x)
				else:
					if self.args.output_attention:
						if self.args.model in ['Transformer','Autoformer']:
							outputs = self.model(batch_x, dec_inp)[0]
						else:
							outputs = self.model(batch_x)
					else:
						if self.args.model in ['Transformer','Autoformer']:
							outputs = self.model(batch_x, dec_inp)
						else:
							outputs = self.model(batch_x)
				f_dim = 0  # Feature dimension for prediction
				outputs = outputs[:,-self.args.pred_len:,f_dim:]
				batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
				outputs = outputs.detach().cpu().numpy()
				batch_y = batch_y.detach().cpu().numpy()

				pred = outputs
				true = batch_y
				preds.append(pred)
				trues.append(true)
				if i%20==0:
					input_ = batch_x.detach().cpu().numpy()
					gt = np.concatenate((input_[0,:,-1],true[0,:,-1]),axis=0)
					pd = np.concatenate((input_[0,:,-1],pred[0,:,-1]),axis=0)
					visual(gt,pd,os.path.join(folder_path,str(i)+'.pdf'))
		preds = np.concatenate(preds,axis=0)
		trues = np.concatenate(trues,axis=0)
		print('test shape:', preds.shape, trues.shape)
		preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
		trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

		folder_path = './results/'+setting+'/'
		if not os.path.exists(folder_path):
			os.makedirs(folder_path)
		mae,mse,rmse,mape,mspe = metric(preds,trues)
		print('mse:{}, mae:{}'.format(mse, mae))
		f = open('result.txt','a')
		f.write(setting+'\n')
		f.write('mse:{}, mae:{}'.format(mse, mae))
		f.write('\n')
		f.write('\n')
		f.close()

		np.save(folder_path+'metrics.npy',np.array([mae, mse, rmse, mape, mspe]))
		np.save(folder_path+'pred.npy',preds)
		np.save(folder_path+'true.npy',trues)
		# visual(trues,preds,os.path.join(folder_path+'res.pdf'))
		return

	def train_mark(self,setting):
		train_data,vali_data,test_data,train_loader,vali_loader,test_loader = self._get_data(self.args)
		path = os.path.join(self.args.checkpoints,setting)
		if not os.path.exists(path):
			os.makedirs(path)

		time_now = time.time()
		train_steps = len(train_loader)
		early_stopping = EarlyStopping(patience=self.args.patience,verbose=True)

		model_optim = self._select_optimizer()
		criterion = self._select_criterion()

		if self.args.use_amp:
			scaler = torch.cuda.amp.GradScaler()

		for epoch in range(self.args.train_epochs):
			iter_count = 0
			train_loss = []
			self.model.train()
			epoch_time = time.time()

			for i,(batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
				iter_count += 1
				model_optim.zero_grad()
				batch_x = batch_x.float().to(self.device)  # Input sequence
				batch_y = batch_y.float().to(self.device)  # Target sequence
				batch_x_mark = batch_x_mark.float().to(self.device)  # Input marks (temperature, time)
				batch_y_mark = batch_y_mark.float().to(self.device)  # Target marks

				dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).float()  # Initialize decoder input
				dec_inp = torch.cat([batch_y[:,:self.args.label_len,:],dec_inp],dim=1).float().to(self.device)  # Combine known + unknown
				# print('batch_x.shape',batch_x.shape)
				# print('batch_x_mark.shape',batch_x_mark.shape)
				# print('dec_inp.shape',dec_inp.shape)
				# print('batch_y_mark.shape',batch_y_mark.shape)
				# print(batch_x)
				if self.args.use_amp:
					with torch.cuda.amp.autocast():
						outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

						outputs = outputs[:,-self.args.pred_len:,-2:]
						batch_y = batch_y[:,-self.args.pred_len:,-2:].to(self.device)
						loss = criterion(outputs,batch_y)
						train_loss.append(loss.item())
				else:
					# print('batch_x.shape:',batch_x.shape)
					# print('batch_x_mark.shape:',batch_x_mark.shape)
					# print('dec_inp.shape:',dec_inp.shape)
					# print('batch_y_mark.shape:',batch_y_mark.shape)
					outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
					# print('outputs.shape:::::',outputs)
					# print('self.args.pred_len',self.args.pred_len)
					outputs = outputs[:, -self.args.pred_len:, -2:]
					batch_y = batch_y[:, -self.args.pred_len:, -2:].to(self.device)
					loss = criterion(outputs, batch_y)
					train_loss.append(loss.item())

				if (i+1) % 100 ==0:
					print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
					speed = (time.time() - time_now) / iter_count
					left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
					print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
					iter_count = 0
					time_now = time.time()
				if self.args.use_amp:
					scaler.scale(loss).backward()
					scaler.step(model_optim)
					scaler.update()
				else:
					loss.backward()
					model_optim.step()
			print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
			train_loss = np.average(train_loss)
			vali_loss = self.vali_mark(vali_data, vali_loader, criterion)
			test_loss = self.vali_mark(test_data, test_loader, criterion)					

			print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
			    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
			early_stopping(vali_loss, self.model, path)
			if early_stopping.early_stop:
				print("Early stopping")
				break
			adjust_learning_rate(model_optim, epoch + 1, self.args)

		best_model_path = path + '/' + 'checkpoint.pth'
		self.model.load_state_dict(torch.load(best_model_path))
		return

	def vali_mark(self,vali_data,vali_loader,criterion):
			total_loss = []
			self.model.eval()
			with torch.no_grad():
				for i,(batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
					batch_x = batch_x.float().to(self.device)
					batch_y = batch_y.float()

					batch_x_mark = batch_x_mark.float().to(self.device)
					batch_y_mark = batch_y_mark.float().to(self.device)

					# Decoder input preparation
					dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).float()
					dec_inp = torch.cat([batch_y[:,:self.args.label_len,:],dec_inp],dim=1).float().to(self.device)
					# print('vali_mark:batch_x.shape',batch_x.shape)
					# print('vali_mark:batch_x_mark.shape',batch_x_mark.shape)
					# print('vali_mark:dec_inp.shape',dec_inp.shape)
					# print('vali_mark:batch_y_mark.shape',batch_y_mark.shape)
					if self.args.use_amp:
						with torch.cuda.amp.autocast():
							outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
					else:
						outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
					outputs = outputs[:,-self.args.pred_len:,-2:]
					batch_y = batch_y[:,-self.args.pred_len:,-2:].to(self.device)

					pred = outputs.detach().cpu()
					true = batch_y.detach().cpu()
					loss = criterion(pred,true)
					total_loss.append(loss)

			total_loss = np.average(total_loss)
			self.model.train()
			return total_loss

	def test_mark(self,setting,load=0):
		_,_,test_data,_,_,test_loader = self._get_data(self.args)
		if load:
			print('loading model')
			self.model.load_state_dict(torch.load(os.path.join('./checkpoints/'+setting,'checkpoint.pth')))
		preds = []
		trues = []
		folder_path = './test_results/'+setting+'/'
		if not os.path.exists(folder_path):
			os.makedirs(folder_path)

		self.model.eval()
		with torch.no_grad():
			for i,(batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
				batch_x = batch_x.float().to(self.device)
				batch_y = batch_y.float().to(self.device)

				batch_x_mark = batch_x_mark.float().to(self.device)
				batch_y_mark = batch_y_mark.float().to(self.device)

				# Decoder input preparation
				dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).float()  # Initialize decoder input
				dec_inp = torch.cat([batch_y[:,:self.args.label_len,:],dec_inp],dim=1).float().to(self.device)  # Combine known + padding

				# Encoder-decoder forward pass
				if self.args.use_amp:
					with torch.cuda.amp.autocast():
						outputs = self.model(batch_x,batch_x_mark,dec_inp,batch_y_mark)
				else:
					outputs = self.model(batch_x,batch_x_mark,dec_inp,batch_y_mark)

				outputs = outputs[:,-self.args.pred_len:,-2:]
				batch_y = batch_y[:,-self.args.pred_len:,-2:].to(self.device)
				outputs = outputs.detach().cpu().numpy()
				batch_y = batch_y.detach().cpu().numpy()

				pred = outputs
				true = batch_y
				preds.append(pred)
				trues.append(true)
				if i%20==0:
					input_ = batch_x.detach().cpu().numpy()
					gt = np.concatenate((input_[0,:,-1],true[0,:,-1]),axis=0)
					pd = np.concatenate((input_[0,:,-1],pred[0,:,-1]),axis=0)
					visual(gt,pd,os.path.join(folder_path,str(i)+'.pdf'))
				preds = np.concatenate(preds,axis=0)
				trues = np.concatenate(trues,axis=0)
				print('test shape:', preds.shape, trues.shape)
				preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
				trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

				folder_path = './results/'+setting+'/'
				if not os.path.exists(folder_path):
					os.makedirs(folder_path)
				mae,mse,rmse,mape,mspe = metric(preds,trues)
				print('mse:{}, mae:{}'.format(mse, mae))
				f = open('result.txt','a')
				f.write(setting+'\n')
				f.write('mse:{}, mae:{}'.format(mse, mae))
				f.write('\n')
				f.write('\n')
				f.close()

				np.save(folder_path+'metrics.npy',np.array([mae, mse, rmse, mape, mspe]))
				np.save(folder_path+'pred.npy',preds)
				np.save(folder_path+'true.npy',trues)
				return 

	def predict_mark(self,setting,load=0):
		pred_data, pred_loader = self.get_pred_data()
		if load:
			print('loading model')
			self.model.load_state_dict(torch.load(os.path.join('./checkpoints/'+setting,'checkpoint.pth')))
		preds = []
		self.model.eval()
		with torch.no_grad():
			for i,(batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
				batch_x = batch_x.float().to(self.device)
				batch_y = batch_y.float()
				batch_x_mark = batch_x_mark.float().to(self.device)
				batch_y_mark = batch_y_mark.float().to(self.device)	

				dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
				dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

				if self.args.use_amp:
					with torch.cuda.amp.autocast():
						if self.args.output_attention:
							outputs = self.model(batch_x,batch_x_mark, dec_inp, batch_y_mark)[0]
						else:
							outputs = self.model(batch_x,batch_x_mark,dec_inp,batch_y_mark)
				else:
					if self.args.output_attention:
						outputs = self.model(batch_x,batch_x_mark, dec_inp, batch_y_mark)[0]
					else:
						outputs = self.model(batch_x,batch_x_mark,dec_inp,batch_y_mark)
				pred = outputs.detach().cpu().numpy()  # .squeeze()
				preds.append(pred)
		preds = np.array(preds)
		preds = preds.reshape(-1,preds.shape[-2], preds.shape[-1])
		# result save
		folder_path = './results/' + setting + '/'
		if not os.path.exists(folder_path):
			os.makedirs(folder_path)

		np.save(folder_path + 'real_prediction.npy', preds)

		return