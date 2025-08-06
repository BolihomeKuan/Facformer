import os
import torch
import numpy as np
class Exp_Basic():
	def __init__(self,args):
		self.args = args
		self.device = self._acquire_device()
		self.model = self._build_model().to(self.device)

	def _build_model(self):
		# Must be implemented in subclasses
		raise NotImplementedError
		return None

	def _acquire_device(self):
		if self.args.use_gpu:
			os.environ['CUDA_VISIBLE_DEVICES'] = str(
				self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
			device = torch.device('cuda:{}'.format(self.args.gpu))
			print('Use GPU :cuda:{}'.format(self.args.gpu))
		else:
			device = torch.device('cpu')
			print('use CPU')
		return device

	def _get_data(self):
		...

	def vali(self):
		...