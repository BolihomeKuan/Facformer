"""
Evaluation Metrics for Facformer Model

This module provides comprehensive evaluation metrics for time series prediction
tasks, specifically designed for aquatic products shelf-life prediction.
"""

import numpy as np


def RSE(pred, true):
	"""Root Relative Squared Error"""
	return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
	"""Correlation coefficient between predictions and ground truth"""
	u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
	d = np.sqrt(((true - true.mean(0)) ** 2).sum(0) * ((pred - pred.mean(0)) ** 2).sum(0))
	return (u / d).mean(-1)


def MAE(pred, true):
	"""Mean Absolute Error"""
	return np.mean(np.abs(pred - true))


def MSE(pred, true):
	"""Mean Squared Error"""
	return np.mean((pred - true) ** 2)


def RMSE(pred, true):
	"""Root Mean Squared Error"""
	return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
	"""Mean Absolute Percentage Error"""
	return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
	"""Mean Squared Percentage Error"""
	return np.mean(np.square((pred - true) / true))


def metric(pred, true):
	"""
	Calculate comprehensive evaluation metrics
	
	Args:
		pred: Model predictions
		true: Ground truth values
		
	Returns:
		Tuple of (MAE, MSE, RMSE, MAPE, MSPE)
	"""
	mae = MAE(pred, true)
	mse = MSE(pred, true)
	rmse = RMSE(pred, true)
	mape = MAPE(pred, true)
	mspe = MSPE(pred, true)

	return mae, mse, rmse, mape, mspe
