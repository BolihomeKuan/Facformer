
import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from typing import List

class TimeFeature:
	def __init__(self):
		...

	def __call__(self,index:pd.DatetimeIndex) -> np.ndarray:
		...

	def __repr__(self):
		return self.__class__.__name__+"()"

class SecondOfMinute(TimeFeature):
	"""Second of minute encoded as value between [-0.5,0.5]"""
	def __call__(self, index: pd.DatetimeIndex)->np.ndarray:
		return index.second /59.0 -0.5  # Range: -0.483 to 0.5


class MinuteOfHour(TimeFeature):
	"""Minute of hour encoded as value between [-0.5,0.5]"""
	def __call__(self, index: pd.DatetimeIndex) ->np.ndarray:
		return index.minute / 59.0 -0.5

class HourOfDay(TimeFeature):
	def __call__(self,index: pd.DatetimeIndex) -> np.ndarray:
		return index.hour/ 23.0 - 0.5

class DayOfWeek(TimeFeature):
	def __call__(self,index: pd.DatetimeIndex) -> np.ndarray:
		return index.dayofweek / 6.0 -0.5

class DayOfMonth(TimeFeature):
	def __call__(self,index:pd.DatetimeIndex) -> np.ndarray:
		return (index.day -1 )/30.0 -0.5

class DayOfYear(TimeFeature):
	def __call__(self,index:pd.DatetimeIndex) -> np.ndarray:
		return (index.dayofyear-1)/365 -0.5

class MonthOfYear(TimeFeature):
	def __call__(self,index:pd.DatetimeIndex) -> np.ndarray:
		return (index.month -1) /11.0 -0.5  # Divided by 11 to map 12 months to [-0.5, 0.5] range

class WeekOfYear(TimeFeature):
	def __call__(self,index:pd.DatetimeIndex) -> np.ndarray:
		return (index.isocalendar().week -1 )/52.0 -0.5


def time_features_from_frequency_str(freq_str:str) ->List[TimeFeature]:
	"""
	return a list of time features that will be appropriate for 
	the given frequency string
	parameters ::: freq_str such as "12H",'5min',"1D" [multiple][granularity]
	"""
	features_by_offsets = {
		offsets.YearEnd :[],
		offsets.QuarterEnd:[MonthOfYear],
		offsets.MonthEnd:[MonthOfYear],
		offsets.Week:[DayOfMonth,WeekOfYear],
		offsets.Day:[DayOfWeek,DayOfMonth,DayOfYear],
		offsets.BusinessDay:[DayOfWeek,DayOfMonth,DayOfYear],
		offsets.Hour:[HourOfDay,DayOfWeek,DayOfMonth,DayOfYear],
		offsets.Minute:[MinuteOfHour,HourOfDay,DayOfWeek,DayOfMonth,DayOfYear],
		offsets.Second:[SecondOfMinute,MinuteOfHour,HourOfDay,DayOfWeek,DayOfMonth,DayOfYear]
	}
	offset = to_offset(freq_str)  # Convert input string to appropriate offset
	for offset_type,feature_classes in features_by_offsets.items():
		if isinstance(offset,offset_type):
			return [cls() for cls in feature_classes]
	supported_freq_msg = f"""
	unsipported frequency{freq_str}
	The following frequencies are supported:
	Y 	- yearly
		alias: A
	M 	- monthly
	W 	- weekly
	D 	- daily
	B 	- business days
	H 	- hourly
	T 	-minutely
		alias: min
	S 	- secondly
	"""
	raise RuntimeError(supported_freq_msg)


def time_features(dates,freq='h'):
	return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])


if __name__ == '__main__':
	data_file = 'ETTh1.csv'
	data = pd.read_csv(data_file)
	df_stamp = data[['date']][0:50]
	# print(df_stamp)
	print(type(df_stamp))
	print(type(df_stamp.date))



	# Example usage with time series data
	# This demonstrates how to extract time features from datetime index



