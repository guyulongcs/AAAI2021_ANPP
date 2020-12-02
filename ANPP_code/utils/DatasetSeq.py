from tensorflow.contrib.keras import preprocessing
from collections import defaultdict
import itertools
import os
import tensorflow as tf
import numpy as np
import pandas as pd

import sys
import shutil
sys.path.append("..")
from Config import *
from utils.Dataset import *
from utils.Metrics import *
from Time import *

pad_sequences = preprocessing.sequence.pad_sequences

class DatasetSeq():
	def __init__(self, settings, pad = True, paddingStyle = 'post'):
		self.eventTrain = settings["eventTrain"]
		self.eventTest = settings["eventTest"]
		self.timeTrain = settings["timeTrain"]
		self.timeTest = settings["timeTest"]
		self.args = settings["args"]

		self.pad = pad
		self.paddingStyle = paddingStyle

		#args
		self.scale = self.args.scale
		self.time_scale = self.args.scale
		self.time_method = self.args.time_method

		self.time_bucket_dim = self.args.time_bucket_dim
		self.time_flag_parse_log = self.args.time_parse_log
		self.time_flag_parse_seq_to_startZero = self.args.time_flag_parse_seq_to_startZero
		self.time_flagMinMax = self.args.time_flagMinMax
		self.time_flagScale = self.args.time_flagScale

		self.data = {
			"train": {},
			"eval": {},
			"test": {}
		}

		self.square_std = 0

		self.num_categories = 1

		self.time_interval_bucket = None

	@classmethod
	def process_time_seq_to_log(cls, timeData):
		print("process_time_seq_to_log...")
		# print("timeData:", len(timeData), len(timeData[0]))
		res = []
		for line in timeData:
			arr_new = [np.log(1+float(item)) for item in line]
			res.append(arr_new)

		# print("res:", len(res), len(res[0]))
		return res

	@classmethod
	def process_time_seq_to_startZero(cls, timeData):
		print("process_time_seq_to_startZero...")
		#print("timeData:", len(timeData), len(timeData[0]))
		res = []
		for line in timeData:
			arr_new = [(item-line[0]) for item in line]
			res.append(arr_new)

		#print("res:", len(res), len(res[0]))
		return res

	@classmethod
	def parse_time(cls, y, minTime, maxTime, flagMinMax):
		if (flagMinMax):
			res = (y - minTime) / (maxTime - minTime)
		# res = min(res, 1)
		else:
			res = y
		return res

	@classmethod
	def get_time_delta_prev(cls, seq_time):
		t_diff = np.diff(seq_time, axis=-1)
		t_first = seq_time[:, 0:1]
		t_res = np.concatenate((t_first, t_diff), axis=-1)
		return t_res

	@classmethod
	def get_mask_valid(cls, arr):
		return (arr > 0).astype(np.int8)

	def parse_sequences_time_padding(self, eventTrain, timeTrain, flag_test_mini=False):
		eventTrainIn = [x[:-1] for x in eventTrain]
		eventTrainOut = [x[1:] for x in eventTrain]

		#parse time train/test
		timeTrainIn = [[DatasetSeq.parse_time(y, self.minTime, self.maxTime, self.time_flagMinMax) for y in x[:-1]] for x in timeTrain]
		timeTrainOut = [[DatasetSeq.parse_time(y, self.minTime, self.maxTime, self.time_flagMinMax) for y in x[1:]] for x in timeTrain]

		if self.pad:
			train_event_in_seq = pad_sequences(eventTrainIn, value=0, padding=self.paddingStyle)
			train_event_out_seq = pad_sequences(eventTrainOut, value=0, padding=self.paddingStyle)
			train_time_in_seq = pad_sequences(timeTrainIn, dtype=float, padding=self.paddingStyle)
			train_time_out_seq = pad_sequences(timeTrainOut, dtype=float, padding=self.paddingStyle)
		else:
			train_event_in_seq = eventTrainIn
			train_event_out_seq = eventTrainOut
			train_time_in_seq = timeTrainIn
			train_time_out_seq = timeTrainOut



		if (flag_test_mini):
			n_test = 1000
			train_event_in_seq = train_event_in_seq[:n_test]
			train_event_out_seq = train_event_out_seq[:n_test]
			train_time_in_seq = train_time_in_seq[:n_test]
			train_time_out_seq = train_time_out_seq[:n_test]


		return (train_event_in_seq, train_event_out_seq, train_time_in_seq, train_time_out_seq)

	def build_dataset(self):
		# process time train/test to startZero if need

		#print("timeTest:", len(self.timeTest), len(self.timeTest[0]))

		print("self.timeTest old:", self.timeTest[0])
		if(self.time_flag_parse_seq_to_startZero and self.args.dataset in Config.dataset_should_process_time_seq_to_startZero):
			self.timeTrain = DatasetSeq.process_time_seq_to_startZero(self.timeTrain)
			self.timeTest = DatasetSeq.process_time_seq_to_startZero(self.timeTest)
		print("self.timeTest new:", self.timeTest[0])

		print("self.timeTest old:", self.timeTest[0])
		if (self.time_flag_parse_log == True):
			self.timeTrain = DatasetSeq.process_time_seq_to_log(self.timeTrain)
			self.timeTest = DatasetSeq.process_time_seq_to_log(self.timeTest)
		print("self.timeTest new:", self.timeTest[0])

		#min max time
		self.maxTime = max(itertools.chain((max(x) for x in self.timeTrain), (max(x) for x in self.timeTest)))
		self.minTime = min(itertools.chain((min(x) for x in self.timeTrain), (min(x) for x in self.timeTest)))
		print "minTime:%f, maxTime:%f" % (self.minTime, self.maxTime)


		#get num_categories
		unique_samples = set()
		for x in self.eventTrain + self.eventTest:
			unique_samples = unique_samples.union(x)
		self.num_categories = len(unique_samples)


		#train event/time in/out
		(train_event_in_seq, train_event_out_seq, train_time_in_seq, train_time_out_seq) = self.parse_sequences_time_padding(self.eventTrain, self.timeTrain, flag_test_mini=False)
		#test event/time in/out
		(test_event_in_seq, test_event_out_seq, test_time_in_seq, test_time_out_seq) = self.parse_sequences_time_padding(self.eventTest, self.timeTest, flag_test_mini=Config.flag_test_mini)

		print("test_time_in_seq:", test_time_in_seq[0])
		print("test_time_out_seq:", test_time_out_seq[0])
		#print("timeTest:", len(self.timeTest), len(self.timeTest[0]))
		#print("test_time_in_seq", test_time_in_seq.shape)
		train_mask_valid_in_seq = DatasetSeq.get_mask_valid(train_event_in_seq)
		test_mask_valid_in_seq = DatasetSeq.get_mask_valid(test_event_in_seq)

		print("test_time_out_seq[0]-test_time_in_seq[0]:", test_time_out_seq[0]-test_time_in_seq[0])

		print('delta-t (testing) [Initial State]= ')
		test_valid = test_event_in_seq > 0
		#print("test_time_out_seq:", test_time_out_seq.shape)
		#print("test_time_in_seq:", test_time_in_seq.shape)
		#print("test_valid:", test_valid.shape)
		print(pd.Series((test_time_out_seq - test_time_in_seq)[test_valid]).describe())



		self.time_scale = 1
		if(self.time_flagMinMax):
			self.time_scale *= (self.maxTime - self.minTime)
		if (self.time_flagScale):
			train_time_in_seq /= self.scale
			train_time_out_seq /= self.scale
			test_time_in_seq /= self.scale
			test_time_out_seq /= self.scale
			self.time_scale *= self.scale

		# get time diff
		train_time_in_seq_delta_prev = DatasetSeq.get_time_delta_prev(train_time_in_seq)
		test_time_in_seq_delta_prev = DatasetSeq.get_time_delta_prev(test_time_in_seq)

		print("test_time_in_seq_delta_prev[0]:", test_time_in_seq_delta_prev[0])

		#get time bucket
		train_valid_in_seq = train_event_in_seq > 0
		time_interval_list = (train_time_out_seq - train_time_in_seq)[train_valid_in_seq].flatten()
		self.time_interval_bucket = Time.get_list_num_bucket(time_interval_list, self.time_bucket_dim)
		print("time_interval_bucket:", self.time_interval_bucket)
		train_time_in_seq_delta_prev_bucket = Time.convert_time_to_bucket(train_time_in_seq_delta_prev, self.time_interval_bucket, self.time_bucket_dim)
		test_time_in_seq_delta_prev_bucket = Time.convert_time_to_bucket(test_time_in_seq_delta_prev, self.time_interval_bucket, self.time_bucket_dim)

		print("test_time_in_seq_delta_prev_bucket[0]:", test_time_in_seq_delta_prev_bucket[0])
		train_cnt = int(len(train_event_in_seq) * float(7 / 8.))
		#print("train_event_in_seq", train_event_in_seq.shape)

		# get std
		train_valid = train_event_in_seq > 0
		std = pd.Series((train_time_out_seq - train_time_in_seq)[train_valid]).describe()["std"]
		square_std = std ** 2
		self.square_std = square_std

		#process final result
		#train
		self.data["train"] = {
			"event_in_seq": np.array(train_event_in_seq[:train_cnt, :]),
			"event_out_seq": np.array(train_event_out_seq[:train_cnt, :]),
			"time_in_seq": np.array(train_time_in_seq[:train_cnt, :]),
			"time_out_seq": np.array(train_time_out_seq[:train_cnt, :]),
			"time_in_seq_delta_prev": train_time_in_seq_delta_prev[:train_cnt, :],
			"time_in_seq_delta_prev_bucket": train_time_in_seq_delta_prev_bucket[:train_cnt, :],
		}

		#eval
		self.data["eval"] = {
			"event_in_seq": np.array(train_event_in_seq[train_cnt:, :]),
			"event_out_seq": np.array(train_event_out_seq[train_cnt:, :]),
			"time_in_seq": np.array(train_time_in_seq[train_cnt:, :]),
			"time_out_seq": np.array(train_time_out_seq[train_cnt:, :]),
			"time_in_seq_delta_prev": train_time_in_seq_delta_prev[train_cnt:, :],
			"time_in_seq_delta_prev_bucket": train_time_in_seq_delta_prev_bucket[train_cnt:, :],
		}

		#test
		self.data["test"] = {
			"event_in_seq": np.array(test_event_in_seq),
			"event_out_seq": np.array(test_event_out_seq),
			"time_in_seq": np.array(test_time_in_seq),
			"time_out_seq": np.array(test_time_out_seq),
			"time_in_seq_delta_prev": test_time_in_seq_delta_prev,
			"time_in_seq_delta_prev_bucket": test_time_in_seq_delta_prev_bucket,
		}

		print "train:%d, eval:%d, test:%d" % (len(self.data["train"]["event_in_seq"]), len(self.data["eval"]["event_in_seq"]), len(self.data["test"]["event_in_seq"]))
		print "minTime:%f, maxTime:%f" % (self.minTime, self.maxTime)
		print "num_categories:%d" % self.num_categories

	def calc_base_rate(self, training=True):
		"""Calculates the base-rate for intelligent parameter initialization from the training data."""
		suffix = 'train' if training else 'test'

		in_key = 'time_in_seq'
		out_key = 'time_out_seq'
		valid_key = 'event_in_seq'

		dts = (self.data[suffix][out_key] - self.data[suffix][in_key])[self.data[suffix][valid_key] > 0]
		return 1.0 / np.mean(dts)

	def calc_base_event_prob(self, training=True):
		"""Calculates the base probability of event types for intelligent parameter initialization from the training data."""
		dict_key = 'train' if training else 'test'

		class_count = defaultdict(lambda: 0.0)
		for evts in self.data[dict_key]["event_in_seq"]:
			for ev in evts:
				class_count[ev] += 1.0

		total_events = 0.0
		probs = []
		for cat in range(1, self.num_categories + 1):
			total_events += class_count[cat]

		for cat in range(1, self.num_categories + 1):
			probs.append(class_count[cat] / total_events)

		return np.array(probs)

	def data_stats(self):
		"""Prints basic statistics about the dataset."""
		train_valid = self.data["train"]["event_in_seq"] > 0
		test_valid = self.data["test"]["event_in_seq"] > 0

		print('Num categories = ', self.num_categories)
		print('delta-t (training) = ')
		print(pd.Series((self.data["train"]['time_out_seq'] - self.data["train"]['time_in_seq'] )[train_valid]).describe())
		train_base_rate = self.calc_base_rate(training=True)
		print('base-rate = {}, log(base_rate) = {}'.format(train_base_rate, np.log(train_base_rate)))
		print('Class probs = ', self.calc_base_event_prob( training=True))

		print('delta-t (testing) = ')
		print(pd.Series((self.data["test"]['time_out_seq'] - self.data["test"]['time_in_seq'])[test_valid]).describe())
		test_base_rate = self.calc_base_rate(training=False)
		print('base-rate = {}, log(base_rate) = {}'.format(test_base_rate, np.log(test_base_rate)))
		print('Class probs = ', self.calc_base_event_prob(training=False))

		print('Training sequence lenghts = ')
		print(pd.Series(train_valid.sum(axis=1)).describe())

		print('Testing sequence lenghts = ')
		print(pd.Series(test_valid.sum(axis=1)).describe())



