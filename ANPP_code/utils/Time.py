import datetime
import time
import math
import numpy as np


class Time():
	@classmethod
	def timestamp_to_datetime(cls, unix_ts):
		t = datetime.datetime.fromtimestamp(unix_ts)
		dt = t.strftime("%Y-%m-%d")
		return dt

	@classmethod
	def datetime_to_timestamp(cls, dt):
		d = datetime.datetime.strptime(dt, "%Y-%m-%d")
		t = d.timetuple()
		ts = int(time.mktime(t))
		return ts

	@classmethod
	def get_time_diff_days(cls, ts_pre, ts_cur):
		ts_diff_seconds = int(ts_cur) - int(ts_pre)
		ts_diff_days = int(round(ts_diff_seconds / float(24 * 3600)))
		return ts_diff_days

	@classmethod
	def get_time_diff_days_str_str(cls, s_pre, s_cur):
		ts_pre = Time.datetime_to_timestamp(s_pre)
		ts_cur = Time.datetime_to_timestamp(s_cur)
		ts_diff_days =  Time.get_time_diff_days(ts_pre, ts_cur)
		return ts_diff_days

	@classmethod
	def buket_timeDays(cls, t):
		res = int(t / 7.0)
		res = min(res, 51)
		return res

	@classmethod
	def get_time_diff_bucket(cls, ts_pre, ts_cur):
		ts_diff_days = Time.get_time_diff_days(ts_pre, ts_cur)
		bucket = Time.buket_timeDays(ts_diff_days)
		return bucket

	@classmethod
	def get_list_num_bucket(cls, arrNum, bucket_dim):
		quantiles = np.zeros((bucket_dim), dtype=np.float32)
		arrNum = sorted(arrNum)
		arrLen = len(arrNum)
		for i in range(bucket_dim):
			index = np.int32(arrLen * i / bucket_dim)
			quantiles[i] = np.float32(arrNum[index])
		quantiles[0] = 0.
		return quantiles

	@classmethod
	def convert_time_to_bucket(cls, arrNum, bucketArr, bucket_dim):
		arrBucket = np.digitize(arrNum, bucketArr)
		arrBucket = np.clip(arrBucket, 0, bucket_dim-1).astype(np.int32)
		return arrBucket


