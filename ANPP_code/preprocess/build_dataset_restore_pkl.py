import random
import pickle
import numpy as np
import sys
import os

sys.path.append("..")

from Config import *
from utils.Time import *
from utils.Dataset import *
from utils.Tool import *


def write_list_list_to_file_event(data, event_file, time_file):
	print("write_list_list_to_file_event...")
	list_list_event = []
	list_list_time = []
	for seq in data:
		time_list = []
		event_list = []
		for item in seq:
			time_since_start = float(item['time_since_start'])
			type_event = int(item['type_event'])+1

			time_list.append(time_since_start)
			event_list.append(type_event)
		list_list_event.append(event_list)
		list_list_time.append(time_list)
	Tool.write_list_list_to_file(list_list_event, event_file)
	Tool.write_list_list_to_file(list_list_time, time_file)


def run(folder):
	print("build_dataset_restore_pkl...")

	#read
	pkl_train = os.path.join(folder, "train.pkl")
	pkl_dev = os.path.join(folder, "train.pkl")
	pkl_test = os.path.join(folder, "test.pkl")

	with open(pkl_train, 'rb') as f:
		d_train = pickle.load(f)
	with open(pkl_dev, 'rb') as f:
		d_dev = pickle.load(f)
	with open(pkl_test, 'rb') as f:
		d_test = pickle.load(f)

	print("load pkl done.")

	#parse
	data_train = []
	data_train.extend(d_train['train'])
	data_train.extend(d_dev['dev'])
	data_test=d_test['test']

	event_train_file = os.path.join(folder, "event-train.txt")
	event_test_file = os.path.join(folder, "event-test.txt")
	time_train_file = os.path.join(folder, "time-train.txt")
	time_test_file = os.path.join(folder, "time-test.txt")

	write_list_list_to_file_event(data_train, event_train_file, time_train_file)
	write_list_list_to_file_event(data_test, event_test_file, time_test_file)


	print("write done!")

if __name__ == "__main__":
	print "\nbuild_dataset_restore_pkl start..."
	# parse parameters
	arr = sys.argv
	assert (len(arr) == 1 + 1)
	folder = arr[1]

	run(folder)


