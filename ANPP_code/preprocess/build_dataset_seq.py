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


def run(folder):
	dataset_tag = "dataset_seq"
	print("build_dataset_seq...")

	event_train_file = os.path.join(folder, "event-train.txt")
	event_test_file = os.path.join(folder, "event-test.txt")
	time_train_file = os.path.join(folder, "time-train.txt")
	time_test_file = os.path.join(folder, "time-test.txt")

	with open(event_train_file, 'r') as in_file:
		eventTrain = [[int(y) for y in x.strip().split()] for x in in_file]

	with open(event_test_file, 'r') as in_file:
		eventTest = [[int(y) for y in x.strip().split()] for x in in_file]

	with open(time_train_file, 'r') as in_file:
		timeTrain = [[float(y) for y in x.strip().split()] for x in in_file]

	with open(time_test_file, 'r') as in_file:
		timeTest = [[float(y) for y in x.strip().split()] for x in in_file]

	assert len(timeTrain) == len(eventTrain)
	assert len(eventTest) == len(timeTest)

	# write pkl dataset
	# output file
	file_pkl_dataset = os.path.join(folder, Config.dict_pkl_dataset[dataset_tag])
	print("write %s" % file_pkl_dataset)
	with open(file_pkl_dataset, 'wb') as f:
		pickle.dump((eventTrain, eventTest, timeTrain, timeTest), f, pickle.HIGHEST_PROTOCOL)
	print("write done!")

if __name__ == "__main__":
	print "\nbuild_dataset_seq start..."
	# parse parameters
	arr = sys.argv
	assert (len(arr) == 1 + 1)
	folder = arr[1]

	run(folder)


