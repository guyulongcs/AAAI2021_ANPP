from tensorflow.contrib.keras import preprocessing
from collections import defaultdict
import itertools
import os
import tensorflow as tf
import numpy as np
import pandas as pd

import sys
sys.path.append("..")
from Config import *
from utils.Dataset import *
from utils.Metrics import *

pad_sequences = preprocessing.sequence.pad_sequences


def create_dir(dirname):
    """Creates a directory if it does not already exist."""
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def read_data(event_train_file, event_test_file, time_train_file, time_test_file,args,
              pad=True):
    """Read data from given files and return it as a dictionary."""

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

    # nb_samples = len(eventTrain)
    # max_seqlen = max(len(x) for x in eventTrain)
    unique_samples = set()

    for x in eventTrain + eventTest:
        unique_samples = unique_samples.union(x)

    maxTime = max(itertools.chain((max(x) for x in timeTrain), (max(x) for x in timeTest)))
    minTime = min(itertools.chain((min(x) for x in timeTrain), (min(x) for x in timeTest)))
    # minTime, maxTime = 0, 1

    eventTrainIn = [x[:-1] for x in eventTrain]
    eventTrainOut = [x[1:] for x in eventTrain]
    timeTrainIn = [[(y - minTime) / (maxTime - minTime) for y in x[:-1]] for x in timeTrain]
    timeTrainOut = [[(y - minTime) / (maxTime - minTime) for y in x[1:]] for x in timeTrain]

    if pad:
        train_event_in_seq = pad_sequences(eventTrainIn, padding='post')
        train_event_out_seq = pad_sequences(eventTrainOut, padding='post')
        train_time_in_seq = pad_sequences(timeTrainIn, dtype=float, padding='post')
        train_time_out_seq = pad_sequences(timeTrainOut, dtype=float, padding='post')
    else:
        train_event_in_seq = eventTrainIn
        train_event_out_seq = eventTrainOut
        train_time_in_seq = timeTrainIn
        train_time_out_seq = timeTrainOut


    eventTestIn = [x[:-1] for x in eventTest]
    eventTestOut = [x[1:] for x in eventTest]
    timeTestIn = [[(y - minTime) / (maxTime - minTime) for y in x[:-1]] for x in timeTest]
    timeTestOut = [[(y - minTime) / (maxTime - minTime) for y in x[1:]] for x in timeTest]

    if pad:
        test_event_in_seq = pad_sequences(eventTestIn, padding='post')
        test_event_out_seq = pad_sequences(eventTestOut, padding='post')
        test_time_in_seq = pad_sequences(timeTestIn, dtype=float, padding='post')
        test_time_out_seq = pad_sequences(timeTestOut, dtype=float, padding='post')
    else:
        test_event_in_seq = eventTestIn
        test_event_out_seq = eventTestOut
        test_time_in_seq = timeTestIn
        test_time_out_seq = timeTestOut

    time_scale = maxTime - minTime

    train_time_in_seq /= args.scale
    train_time_out_seq /= args.scale
    test_time_in_seq /= args.scale
    test_time_out_seq /= args.scale

    time_scale *= args.scale

    train_cnt = int(len(train_event_in_seq) * float(7/8.))
    print("train_event_in_seq", train_event_in_seq.shape)


    dict_res_item = {
        'train_event_in_seq': np.array(train_event_in_seq[:train_cnt,:]),
        'train_event_out_seq': np.array(train_event_out_seq[:train_cnt,:]),
        'train_time_in_seq': np.array(train_time_in_seq[:train_cnt,:]),
        'train_time_out_seq': np.array(train_time_out_seq[:train_cnt,:]),

        'eval_event_in_seq': np.array(train_event_in_seq[train_cnt:,:]),
        'eval_event_out_seq': np.array(train_event_out_seq[train_cnt:,:]),
        'eval_time_in_seq': np.array(train_time_in_seq[train_cnt:,:]),
        'eval_time_out_seq': np.array(train_time_out_seq[train_cnt:,:]),

        'test_event_in_seq': np.array(test_event_in_seq),
        'test_event_out_seq': np.array(test_event_out_seq),
        'test_time_in_seq': np.array(test_time_in_seq),
        'test_time_out_seq': np.array(test_time_out_seq),

        'num_categories': len(unique_samples),
        "minTime" : minTime,
        "maxTime" : maxTime,
        'time_scale': time_scale
    }

    return dict_res_item

def read_data_pkl(eventTrain, eventTest, timeTrain, timeTest, args,
              pad=True, paddingStyle='post'):
    flagMinMax = args.flagMinMax
    flagScale = args.flagScale

    """Read data from given files and return it as a dictionary."""
    assert len(timeTrain) == len(eventTrain)
    assert len(eventTest) == len(timeTest)


    maxTime = max(itertools.chain((max(x) for x in timeTrain), (max(x) for x in timeTest)))
    minTime = min(itertools.chain((min(x) for x in timeTrain), (min(x) for x in timeTest)))

    # maxTime = max((max(x) for x in timeTrain))
    # minTime = min((min(x) for x in timeTrain))

    # minTime, maxTime = 0, 1

    print "minTime:%f, maxTime:%f" % (minTime, maxTime)


    # nb_samples = len(eventTrain)
    # max_seqlen = max(len(x) for x in eventTrain)
    unique_samples = set()

    for x in eventTrain + eventTest:
        unique_samples = unique_samples.union(x)

    # if(flag_train_cate):
    #     item_count = cate_count
    #     eventTrain = cateTrain
    #     eventTest = cateTest

    #unique_samples = set(range(1, item_count+1))
    eventTrainIn = [x[:-1] for x in eventTrain]
    eventTrainOut = [x[1:] for x in eventTrain]

    def parse_time(y, minTime, maxTime):
        if(flagMinMax):
            res = (y - minTime) / (maxTime - minTime)
        #res = min(res, 1)
        else:
            res = y
        return res
    timeTrainIn = [[parse_time(y, minTime, maxTime) for y in x[:-1]] for x in timeTrain]
    timeTrainOut = [[parse_time(y, minTime, maxTime) for y in x[1:]] for x in timeTrain]

    if pad:
        train_event_in_seq = pad_sequences(eventTrainIn, value=0, padding=paddingStyle)
        train_event_out_seq = pad_sequences(eventTrainOut, value=0, padding=paddingStyle)
        train_time_in_seq = pad_sequences(timeTrainIn, dtype=float, padding=paddingStyle)
        train_time_out_seq = pad_sequences(timeTrainOut, dtype=float, padding=paddingStyle)
    else:
        train_event_in_seq = eventTrainIn
        train_event_out_seq = eventTrainOut
        train_time_in_seq = timeTrainIn
        train_time_out_seq = timeTrainOut


    eventTestIn = [x[:-1] for x in eventTest]
    eventTestOut = [x[1:] for x in eventTest]
    timeTestIn = [[parse_time(y, minTime, maxTime) for y in x[:-1]] for x in timeTest]
    timeTestOut = [[parse_time(y, minTime, maxTime) for y in x[1:]] for x in timeTest]

    if pad:
        test_event_in_seq = pad_sequences(eventTestIn, value=0, padding=paddingStyle)
        test_event_out_seq = pad_sequences(eventTestOut, value=0, padding=paddingStyle)
        test_time_in_seq = pad_sequences(timeTestIn, dtype=float, padding=paddingStyle)
        test_time_out_seq = pad_sequences(timeTestOut, dtype=float, padding=paddingStyle)
    else:
        test_event_in_seq = eventTestIn
        test_event_out_seq = eventTestOut
        test_time_in_seq = timeTestIn
        test_time_out_seq = timeTestOut

    if(Config.flag_test_mini):
        n_test = 1000
        test_event_in_seq = test_event_in_seq[:n_test]
        test_event_out_seq = test_event_out_seq[:n_test]
        test_time_in_seq = test_time_in_seq[:n_test]
        test_time_out_seq = test_time_out_seq[:n_test]

    print('delta-t (testing) [Initial State]= ')
    test_valid = test_event_in_seq > 0
    print(pd.Series((test_time_out_seq - test_time_in_seq)[test_valid]).describe())


    time_scale = 1
    if(flagMinMax):
        time_scale = maxTime - minTime

    if(flagScale):
        train_time_in_seq /= args.scale
        train_time_out_seq /= args.scale
        test_time_in_seq /= args.scale
        test_time_out_seq /= args.scale
        time_scale *= args.scale

    train_cnt = int(len(train_event_in_seq) * float(7 / 8.))
    print("train_event_in_seq", train_event_in_seq.shape)

    #get std
    train_valid = train_event_in_seq > 0
    std=pd.Series((train_time_out_seq - train_time_in_seq)[train_valid]).describe()["std"]
    square_std=std**2

    dict_res_item = {
        'train_event_in_seq': np.array(train_event_in_seq[:train_cnt, :]),
        'train_event_out_seq': np.array(train_event_out_seq[:train_cnt, :]),
        'train_time_in_seq': np.array(train_time_in_seq[:train_cnt, :]),
        'train_time_out_seq': np.array(train_time_out_seq[:train_cnt, :]),

        'eval_event_in_seq': np.array(train_event_in_seq[train_cnt:, :]),
        'eval_event_out_seq': np.array(train_event_out_seq[train_cnt:, :]),
        'eval_time_in_seq': np.array(train_time_in_seq[train_cnt:, :]),
        'eval_time_out_seq': np.array(train_time_out_seq[train_cnt:, :]),

        'test_event_in_seq': np.array(test_event_in_seq),
        'test_event_out_seq': np.array(test_event_out_seq),
        'test_time_in_seq': np.array(test_time_in_seq),
        'test_time_out_seq': np.array(test_time_out_seq),

        'num_categories': len(unique_samples),
        "minTime": minTime,
        "maxTime": maxTime,
        'time_scale': time_scale,
        'square_std': square_std
    }

    print "train:%d, eval:%d, test:%d" % (len(dict_res_item["train_event_in_seq"]), len(dict_res_item["eval_event_in_seq"]), len(dict_res_item["test_event_in_seq"]))
    print "minTime:%f, maxTime:%f" % (dict_res_item["minTime"], dict_res_item["maxTime"])

    return dict_res_item



def calc_base_rate(data, training=True):
    """Calculates the base-rate for intelligent parameter initialization from the training data."""
    suffix = 'train' if training else 'test'
    in_key = suffix + '_time_in_seq'
    out_key = suffix + '_time_out_seq'
    valid_key = suffix + '_event_in_seq'

    dts = (data[out_key] - data[in_key])[data[valid_key] > 0]
    return 1.0 / np.mean(dts)


def calc_base_event_prob(data, training=True):
    """Calculates the base probability of event types for intelligent parameter initialization from the training data."""
    dict_key = 'train_event_in_seq' if training else 'test_event_in_seq'

    class_count = defaultdict(lambda: 0.0)
    for evts in data[dict_key]:
        for ev in evts:
            class_count[ev] += 1.0

    total_events = 0.0
    probs = []
    for cat in range(1, data['num_categories'] + 1):
        total_events += class_count[cat]

    for cat in range(1, data['num_categories'] + 1):
        probs.append(class_count[cat] / total_events)

    return np.array(probs)


def data_stats(data):
    """Prints basic statistics about the dataset."""
    train_valid = data['train_event_in_seq'] > 0
    test_valid = data['test_event_in_seq'] > 0

    print('Num categories = ', data['num_categories'])
    print('delta-t (training) = ')
    print(pd.Series((data['train_time_out_seq'] - data['train_time_in_seq'])[train_valid]).describe())
    train_base_rate = calc_base_rate(data, training=True)
    print('base-rate = {}, log(base_rate) = {}'.format(train_base_rate, np.log(train_base_rate)))
    print('Class probs = ', calc_base_event_prob(data, training=True))

    print('delta-t (testing) = ')
    print(pd.Series((data['test_time_out_seq'] - data['test_time_in_seq'])[test_valid]).describe())
    test_base_rate = calc_base_rate(data, training=False)
    print('base-rate = {}, log(base_rate) = {}'.format(test_base_rate, np.log(test_base_rate)))
    print('Class probs = ', calc_base_event_prob(data, training=False))

    print('Training sequence lenghts = ')
    print(pd.Series(train_valid.sum(axis=1)).describe())

    print('Testing sequence lenghts = ')
    print(pd.Series(test_valid.sum(axis=1)).describe())


def variable_summaries(var, name=None):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    if name is None:
        name = var.name.split('/')[-1][:-2]

    with tf.name_scope('summaries-' + name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def MAE(time_preds, time_true, events_out):
    """Calculates the MAE between the provided and the given time, ignoring the inf
    and nans. Returns both the MAE and the number of items considered."""

    # Predictions may not cover the entire time dimension.
    # This clips time_true to the correct size.
    seq_limit = time_preds.shape[1]
    clipped_time_true = time_true[:, :seq_limit]
    clipped_events_out = events_out[:, :seq_limit]

    is_finite = np.isfinite(time_preds) & (clipped_events_out > 0)

    return np.mean(np.abs(time_preds - clipped_time_true)[is_finite]), np.sum(is_finite)

def MAE_last(time_preds, time_true):
    return Metrics.MAE(time_preds, time_true)

def Rank_last(event_preds, event_true, K):
    print "event_preds:", event_preds.shape
    print "event_preds[0]:", event_preds[0]
    print "event_true:", event_true.shape
    print "event_true[0]:", event_true[0]
    print "event_preds[0][event_true[0]]:", event_preds[0][event_true[0]]
    (N, acc, NDCG_K, Hit_K) = Metrics.Rank(event_preds, event_true, K)
    return (N, acc, NDCG_K, Hit_K)

def ACC(event_preds, event_true):
    """Returns the accuracy of the event prediction, provided the output probability vector."""
    #print("event_true:", event_true[0])
    clipped_event_true = event_true[:, :event_preds.shape[1]]
    is_valid = clipped_event_true > 0

    # The indexes start from 0 whereare event_preds start from 1.
    highest_prob_ev = event_preds.argmax(axis=-1) + 1

    #print("highest_prob_ev:", highest_prob_ev[0])
    return np.sum((highest_prob_ev == clipped_event_true)[is_valid]) / float(np.sum(is_valid))
