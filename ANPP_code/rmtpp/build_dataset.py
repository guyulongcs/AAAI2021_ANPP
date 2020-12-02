import random
import pickle
import numpy as np
import sys
import os

sys.path.append("..")
from Config import *
from utils.Dataset import *
from utils.Tool import *


random.seed(1234)

print "build_dataset begin..."
arr = sys.argv
assert(len(arr) == 1+2)
folder_dataset, folder_dataset_model = arr[1], arr[2]

def build_dataset():
  print("build_dataset...")
  model_tag = "rmtpp"
  dataset_name = "dataset_splitGap"

  # src: dataset.pickle
  file_pkl_dataset = os.path.join(folder_dataset, Config.dict_pkl_dataset[dataset_name])
  (dict_user_dataset, cate_list, user_count, item_count, cate_count, dict_item_cate, dict_user_test_negN) = \
    Dataset.load_dataset_dataset_pkl(file_pkl_dataset)

  # dst: dataset_model.pickle
  file_pkl_dataset_model = os.path.join(folder_dataset, Config.dict_pkl_dataset[model_tag])

  event_train = []
  event_test = []
  time_train = []
  time_test = []
  cate_train = []
  cate_test = []
  event_test_negN = []

  for user_id in dict_user_dataset.keys():
    [user_item_list, user_cate_list, user_time_list, user_time_list_ts] = dict_user_dataset[user_id]
    seq_len = len(user_item_list)
    def gen_neg():
      neg = user_item_list[0]
      while neg in user_item_list:
        neg = random.randint(1, item_count + 1)
      return neg

    seq_len_valid = min(seq_len, Config.max_seq_len)
    #seq_len_valid = 1

    user_time_list_diff = [Dataset.proc_time_get_diff(user_time_list[0], t) for t in user_time_list]

    event_train.append(user_time_list[:-1])
    event_test.append(user_time_list[-seq_len_valid:])

    cate_train.append(user_cate_list[:-1])
    cate_test.append(user_cate_list[-seq_len_valid:])

    time_train.append(user_time_list_diff[:-1])
    time_test.append(user_time_list_diff[-seq_len_valid:])

    event_test_negN.append(dict_user_test_negN[user_id][0])


  #write pkl dataset
  with open(file_pkl_dataset_model, 'wb') as f:
    pickle.dump((event_train, event_test), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((cate_train, cate_test), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((time_train, time_test), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(dict_item_cate, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(event_test_negN, f, pickle.HIGHEST_PROTOCOL)

  print("build done!")




def build_dataset_splitGap():
  print("build_dataset_splitGap...")
  model_tag = "rmtpp"
  dataset_name = "dataset_splitGap"

  # src: dataset.pickle
  file_pkl_dataset = os.path.join(folder_dataset, Config.dict_pkl_dataset[dataset_name])
  (dict_user_dataset, cate_list, user_count, item_count, cate_count, dict_item_cate, dict_user_test_negN) = \
    Dataset.load_dataset_dataset_pkl(file_pkl_dataset)

  # dst: dataset_model.pickle
  file_pkl_dataset_model = os.path.join(folder_dataset, Config.dict_pkl_dataset[model_tag])

  event_train = []
  event_test = []
  time_train = []
  time_test = []
  cate_train = []
  cate_test = []
  event_test_negN = []

  valid_user_cnt=0
  for user_id in dict_user_dataset.keys():
    Tool.output("user_id", user_id, Config.flag_debug)
    N_seq = len(dict_user_dataset[user_id])

    event_train_user = []
    event_test_user = []
    time_train_user = []
    time_test_user = []
    cate_train_user = []
    cate_test_user = []

    for i_seq in range(N_seq):
      [user_item_list_cur, user_time_list_ts_cur] = dict_user_dataset[user_id][i_seq]
      Tool.output("user_item_list_cur", user_item_list_cur, Config.flag_debug)
      Tool.output("user_time_list_ts_cur", user_time_list_ts_cur, Config.flag_debug)

      user_cate_list_cur = [dict_item_cate[i] for i in user_item_list_cur]

      #every group
      user_item_train = user_item_list_cur
      user_cate_train = user_cate_list_cur
      user_time_train_ts = user_time_list_ts_cur

      #last group
      if(i_seq == N_seq -1):
        seq_len = len(user_item_list_cur)
        seq_len_valid = min(seq_len, Config.max_seq_len)
        #train
        user_item_train = user_item_train[:-1]
        user_cate_train = user_cate_train[:-1]
        user_time_train_ts = user_time_train_ts[:-1]
        #test
        event_test_user.append(user_item_list_cur[-seq_len_valid:])
        cate_test_user.append(user_cate_list_cur[-seq_len_valid:])
        time_test_user.append(user_time_list_ts_cur[-seq_len_valid:])

      #train
      event_train_user.append(user_item_train)
      cate_train_user.append(user_cate_train)
      time_train_user.append(user_time_train_ts)

    Tool.output("event_train_user", event_train_user, Config.flag_debug)
    Tool.output("time_train_user", time_train_user, Config.flag_debug)
    Tool.output("event_test_user", event_test_user, Config.flag_debug)
    Tool.output("time_test_user", time_test_user, Config.flag_debug)

    event_train.extend(event_train_user)
    event_test.extend(event_test_user)
    time_train.extend(time_train_user)
    time_test.extend(time_test_user)

    event_test_negN.append(dict_user_test_negN[user_id][0])

    valid_user_cnt += 1

    if(Config.flag_debug and valid_user_cnt > 5):
      break

  Tool.output("time_train", time_train, Config.flag_debug)

  #time_train = Dataset.pro_list_seq_time_to_timeDelta(time_train)
  #time ts seq to time day(start from 0)
  time_train = Dataset.pro_list_seq_time_to_dayStartZero(time_train)

  Tool.output("time_train new", time_train, Config.flag_debug)
    #time ts seq to time day(start from 0)
  time_test = Dataset.pro_list_seq_time_to_dayStartZero(time_test)

  cate_train = Dataset.pro_list_seqEvent_to_seqCate(event_train, dict_item_cate)
  cate_test = Dataset.pro_list_seqEvent_to_seqCate(event_test, dict_item_cate)

  #write pkl dataset
  with open(file_pkl_dataset_model, 'wb') as f:
    pickle.dump((event_train, event_test), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((cate_train, cate_test), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((time_train, time_test), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(dict_item_cate, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(event_test_negN, f, pickle.HIGHEST_PROTOCOL)

  print("build done!")


def build_dataset_rmtpp():
  print("build_dataset_rmtpp...")

  dataset_name = "dataset_seq"

  # src: dataset.pickle
  file_pkl_dataset = os.path.join(folder_dataset, Config.dict_pkl_dataset[dataset_name])
  (eventTrain, eventTest, timeTrain, timeTest) = Dataset.load_dataset_seq_pkl(file_pkl_dataset)

  # dst: dataset_model.pickle
  model_tag = "rmtpp"
  file_pkl_dataset_model = os.path.join(folder_dataset, Config.dict_pkl_dataset[model_tag])

  # write pkl dataset
  with open(file_pkl_dataset_model, 'wb') as f:
    pickle.dump((eventTrain, eventTest, timeTrain, timeTest), f, pickle.HIGHEST_PROTOCOL)

  print("build done!")


#build_dataset()
#build_dataset_splitGap()
build_dataset_rmtpp()

print "build_dataset done!"
