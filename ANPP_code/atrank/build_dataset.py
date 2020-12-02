import random
import pickle
import numpy as np
import sys
import os

sys.path.append("..")
from Config import *
from utils.Dataset import *


random.seed(1234)

print "build_dataset begin..."
arr = sys.argv
assert(len(arr) == 1+3)
folder_dataset, folder_dataset_model, test_neg_N_item = arr[1], arr[2], int(arr[3])

def build_atrank():
  print("build_atrank...")

  # src: dataset.pickle
  file_pkl_dataset = os.path.join(folder_dataset, Config.dict_pkl_dataset["dataset"])
  (dict_user_dataset, cate_list, user_count, item_count, cate_count, dict_item_cate, dict_user_test_negN) = \
    Dataset.load_dataset_dataset_pkl(file_pkl_dataset)

  # dst: dataset_model.pickle
  file_pkl_dataset_model = os.path.join(folder_dataset, Config.dict_pkl_dataset["atrank"])

  train_set = []
  test_set = []

  for user_id in dict_user_dataset.keys():
    [user_item_list, user_cate_list, user_time_list, user_time_list_ts] = dict_user_dataset[user_id]
    seq_len = len(user_item_list)
    def gen_neg():
      neg = user_item_list[0]
      while neg in user_item_list:
        neg = random.randint(1, item_count + 1)
      return neg

    neg_list = [gen_neg() for i in range(len(user_item_list))]
    for i in range(1, seq_len):
      hist_i = user_item_list[:i]
      hist_t = Dataset.proc_time_emb(user_time_list[:i], user_time_list[i])
      # test
      if i == seq_len - 1:
        label = (user_item_list[i], neg_list[i])
        test_set.append((user_id, hist_i, hist_t, label))
      # valid
      #if i == seq_len - 2:
      #  label = (user_item_list[i], neg_list[i])
      #  valid_set.append((user_id, hist_i, hist_t, label, hist_t_ts))
      # train
      else:
        train_set.append((user_id, hist_i, hist_t, user_item_list[i], 1))
        train_set.append((user_id, hist_i, hist_t, neg_list[i], 0))

  #write pkl dataset
  with open(file_pkl_dataset_model, 'wb') as f:
    pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(dict_item_cate, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(dict_user_test_negN, f, pickle.HIGHEST_PROTOCOL)

  print("build done!")

build_atrank()

print "build_dataset done!"