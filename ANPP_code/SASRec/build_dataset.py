import random
import pickle
import numpy as np
import sys
import os

sys.path.append("..")
from Config import *
from utils.Dataset import *


random.seed(1234)


def build_SASRec():
  print("build_SASRec...")

  # src: dataset.pickle
  file_pkl_dataset = os.path.join(folder_dataset, Config.dict_pkl_dataset["dataset"])
  (dict_user_dataset, cate_list, user_count, item_count, cate_count, dict_item_cate, dict_user_test_negN) = \
    Dataset.load_dataset_dataset_pkl(file_pkl_dataset)

  # dst: dataset_model.pickle
  file_pkl_dataset_model = os.path.join(folder_dataset, Config.dict_pkl_dataset["SASRec"])

  #item
  user_train_item = {}
  user_valid_item = {}
  user_test_item = {}

  #cate
  user_train_cate = {}
  user_valid_cate = {}
  user_test_cate = {}


  for user_id in dict_user_dataset.keys():
    [user_item_list, user_cate_list, user_time_list, user_time_list_ts] = dict_user_dataset[user_id]

    #item
    user_train_item[user_id] = user_item_list[:-2]
    user_valid_item[user_id] = [user_item_list[-2]]
    user_test_item[user_id] = [user_item_list[-1]]

    #cate
    user_train_cate[user_id] = user_cate_list[:-2]
    user_valid_cate[user_id] = [user_cate_list[-2]]
    user_test_cate[user_id] = [user_cate_list[-1]]

  #write pkl dataset
  with open(file_pkl_dataset_model, 'wb') as f:
    #item
    pickle.dump((user_train_item, user_valid_item, user_test_item), f, pickle.HIGHEST_PROTOCOL)
    #cate
    pickle.dump((user_train_cate, user_valid_cate, user_test_cate), f, pickle.HIGHEST_PROTOCOL)
    #other
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(dict_item_cate, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(dict_user_test_negN, f, pickle.HIGHEST_PROTOCOL)

  print("build done!")


def load_dataset_SASRec(file):
  print('load_dataset_SASRec..')
  with open(file, 'rb') as f:
    (user_train_item, user_valid_item, user_test_item) = pickle.load(f)
    (user_train_cate, user_valid_cate, user_test_cate) = pickle.load(f)
    cate_list = pickle.load(f)
    user_count, item_count, cate_count = pickle.load(f)
    dict_item_cate = pickle.load(f)
    dict_user_test_negN = pickle.load(f)

    return (user_train_item, user_valid_item, user_test_item, user_train_cate, user_valid_cate, user_test_cate, cate_list, user_count, item_count, cate_count, dict_item_cate,
            dict_user_test_negN)


def run():
  build_SASRec()

if __name__ == "__main__":

  arr = sys.argv
  assert (len(arr) == 1 + 3)
  folder_dataset, folder_dataset_model, test_neg_N_item = arr[1], arr[2], int(arr[3])
  run()
