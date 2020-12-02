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

random.seed(1234)


def build_dataset_train(folder_src, folder_dst, T, flag_split_user_seq_OnlyOneTrainFromLast=True):
  print("build_dataset_train...")

  file_dataset = os.path.join(folder_src, Config.dict_pkl_dataset["dataset"])

  (dict_user_dataset, cate_list, user_count, item_count, cate_count, dict_item_cate, dict_user_test_negN) = Dataset.load_dataset_dataset_pkl(file_dataset)

  valid_user_count = 0

  train_set = []
  valid_set = []
  test_set = []

  print("Users: %d" % len(dict_user_dataset))
  #split user sequence
  for user_id in dict_user_dataset.keys():
    Tool.output("user_id", user_id, Config.flag_debug)
    [user_item_list, user_cate_list, user_time_list, user_time_list_ts] = dict_user_dataset[user_id]
    [neg_list_N_item_test, neg_list_N_cate_test, neg_list_N_item_valid, neg_list_N_cate_valid] =  dict_user_test_negN[user_id]

    Tool.output("user_item_list", user_item_list, Config.flag_debug)
    seq_len = len(user_item_list)
    if(seq_len < 3):
      continue

    valid_user_count += 1

    #test
    for index in [seq_len-1, seq_len-2]:
      hist_item = user_item_list[max(index - T, 0):index]
      hist_cate = Dataset.trans_itemList_to_cateList(hist_item, dict_item_cate)
      hist_time_ts = user_time_list_ts[max(index - T, 0):index]
      cur_item = user_item_list[index]
      cur_cate = dict_item_cate[cur_item]
      cur_time_ts = user_time_list_ts[index]

      #test: [user, hist_item, hist_time_ts,]
      if(index == seq_len-1):
        test_set.append([user_id, (hist_item, cur_item, neg_list_N_item_test), (hist_cate, cur_cate, neg_list_N_cate_test), (hist_time_ts, cur_time_ts)])
        Tool.output("test_set hist_item", hist_item, Config.flag_debug)
        Tool.output("test_set cur_item", cur_item, Config.flag_debug)
        Tool.output("neg_list_N_item_test", neg_list_N_item_test, Config.flag_debug)

      #valid
      if (index == seq_len - 2):
        valid_set.append([user_id, (hist_item, cur_item, neg_list_N_item_valid), (hist_cate, cur_cate, neg_list_N_cate_valid), (hist_time_ts, cur_time_ts)])
        Tool.output("valid_set hist_item", hist_item, Config.flag_debug)
        Tool.output("valid_set cur_item", cur_item, Config.flag_debug)
    #train: [user, hist_item, hist_time_ts]  max len:T+1
    for index_right in range(seq_len-2, 1, -T):
      index_left = max(0, index_right-(T+1))
      #check len
      cur_len = index_right - index_left
      if (cur_len < 2):
        continue

      index = index_left
      #hist
      hist_item = user_item_list[index:index_right]
      hist_time_ts = user_time_list_ts[index:index_right]
      hist_cate = Dataset.trans_itemList_to_cateList(hist_item, dict_item_cate)

      train_set.append([user_id, hist_item, hist_cate, hist_time_ts])


      Tool.output("train_set hist_item", hist_item, Config.flag_debug)
      if(flag_split_user_seq_OnlyOneTrainFromLast):
        break

  print("valid_user_count:%d, train_set:%d, valid_test:%d, test_set:%d" % (valid_user_count, len(train_set), len(valid_set), len(test_set)))


  #write pkl dataset
  dataset_tag = "dataset_train"
  file_pkl_dataset = os.path.join(folder_dst, Config.dict_pkl_dataset[dataset_tag])
  print("write pkl dataset:%s" % file_pkl_dataset)

  with open(file_pkl_dataset, 'wb') as f:
    pickle.dump((train_set, valid_set, test_set), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(dict_item_cate, f, pickle.HIGHEST_PROTOCOL)

  return valid_user_count
  print("build done!")


def build_dataset_train_seq_split_time_splitGap(folder_src, folder_dst, T, flag_split_user_seq_OnlyOneTrainFromLast=True):
  print("build_dataset_train_seq_split_time_splitGap...")

  file_dataset = os.path.join(folder_src, Config.dict_pkl_dataset["dataset_splitGap"])

  (dict_user_dataset, cate_list, user_count, item_count, cate_count, dict_item_cate, dict_user_test_negN) = Dataset.load_dataset_dataset_pkl(file_dataset)

  valid_user_count = 0

  data = {
    "train": {
      "event": [],
      "time": [],
    },
    "test": {
      "event": [],
      "time": [],
    }
  }

  print("Users: %d" % len(dict_user_dataset))

  train_ratio=0.8
  #split user sequence
  user_id_list = dict_user_dataset.keys()
  user_id_list_len = len(user_id_list)
  user_id_list_train_len = int(user_id_list_len * train_ratio)

  for index in range(user_id_list_len):
    user_id = user_id_list[index]
    Tool.output("user_id", user_id, Config.flag_debug)
    list_seq = dict_user_dataset[user_id]
    [neg_list_N_item, neg_list_N_cate] = dict_user_test_negN[user_id]

    for index_seq in range(len(list_seq)):
      [user_item_list, user_cate_list, user_time_list, user_time_list_ts] = list_seq[index_seq]
      if(index <= user_id_list_train_len):
        data["train"]["event"].append(user_cate_list)
        data["train"]["time"].append(user_time_list)
      else:
        data["test"]["event"].append(user_cate_list)
        data["test"]["time"].append(user_time_list)

  print("train_set:%d, test_set:%d" % (len(data["train"]["event"]), len(data["test"]["event"])))

  #write to file
  Tool.write_list_list_to_file(data["train"]["event"], os.path.join(folder_dst, "event-train.txt"))
  Tool.write_list_list_to_file(data["test"]["event"], os.path.join(folder_dst, "event-test.txt"))
  Tool.write_list_list_to_file(data["train"]["time"], os.path.join(folder_dst, "time-train.txt"))
  Tool.write_list_list_to_file(data["test"]["time"], os.path.join(folder_dst, "time-test.txt"))

  print("build done!")
#
# def build_dataset_train_EnDe_old(folder_src, folder_dst, T, flag_split_user_seq_OnlyOneTrainFromLast=True):
#   print("build_dataset_train_EnDe...")
#
#   file_dataset = os.path.join(folder_src, Config.dict_pkl_dataset["dataset"])
#
#   (dict_user_dataset, cate_list, user_count, item_count, cate_count, dict_item_cate, dict_user_test_negN) = Dataset.load_dataset_dataset_pkl(file_dataset)
#
#   valid_user_count = 0
#
#   train_set = []
#   valid_set = []
#   test_set = []
#
#   print("Users: %d" % len(dict_user_dataset))
#   #split user sequence
#   for user_id in dict_user_dataset.keys():
#     Tool.output("user_id", user_id, Config.flag_debug)
#     [user_item_list, user_cate_list, user_time_list, user_time_list_ts] = dict_user_dataset[user_id]
#     [neg_list_N_item_test, neg_list_N_cate_test, neg_list_N_item_valid, neg_list_N_cate_valid] =  dict_user_test_negN[user_id]
#
#     Tool.output("user_item_list", user_item_list, Config.flag_debug)
#     seq_len = len(user_item_list)
#     if(seq_len < 3):
#       continue
#
#     valid_user_count += 1
#
#     #test
#     for index in [seq_len-1, seq_len-2]:
#       hist_item = user_item_list[max(index - T, 0):index]
#       hist_cate = Dataset.trans_itemList_to_cateList(hist_item, dict_item_cate)
#       hist_time_ts = user_time_list_ts[max(index - T, 0):index]
#       cur_item = user_item_list[index]
#       cur_cate = dict_item_cate[cur_item]
#       cur_time_ts = user_time_list_ts[index]
#
#       #test: [user, hist_item, hist_time_ts,]
#       if(index == seq_len-1):
#         test_set.append([user_id, (hist_item, cur_item, neg_list_N_item_test), (hist_cate, cur_cate, neg_list_N_cate_test), (hist_time_ts, cur_time_ts)])
#         Tool.output("test_set hist_item", hist_item, Config.flag_debug)
#         Tool.output("test_set cur_item", cur_item, Config.flag_debug)
#         Tool.output("neg_list_N_item_test", neg_list_N_item_test, Config.flag_debug)
#
#       #valid
#       if (index == seq_len - 2):
#         valid_set.append([user_id, (hist_item, cur_item, neg_list_N_item_valid), (hist_cate, cur_cate, neg_list_N_cate_valid), (hist_time_ts, cur_time_ts)])
#         Tool.output("valid_set hist_item", hist_item, Config.flag_debug)
#         Tool.output("valid_set cur_item", cur_item, Config.flag_debug)
#     #train: [user, hist_item, hist_time_ts]  max len:T+1
#     for index_right in range(seq_len-2, 1, -T):
#       index_left = max(0, index_right-(T+1))
#       #check len
#       cur_len = index_right - index_left
#       if (cur_len < 2):
#         continue
#
#       index = index_left
#       #hist
#       hist_item = user_item_list[index:index_right]
#       hist_time_ts = user_time_list_ts[index:index_right]
#       hist_cate = Dataset.trans_itemList_to_cateList(hist_item, dict_item_cate)
#
#       train_set.append([user_id, hist_item, hist_cate, hist_time_ts])
#
#
#       Tool.output("train_set hist_item", hist_item, Config.flag_debug)
#       if(flag_split_user_seq_OnlyOneTrainFromLast):
#         break
#
#   print("valid_user_count:%d, train_set:%d, valid_test:%d, test_set:%d" % (valid_user_count, len(train_set), len(valid_set), len(test_set)))
#
#
#   #write pkl dataset
#   dataset_tag = "dataset_train"
#   file_pkl_dataset = os.path.join(folder_dst, Config.dict_pkl_dataset[dataset_tag])
#   print("write pkl dataset:%s" % file_pkl_dataset)
#
#   with open(file_pkl_dataset, 'wb') as f:
#     pickle.dump((train_set, valid_set, test_set), f, pickle.HIGHEST_PROTOCOL)
#     pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
#     pickle.dump(dict_item_cate, f, pickle.HIGHEST_PROTOCOL)
#
#   return valid_user_count
#   print("build done!")
#



def build_dataset_train_EnDe(folder_src, folder_dst, T, flag_split_user_seq_OnlyOneTrainFromLast=True):
  print("build_dataset_train_EnDe...")

  file_dataset = os.path.join(folder_src, Config.dict_pkl_dataset["dataset"])

  (dict_user_dataset, cate_list, user_count, item_count, cate_count, dict_item_cate, dict_user_test_negN) = Dataset.load_dataset_dataset_pkl(file_dataset)

  valid_user_count = 0

  train_set = []
  valid_set = []
  test_set = []

  print("Users: %d" % len(dict_user_dataset))
  #split user sequence
  for user_id in dict_user_dataset.keys():
    Tool.output("user_id", user_id, Config.flag_debug)
    [user_item_list, user_cate_list, user_time_list, user_time_list_ts] = dict_user_dataset[user_id]
    [neg_list_N_item_test, neg_list_N_cate_test, neg_list_N_item_valid, neg_list_N_cate_valid] =  dict_user_test_negN[user_id]

    Tool.output("user_item_list", user_item_list, Config.flag_debug)
    seq_len = len(user_item_list)
    if(seq_len < 3):
      continue

    valid_user_count += 1


    neg_item_list = Dataset.random_neq_N_seq(1, item_count, user_item_list, seq_len)
    neg_cate_list = Dataset.random_neq_N_seq(1, cate_count, user_cate_list, seq_len)

    #neg_item_list = Dataset.random_neq_N_seq_each(1, item_count+1, user_item_list, seq_len)
    #neg_cate_list = Dataset.random_neq_N_seq_each(1, cate_count+1, user_cate_list, seq_len)

    #test
    for index in [seq_len-1, seq_len-2]:
      hist_item = user_item_list[:index]
      hist_cate = Dataset.trans_itemList_to_cateList(hist_item, dict_item_cate)
      hist_time_ts = user_time_list_ts[:index]

      #hist_time = Dataset.proc_time_emb(user_time_list[:index], user_time_list[index])

      cur_item = user_item_list[index]
      cur_cate = dict_item_cate[cur_item]
      cur_time_ts = user_time_list_ts[index]

      #test: [user, hist_item, hist_time_ts,]
      if(index == seq_len-1):
        test_set.append([user_id, (hist_item, cur_item, neg_list_N_item_test), (hist_cate, cur_cate, neg_list_N_cate_test), (hist_time_ts, cur_time_ts)])
        Tool.output("test_set hist_item", hist_item, Config.flag_debug)
        Tool.output("test_set cur_item", cur_item, Config.flag_debug)
        Tool.output("neg_list_N_item_test", neg_list_N_item_test, Config.flag_debug)

      #valid
      if (index == seq_len - 2):
        valid_set.append([user_id, (hist_item, cur_item, neg_list_N_item_valid), (hist_cate, cur_cate, neg_list_N_cate_valid), (hist_time_ts, cur_time_ts)])
        Tool.output("valid_set hist_item", hist_item, Config.flag_debug)
        Tool.output("valid_set cur_item", cur_item, Config.flag_debug)

    #train: [user, hist_item, hist_time_ts]  max len:T+1
    for index in range(seq_len-3, 0, -1):
      #hist
      hist_item = user_item_list[:index]
      hist_time_ts = user_time_list_ts[:index]
      hist_cate = Dataset.trans_itemList_to_cateList(hist_item, dict_item_cate)

      hist_time = Dataset.proc_time_emb(user_time_list[:index], user_time_list[index])

      cur_item = user_item_list[index]
      cur_cate = dict_item_cate[cur_item]
      cur_time_ts = user_time_list_ts[index]

      train_set.append([user_id, (hist_item, cur_item, neg_item_list[index]), (hist_cate, cur_cate, neg_cate_list[index]), (hist_time_ts, cur_time_ts)])

      Tool.output("train_set hist_item", hist_item, Config.flag_debug)
      if(flag_split_user_seq_OnlyOneTrainFromLast):
       break

    if(valid_user_count % 1000 == 0):
      print("processed %d / %d" % (valid_user_count, len(dict_user_dataset)))

  print("valid_user_count:%d, train_set:%d, valid_test:%d, test_set:%d" % (valid_user_count, len(train_set), len(valid_set), len(test_set)))


  #write pkl dataset
  dataset_tag = "AMPP_EnDe"
  file_pkl_dataset = os.path.join(folder_dst, Config.dict_pkl_dataset[dataset_tag])
  print("write pkl dataset:%s" % file_pkl_dataset)

  with open(file_pkl_dataset, 'wb') as f:
    pickle.dump((train_set, valid_set, test_set), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(dict_item_cate, f, pickle.HIGHEST_PROTOCOL)

  return valid_user_count
  print("build done!")


def build_dataset_train_ATRank(folder_src, folder_dst):
  print("build_dataset_train_ATRank...")

  file_dataset = os.path.join(folder_src, Config.dict_pkl_dataset["dataset"])

  (dict_user_dataset, cate_list, user_count, item_count, cate_count, dict_item_cate, dict_user_test_negN) = Dataset.load_dataset_dataset_pkl(file_dataset)

  valid_user_count = 0

  print("item_count:%d, cate_count:%d" % (item_count, cate_count))

  train_set = []
  valid_set = []
  test_set = []

  print("Users: %d" % len(dict_user_dataset))
  #split user sequence
  for user_id in dict_user_dataset.keys():
    Tool.output("user_id", user_id, Config.flag_debug)
    [user_item_list, user_cate_list, user_time_list, user_time_list_ts] = dict_user_dataset[user_id]
    [neg_list_N_item_test, neg_list_N_cate_test, neg_list_N_item_valid, neg_list_N_cate_valid] =  dict_user_test_negN[user_id]

    Tool.output("user_item_list", user_item_list, Config.flag_debug)
    seq_len = len(user_item_list)
    if(seq_len < 3):
      continue

    valid_user_count += 1

    neg_item_list = Dataset.random_neq_N_seq(1, item_count, user_item_list, seq_len)
    #neg_item_cate_list = Dataset.trans_itemList_to_cateList(neg_item_list, dict_item_cate)
    neg_cate_list = Dataset.random_neq_N_seq(1, cate_count, user_cate_list, seq_len)

    #test
    for index in [seq_len-1, seq_len-2]:
      hist_item = user_item_list[:index]
      hist_cate = Dataset.trans_itemList_to_cateList(hist_item, dict_item_cate)
      hist_time_ts = user_time_list_ts[:index]

      hist_time = Dataset.proc_time_emb(user_time_list[:index], user_time_list[index])

      cur_item = user_item_list[index]
      cur_cate = dict_item_cate[cur_item]
      cur_time_ts = user_time_list_ts[index]

      #test: [user, hist_item, hist_time_ts,]
      if(index == seq_len-1):
        test_set.append([user_id, (hist_item, cur_item, neg_list_N_item_test), (hist_cate, cur_cate, neg_list_N_cate_test), (hist_time, cur_time_ts)])
        Tool.output("test_set hist_item", hist_item, Config.flag_debug)
        Tool.output("test_set cur_item", cur_item, Config.flag_debug)
        Tool.output("neg_list_N_item_test", neg_list_N_item_test, Config.flag_debug)

      #valid
      if (index == seq_len - 2):
        valid_set.append([user_id, (hist_item, cur_item, neg_list_N_item_valid), (hist_cate, cur_cate, neg_list_N_cate_valid), (hist_time, cur_time_ts)])
        Tool.output("valid_set hist_item", hist_item, Config.flag_debug)
        Tool.output("valid_set cur_item", cur_item, Config.flag_debug)

    #train: [user, hist_item, hist_time_ts]  max len:T+1
    for index in range(seq_len-3, 0, -1):
      #hist
      hist_item = user_item_list[:index]
      hist_time_ts = user_time_list_ts[:index]
      hist_cate = Dataset.trans_itemList_to_cateList(hist_item, dict_item_cate)

      hist_time = Dataset.proc_time_emb(user_time_list[:index], user_time_list[index])

      cur_item = user_item_list[index]
      cur_cate = dict_item_cate[cur_item]
      cur_time_ts = user_time_list_ts[index]

      train_set.append([user_id, (hist_item, cur_item), (hist_cate, cur_cate), (hist_time, cur_time_ts), 1])
      train_set.append([user_id, (hist_item, neg_item_list[index]), (hist_cate, neg_cate_list[index]), (hist_time, cur_time_ts), 0])

      Tool.output("train_set hist_item", hist_item, Config.flag_debug)
      #if(flag_split_user_seq_OnlyOneTrainFromLast):
      #  break

    if(valid_user_count % 1000 == 0):
      print("processed %d / %d" % (valid_user_count, len(dict_user_dataset)))

  print("valid_user_count:%d, train_set:%d, valid_test:%d, test_set:%d" % (valid_user_count, len(train_set), len(valid_set), len(test_set)))


  #write pkl dataset
  dataset_tag = "AMPP_EnDe_ATRank"
  file_pkl_dataset = os.path.join(folder_dst, Config.dict_pkl_dataset[dataset_tag])
  print("write pkl dataset:%s" % file_pkl_dataset)

  with open(file_pkl_dataset, 'wb') as f:
    pickle.dump((train_set, valid_set, test_set), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(dict_item_cate, f, pickle.HIGHEST_PROTOCOL)

  return valid_user_count
  print("build done!")

def save_sequences_user_split(user_item_list,  user_time_list_ts, dict_user_dataset, dict_user_test_negN, user_id, user_split_index_arr):
  dict_user_dataset[user_id] = []
  N_seq_mini = len(user_split_index_arr)
  for i in range(N_seq_mini):
    seq_i = user_split_index_arr[i]
    user_time_list_ts_cur = []
    user_item_list_cur = []
    for index in seq_i:
      user_time_list_ts_cur.append(user_time_list_ts[index])
      user_item_list_cur.append(user_item_list[index])
    dict_user_dataset[user_id].append([user_item_list_cur, user_time_list_ts_cur])

    if (i == N_seq_mini - 1):
      user_cate_list_cur = [dict_item_cate[i] for i in user_item_list_cur]
      # neg
      neg_list_N_item = gen_neg_N(user_item_list_cur, item_count, Config.test_neg_N_item)
      neg_list_N_cate = gen_neg_N(user_cate_list_cur, cate_count, Config.test_neg_N_cate)

  dict_user_test_negN[user_id] = [neg_list_N_item, neg_list_N_cate]

#write pkl dataset
def write_pkl_dataset(file_pkl_dataset, dict_user_dataset, cate_list, user_count, item_count, cate_count, dict_item_cate, dict_user_test_negN):
  with open(file_pkl_dataset, 'wb') as f:
    pickle.dump(dict_user_dataset, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(dict_item_cate, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(dict_user_test_negN, f, pickle.HIGHEST_PROTOCOL)

def build_dataset_split_time_T():
  dataset_tag = "dataset_T"
  print("build_dataset_split_time_T...")
  dict_user_dataset = {}
  dict_user_test_negN = {}

  file_pkl_dataset = os.path.join(folder_dst, Config.dict_pkl_dataset[dataset_tag])
  valid_user_count = 0
  for user_id, hist in user_df.groupby('user'):
    user_item_list = hist['item'].tolist()
    user_time_list_ts = hist['timestamp'].tolist()
    seq_len = len(user_item_list)
    if (not(seq_len >= Config.sequence_mini_min_len)):
      continue

    #split sequence by T
    user_split_index_arr = []
    index_cur = [0]
    start_index=0
    for index in range(seq_len):
      #gap > threshold
      if(Time.get_time_diff_days(user_time_list_ts[start_index], user_time_list_ts[index]) <= Config.split_sequence_time_T):
        index_cur.append(index)
      else:
        if (len(index_cur) >= Config.sequence_mini_min_len):
          user_split_index_arr.append(index_cur)
        index_cur = [index]
        start_index = index
    if(len(index_cur) >= Config.sequence_mini_min_len):
      user_split_index_arr.append(index_cur)

    if(len(user_split_index_arr) == 0):
      continue

    valid_user_count += 1

    #save sequences of users
    save_sequences_user_split(user_item_list, user_time_list_ts, dict_user_dataset, dict_user_test_negN, user_id,
                              user_split_index_arr)

  #write pkl dataset
  write_pkl_dataset(file_pkl_dataset, dict_user_dataset, cate_list, user_count, item_count, cate_count, dict_item_cate, dict_user_test_negN)

  return valid_user_count
  print("build done!")

def build_dataset_split_time_splitGap():
  dataset_tag = "dataset_splitGap"
  # print("build_dataset_split_time_splitGap...")
  dict_user_dataset = {}
  dict_user_test_negN = {}

  file_pkl_dataset = os.path.join(folder_dst, Config.dict_pkl_dataset[dataset_tag])
  valid_user_count = 0
  for user_id, hist in user_df.groupby('user'):
    #print "\nuser_id:", user_id
    user_item_list = hist['item'].tolist()
    user_time_list_ts = hist['timestamp'].tolist()
    seq_len = len(user_item_list)
    if (not(seq_len >= Config.sequence_mini_min_len)):
      continue

    # print "user_item_list:", user_item_list
    # print "user_time_list_ts:", user_time_list_ts

    #split sequence by gap
    user_split_index_arr = []
    index_cur = []
    prev_index = -1
    for index in range(seq_len):
      #gap > threshold
      if(prev_index >=0 and Time.get_time_diff_days(user_time_list_ts[prev_index], user_time_list_ts[index]) > Config.split_sequence_time_gap_max):
        if (len(index_cur) >= Config.sequence_mini_min_len):
          user_split_index_arr.append(index_cur)
        index_cur = [index]
        prev_index = index
      else:
        index_cur.append(index)
        prev_index = index

    #append last group
    if(len(index_cur) >= Config.sequence_mini_min_len):
      user_split_index_arr.append(index_cur)

    if(len(user_split_index_arr) == 0):
      continue

    valid_user_count += 1

    #save sequences of users
    save_sequences_user_split(user_item_list, user_time_list_ts, dict_user_dataset, dict_user_test_negN, user_id,
                              user_split_index_arr)

    # print("dict_user_dataset[user_id]:")
    # for seq in dict_user_dataset[user_id]:
    #   [user_item_list_cur, user_time_list_ts_cur] = seq
      # print("\n user_item_list_cur:"), user_item_list_cur
      # print("\n user_time_list_ts_cur:"), user_time_list_ts_cur

    #if(valid_user_count > 10):
    #  break

  # write pkl dataset
  write_pkl_dataset(file_pkl_dataset, dict_user_dataset, cate_list, user_count, item_count, cate_count, dict_item_cate, dict_user_test_negN)

  print("build done!")
  print("valid_user_count:", valid_user_count)
  return valid_user_count




def run(folder_src, folder_dst, test_neg_N_item, test_neg_N_cat):
  print("run...")
  #user_num_0 = build_test_user_negN(item_count)
  #user_num_2 = build_SASRec()

  build_dataset_train(folder_src, folder_dst, Config.max_seq_len, Config.flag_split_user_seq_OnlyOneTrainFromLast)

  build_dataset_train_seq_split_time_splitGap(folder_src, folder_dst, Config.max_seq_len)

  #build_dataset_train_ATRank(folder_src, folder_dst)

  #build_dataset_train_EnDe(folder_src, folder_dst, Config.max_seq_len, Config.flag_split_user_seq_OnlyOneTrainFromLast)

  #print build_dataset_split_time_splitGap()

  #assert(user_num_0 == user_num_1 == user_num_2)


  #SASRec

  # train_set_pos_neg.append((user_id, hist_i, hist_t, user_item_list[i], neg_list[i], hist_t_ts))
  #
  # seq = np.zeros([Config.max_seq_len], dtype=np.int32)
  # pos = np.zeros([Config.max_seq_len], dtype=np.int32)
  # neg = np.zeros([Config.max_seq_len], dtype=np.int32)
  # nxt = userlist[-1]
  # idx = Config.max_seq_len - 1
  #
  # ts = set(user_seq_list)
  # for i in reversed(user_seq_list[:-1]):
  #   seq[idx] = i
  #   pos[idx] = nxt
  #   if nxt != 0:
  #     neg[idx] = random_neq(1, item_count+ 1, ts)
  #   nxt = i
  #   idx -= 1
  #   if idx == -1: break
  #
  # train_set_pos_neg.append(user, seq, pos, neg)




if __name__ == "__main__":
  print "\n4_build_dataset train..."
  # parse parameters
  arr = sys.argv
  assert (len(arr) == 1 + 4)
  folder_src, folder_dst, test_neg_N_item, test_neg_N_cate = arr[1], arr[2],  int(arr[3]), int(
    arr[4])
  run(folder_src, folder_dst, test_neg_N_item, test_neg_N_cate)

  print "4_build_dataset train end..."


