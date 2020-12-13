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


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def gen_neg_N(user_item_list, item_count, N):
  neg_list = []
  for i in range(N):
    neg = random.randint(1, item_count)
    while ((neg in user_item_list) or (neg in neg_list)):
      neg = random.randint(1, item_count)
    neg_list.append(neg)
  return neg_list

def build_test_user_negN(item_num):
  print("build_test_user_negN...")
  folder, item_count, neg_count =  folder_src, item_num, Config.test_neg_N_item
  file_pkl_dataset = os.path.join(folder, Config.dict_pkl_dataset["test_user_negN"])
  dict_user_neg_test_N = {}
  valid_user_count = 0
  for user_id, hist in user_df.groupby('user'):
    user_item_list = hist['item'].tolist()
    user_time_list_ts = hist['timestamp'].tolist()
    seq_len = len(user_item_list)
    if (seq_len < 4):
      continue
    valid_user_count += 1
    neg_list = gen_neg_N(user_item_list, item_count, neg_count)
    dict_user_neg_test_N[user_id] = neg_list

  #write pkl dataset
  with open(file_pkl_dataset, 'wb') as f:
    pickle.dump(dict_user_neg_test_N, f, pickle.HIGHEST_PROTOCOL)

  return valid_user_count


def build_atrank():
  print("build_atrank...")
  train_set = []
  test_set = []

  file_pkl_dataset = os.path.join(folder_dst, Config.dict_pkl_dataset["atrank"])
  valid_user_count = 0
  for user_id, hist in user_df.groupby('user'):
    user_item_list = hist['item'].tolist()
    user_time_list_ts = hist['timestamp'].tolist()
    seq_len = len(user_item_list)
    if (seq_len < 4):
      continue
    valid_user_count += 1

    user_time_list = [i // 3600 // 24 for i in user_time_list_ts]

    def gen_neg():
      neg = user_item_list[0]
      while neg in user_item_list:
        neg = random.randint(1, item_count + 1)
      return neg
    #
    # def gen_neg_N(user_item_list, N):
    #   neg_list = []
    #   for i in range(N):
    #     neg = random.randint(1, item_count + 1)
    #     while ((neg in user_item_list) or (neg in neg_list)):
    #       neg = random.randint(1, item_count + 1)
    #     neg_list.append(neg)
    #   return neg_list

    neg_list = [gen_neg() for i in range(len(user_item_list))]
    #neg_list_N_valid = gen_neg_N(user_item_list, test_neg_N_item)
    #neg_List_N_test = gen_neg_N(user_item_list, test_neg_N_item)

    # atrank
    for i in range(1, seq_len):
      hist_i = user_item_list[:i]
      hist_t = proc_time_emb(user_time_list[:i], user_time_list[i])
      #hist_t_ts = (user_time_list_ts[:i], user_time_list_ts[i])

      # test
      if i == seq_len - 1:
        label = (user_item_list[i], neg_list[i])
        test_set.append((user_id, hist_i, hist_t, label))

      else:
        train_set.append((user_id, hist_i, hist_t, user_item_list[i], 1))
        train_set.append((user_id, hist_i, hist_t, neg_list[i], 0))

  assert len(test_set) == valid_user_count
  # assert(len(test_set) + len(train_set) // 2 == user_df.shape[0])

  #write pkl dataset
  with open(file_pkl_dataset, 'wb') as f:
    pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)

  return valid_user_count
  print("build done!")

def build_SASRec():
  print("build_SASRec...")
  file_pkl_dataset = os.path.join(folder_dst, Config.dict_pkl_dataset["SASRec"])

  user_train = {}
  user_valid = {}
  user_test = {}

  valid_user_count = 0
  for user_id, hist in user_df.groupby('user'):
    user_item_list = hist['item'].tolist()
    user_time_list_ts = hist['timestamp'].tolist()
    seq_len = len(user_item_list)
    if (seq_len < 4):
      continue
    valid_user_count += 1


    user_train[user_id] = user_item_list[:-2]
    user_valid[user_id] = []
    user_valid[user_id].append(user_item_list[-2])
    user_test[user_id] = []
    user_test[user_id].append(user_item_list[-1])

  assert len(user_test) == valid_user_count
  # assert(len(test_set) + len(train_set) // 2 == user_df.shape[0])

  #write pkl dataset
  with open(file_pkl_dataset, 'wb') as f:
    pickle.dump(user_train, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(user_valid, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(user_test, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)

  return valid_user_count
  print("build done!")

def build_dataset(folder_src, file_pkl_remap, folder_dst, test_neg_N_item, test_neg_N_cate):
  dataset_tag = "dataset"
  print("build_dataset...")

  # read file remap pkl
  (
  user_df, cate_list, user_count, item_count, cate_count, example_count, item_map, cate_map, user_map, dict_item_cate) = \
    read_file_remap_pkl(folder_src, file_pkl_remap)


  dict_user_dataset = {}
  dict_user_test_negN = {}
  # train_set = []
  # valid_set = []
  # test_set = []


  valid_user_count = 0
  for user_id, hist in user_df.groupby('user'):
    Tool.output("user_id", user_id, Config.flag_debug)
    user_item_list = hist['item'].tolist()
    user_time_list_ts = hist['timestamp'].tolist()
    seq_len = len(user_item_list)
    if (seq_len < 4):
      continue
    valid_user_count += 1

    #timestamp to days
    user_time_list = [i // 3600 // 24 for i in user_time_list_ts]
    user_cate_list = [dict_item_cate[i] for i in user_item_list]
    #user_time_train = user_time_list[:-2]

    #neg
    neg_list_N_item_test = gen_neg_N(user_item_list, item_count, test_neg_N_item)
    neg_list_N_cate_test = gen_neg_N(user_cate_list, cate_count, test_neg_N_cate)

    neg_list_N_item_valid = gen_neg_N(user_item_list, item_count, test_neg_N_item)
    neg_list_N_cate_valid = gen_neg_N(user_cate_list, cate_count, test_neg_N_cate)

    dict_user_dataset[user_id] = [user_item_list, user_cate_list, user_time_list, user_time_list_ts]
    dict_user_test_negN[user_id] = [neg_list_N_item_test, neg_list_N_cate_test, neg_list_N_item_valid, neg_list_N_cate_valid]

    Tool.output("neg_list_N_item_test ", neg_list_N_item_test , Config.flag_debug)
  #write pkl dataset
  #output file
  file_pkl_dataset = os.path.join(folder_dst, Config.dict_pkl_dataset["dataset"])
  print("write %s" % file_pkl_dataset)
  with open(file_pkl_dataset, 'wb') as f:
    pickle.dump(dict_user_dataset, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(dict_item_cate, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(dict_user_test_negN, f, pickle.HIGHEST_PROTOCOL)

  print("build done!")
  return valid_user_count


def save_sequences_user_split(user_item_list, user_cate_list, user_time_list_ts, user_time_list, dict_user_dataset, dict_user_test_negN, user_id, user_split_index_arr, dict_item_cate, item_count, cate_count):
  dict_user_dataset[user_id] = []
  N_seq_mini = len(user_split_index_arr)
  for i in range(N_seq_mini):
    seq_i = user_split_index_arr[i]
    user_time_list_ts_cur = []
    user_time_list_cur = []
    user_item_list_cur = []
    user_cate_list_cur = []

    for index in seq_i:
      user_time_list_ts_cur.append(user_time_list_ts[index])
      user_time_list_cur.append(user_time_list[index])
      user_item_list_cur.append(user_item_list[index])
      user_cate_list_cur.append(user_cate_list[index])

    dict_user_dataset[user_id].append([user_item_list_cur, user_cate_list_cur, user_time_list_cur, user_time_list_ts_cur])

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

def build_dataset_split_time_splitGap(folder_src, file_pkl_remap, folder_dst):
  dataset_tag = "dataset_splitGap"
  # print("build_dataset_split_time_splitGap...")
  dict_user_dataset = {}
  dict_user_test_negN = {}

  # read file remap pkl
  (user_df, cate_list, user_count, item_count, cate_count, example_count, item_map, cate_map, user_map, dict_item_cate) = \
    read_file_remap_pkl(folder_src, file_pkl_remap)


  dict_user_dataset = {}
  file_pkl_dataset = os.path.join(folder_dst, Config.dict_pkl_dataset[dataset_tag])
  valid_user_count = 0
  for user_id, hist in user_df.groupby('user'):
    #print "\nuser_id:", user_id
    user_item_list = hist['item'].tolist()
    user_time_list_ts = hist['timestamp'].tolist()
    user_time_list = [i // 3600 // 24 for i in user_time_list_ts]
    seq_len = len(user_item_list)
    if (not(seq_len >= Config.sequence_mini_min_len)):
      continue

    # print "user_item_list:", user_item_list
    # print "user_time_list_ts:", user_time_list_ts

    user_cate_list = [dict_item_cate[i] for i in user_item_list]
    # set cateList to itemList
    #user_item_list = user_cate_list

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
    save_sequences_user_split(user_item_list, user_cate_list, user_time_list_ts, user_time_list, dict_user_dataset, dict_user_test_negN, user_id,
                              user_split_index_arr, dict_item_cate, item_count, cate_count)

  # write pkl dataset
  write_pkl_dataset(file_pkl_dataset, dict_user_dataset, cate_list, user_count, item_count, cate_count, dict_item_cate, dict_user_test_negN)

  print("build done!")
  print("valid_user_count:", valid_user_count)
  return valid_user_count


def read_file_remap_pkl(folder_src, file_pkl_remap):
  # read file remap.pkl
  file_pkl_remap_path = os.path.join(folder_src, file_pkl_remap)
  print("read file %s" % file_pkl_remap_path)
  with open(file_pkl_remap_path, 'rb') as f:
    user_df = pickle.load(f)
    cate_list = pickle.load(f)
    user_count, item_count, cate_count, example_count = pickle.load(f)
    item_map, cate_map, user_map = pickle.load(f)
    dict_item_cate = pickle.load(f)
  return (user_df, cate_list, user_count, item_count, cate_count, example_count, item_map, cate_map, user_map, dict_item_cate)

def run(folder_src, file_pkl_remap, folder_dst, test_neg_N_item, test_neg_N_cate):
  build_dataset(folder_src, file_pkl_remap, folder_dst, test_neg_N_item, test_neg_N_cate)

  build_dataset_split_time_splitGap(folder_src, file_pkl_remap, folder_dst)


  print "build_dataset done!"

  print "3_build_dataset end..."



if __name__ == "__main__":
  print "\n3_build_dataset start..."
  # parse parameters
  arr = sys.argv
  assert (len(arr) == 1 + 5)
  folder_src, file_pkl_remap, folder_dst, test_neg_N_item, test_neg_N_cate = arr[1], arr[2], arr[3], int(arr[4]), int(arr[5])

  run(folder_src, file_pkl_remap, folder_dst, test_neg_N_item, test_neg_N_cate)


