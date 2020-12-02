import random
import pickle
import numpy as np
import sys
import os

sys.path.append("..")
from Config import *
from utils.Dataset import *


random.seed(1234)

arr = sys.argv
assert (len(arr) == 1 + 2)
folder_dataset, folder_dataset_model = arr[1], arr[2]

def build_SASRec():
  print("build_SASRec...")

  # src: dataset.pickle
  file_pkl_dataset = os.path.join(folder_dataset, Config.dict_pkl_dataset["dataset"])
  (dict_user_dataset, cate_list, user_count, item_count, cate_count, dict_item_cate, dict_user_test_negN) = \
    Dataset.load_dataset_dataset_pkl(file_pkl_dataset)

  # dst: dataset_model.pickle
  file_pkl_dataset_model = os.path.join(folder_dataset, Config.dict_pkl_dataset["SASRec"])

  #item
  user_train= {
    "item": {},
    "cate": {},
    "timestamp": {}
  }

  user_valid={
    "item": {},
    "cate": {},
    "timestamp": {}
  }

  user_test={
    "item": {},
    "cate": {},
    "timestamp": {}
  }

  for user_id in dict_user_dataset.keys():
    [user_item_list, user_cate_list, user_time_list, user_time_list_ts] = dict_user_dataset[user_id]

    #item
    user_train["item"][user_id] = user_item_list[:-2]
    user_valid["item"][user_id] = [user_item_list[-2]]
    user_test["item"][user_id] = [user_item_list[-1]]

    #cate
    user_train["cate"][user_id] = user_cate_list[:-2]
    user_valid["cate"][user_id] = [user_cate_list[-2]]
    user_test["cate"][user_id] = [user_cate_list[-1]]

    #timestamp
    user_train["timestamp"][user_id] = user_time_list_ts[:-2]
    user_valid["timestamp"][user_id] = [user_time_list_ts[-2]]
    user_test["timestamp"][user_id] = [user_time_list_ts[-1]]


  #write pkl dataset
  with open(file_pkl_dataset_model, 'wb') as f:
    #item
    pickle.dump((user_train, user_valid, user_test), f, pickle.HIGHEST_PROTOCOL)
    #other
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(dict_item_cate, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(dict_user_test_negN, f, pickle.HIGHEST_PROTOCOL)

  print("build done!")


def load_dataset_SASRec(file):
  print('load_dataset_SASRec..')
  with open(file, 'rb') as f:
    (user_train, user_valid, user_test) = pickle.load(f)
    cate_list = pickle.load(f)
    user_count, item_count, cate_count = pickle.load(f)
    dict_item_cate = pickle.load(f)
    dict_user_test_negN = pickle.load(f)

    return (user_train, user_valid, user_test, cate_list, user_count, item_count, cate_count, dict_item_cate,
            dict_user_test_negN)

def build_dataset_AMPP_En_Seq():
  print("build_dataset_AMPP_En_Seq...")

  dataset_name = "dataset_seq"

  # src: dataset.pickle
  file_pkl_dataset = os.path.join(folder_dataset, Config.dict_pkl_dataset[dataset_name])
  (eventTrain, eventTest, timeTrain, timeTest) = Dataset.load_dataset_seq_pkl(file_pkl_dataset)

  # dst: dataset_model.pickle
  model_tag = "AMPP_En_Seq"
  file_pkl_dataset_model = os.path.join(folder_dataset_model, Config.dict_pkl_dataset[model_tag])

  # write pkl dataset
  with open(file_pkl_dataset_model, 'wb') as f:
    pickle.dump((eventTrain, eventTest, timeTrain, timeTest), f, pickle.HIGHEST_PROTOCOL)

  print("build done!")


def stat_dataset():
  print("stat_dataset...")
  dataset_name = "dataset_seq"

  # src: dataset.pickle
  file_pkl_dataset = os.path.join(folder_dataset, Config.dict_pkl_dataset[dataset_name])
  (eventTrain, eventTest, timeTrain, timeTest) = Dataset.load_dataset_seq_pkl(file_pkl_dataset)

  #stat
  eventData=eventTrain
  eventData.extend(eventTest)

  timeData = timeTrain
  timeData.extend(timeTest)

  assert (len(eventData) == len(timeData))
  N = len(eventData)


  eventSet = set()
  eventTotalNum = 0
  eventSeqLenList = []
  eventTimeDiffList = []
  for lineNo in range(N):
    eventSeq = eventData[lineNo]
    timeSeq = timeData[lineNo]
    eventTotalNum += len(eventSeq)
    eventSeqLenList.append(len(eventSeq))
    if(len(eventSeq) == 0):
      continue
    ts_prev = 0
    for pos in range(len(eventSeq)):
      event = eventSeq[pos]
      ts = timeSeq[pos]
      eventSet.add(event)
      if(pos > 0):
        ts_delta = ts-ts_prev
        eventTimeDiffList.append(ts_delta)
      ts_prev=ts



  print("Marker number:", len(eventSet))
  print("Total Sequence Number:", N)
  print("Total Event Number", eventTotalNum)
  eventSeqLenList = np.array(eventSeqLenList)
  eventTimeDiffList = np.array(eventTimeDiffList)
  print("Sequence Length Min:%f, Max:%f, Average:%f", np.min(eventSeqLenList), np.max(eventSeqLenList), np.average(eventSeqLenList))
  print("Time Interval Min:%f, Max:%f, Average:%f", np.min(eventTimeDiffList), np.max(eventTimeDiffList),
        np.average(eventTimeDiffList))


def main():
  build_dataset_AMPP_En_Seq()
  stat_dataset()


if __name__ == "__main__":
  main()

