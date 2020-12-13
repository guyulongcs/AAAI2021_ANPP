import pickle
import sys
import os
import numpy as np
import tensorflow as tf
from Time import *

from Tool import *
sys.path.append("..")
from Config import *


# [1, 2) = 0, [2, 4) = 1, [4, 8) = 2, [8, 16) = 3...  need len(gap) hot
gap = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
# gap = [2, 7, 15, 30, 60,]
# gap.extend( range(90, 4000, 200) )
# gap = np.array(gap)
#print("gap.shape:",gap.shape[0])

class Dataset():
    # Loading data
    @classmethod
    def load_dataset_dataset_pkl(cls, file):
        with open(file, 'rb') as f:
            dict_user_dataset = pickle.load(f)
            cate_list = pickle.load(f)
            user_count, item_count, cate_count = pickle.load(f)
            dict_item_cate = pickle.load(f)
            dict_user_test_negN = pickle.load(f)
            return (dict_user_dataset, cate_list, user_count, item_count, cate_count, dict_item_cate, dict_user_test_negN)

    @classmethod
    def load_dataset_seq_pkl(cls, file):
        with open(file, 'rb') as f:
            (eventTrain, eventTest, timeTrain, timeTest) = pickle.load(f)

            return (eventTrain, eventTest, timeTrain, timeTest)

    @classmethod
    def proc_time_emb(cls, hist_t, cur_t):
        hist_t = [cur_t - i + 1 for i in hist_t]
        hist_t = [np.sum(i >= gap) for i in hist_t]
        return hist_t

    @classmethod
    def proc_time_get_diff(cls, hist_t, cur_t):
        return cur_t - hist_t + 1

    @classmethod
    def pro_seq_time_to_timeDelta(cls, timeSeq):
        seq_delta = [0]
        for index in range(1, len(timeSeq)):
            seq_delta.append(Time.get_time_diff_days(timeSeq[index-1], timeSeq[index]))
        return seq_delta

    @classmethod
    def pro_list_seq_time_to_timeDelta(cls, timeSeqList):
        return [Dataset.pro_seq_time_to_timeDelta(timeSeq) for timeSeq in timeSeqList]

    @classmethod
    def pro_list_seq_time_to_dayStartZero(cls, timeSeqList):
        return [Dataset.pro_seq_time_to_dayStartZero(timeSeq) for timeSeq in timeSeqList]

    @classmethod
    def pro_seq_time_to_dayStartZero(cls, timeSeq):
        seq_delta = [0]
        for index in range(1, len(timeSeq)):
            seq_delta.append(Time.get_time_diff_days(timeSeq[0], timeSeq[index]))
        return seq_delta

    @classmethod
    def pro_list_seqEvent_to_seqCate(cls, eventSeqList, dictEventCate):
        return [Dataset.pro_seqEvent_to_seqCate(eventSeq, dictEventCate) for eventSeq in eventSeqList]

    @classmethod
    def pro_seqEvent_to_seqCate(cls, eventSeq, dictEventCate):
        return [dictEventCate[event] for event in eventSeq]

    @classmethod
    def load_dataset(cls, model, file):
        print('Loading data...')
        res = None
        if(model == "atrank"):
            res = Dataset.load_dataset_atrank(file)
        elif(model == "SASRec"):
            res = Dataset.load_dataset_SASRec(file)
        elif(model == "rmtpp"):
            res = Dataset.load_dataset_rmtpp(file)
        elif (model == "intensityRNN"):
            res = Dataset.load_dataset_intensityRNN(file)
        elif (model == "ANPP"):
            res = Dataset.load_dataset_ANPP(file)
        elif(model.startswith("AMPP_")):
            res = Dataset.load_dataset_train(file)
        print('Loading data done!')
        return res
        # with open(file, 'rb') as f:
        #     train_set = pickle.load(f)
        #     valid_set = pickle.load(f)
        #     test_set = pickle.load(f)
        #     cate_list = pickle.load(f)
        #     user_count, item_count, cate_count = pickle.load(f)
        #     train_set_pos_neg = pickle.load(f)
        #     return (train_set, valid_set, test_set, cate_list, user_count, item_count, cate_count, train_set_pos_neg)

    @classmethod
    def load_dataset_Encoder(cls, file):
        print("load_dataset_Encoder %s..." % file)

    @classmethod
    def load_dataset_pickle_user_testNegN(cls, file):
        print('load_dataset_pickle_user_testNegN...')
        dict_uer_testNegN = None
        with open(file, 'rb') as f:
            dict_uer_testNegN = pickle.load(f)
        return dict_uer_testNegN

    @classmethod
    def load_dataset_atrank(cls, file):
        print('load_dataset_atrank..')
        with open(file, 'rb') as f:
            train_set = pickle.load(f)
            test_set = pickle.load(f)
            cate_list = pickle.load(f)
            user_count, item_count, cate_count = pickle.load(f)
            dict_item_cate = pickle.load(f)
            dict_user_test_negN = pickle.load(f)
            return (train_set, test_set, cate_list, user_count, item_count, cate_count, dict_item_cate, dict_user_test_negN)


    @classmethod
    def load_dataset_rmtpp(cls, file):
        print('load_dataset_rmtpp..')
        with open(file, 'rb') as f:
            (event_train, event_test, time_train, time_test) = pickle.load(f)

            return (event_train, event_test, time_train, time_test)

    @classmethod
    def load_dataset_intensityRNN(cls, file):
        print('load_dataset_intensityRNN..')
        with open(file, 'rb') as f:
            (event_train, event_test, time_train, time_test) = pickle.load(f)

            return (event_train, event_test, time_train, time_test)

    @classmethod
    def load_dataset_ANPP(cls, file):
        print('load_dataset_ANPP..')
        with open(file, 'rb') as f:
            (event_train, event_test, time_train, time_test) = pickle.load(f)

            return (event_train, event_test, time_train, time_test)

    @classmethod
    def load_dataset_train(cls, file):
        print('load_dataset_train..')
        with open(file, 'rb') as f:
            (train_set, valid_set, test_set) = pickle.load(f)
            (user_count, item_count, cate_count) = pickle.load(f)
            dict_item_cate = pickle.load(f)
            return (train_set, valid_set, test_set, user_count, item_count, cate_count, dict_item_cate)



    @classmethod
    def random_neq(cls, l, r, list_item):
        t = np.random.randint(l, r)
        while t in list_item:
            t = np.random.randint(l, r)
        return t

    @classmethod
    def random_neq_N(cls, l, r, list_item, N):
        neg_list = []
        for i in range(N):
            neg = np.random.randint(l, r)
            while ((neg in list_item) or (neg in neg_list)):
                neg = np.random.randint(l, r)
            neg_list.append(neg)
        return neg_list

    @classmethod
    def random_neq_N_seq(cls, l, r, list_item, N):
        neg_list = []
        for i in range(N):
            neg = np.random.randint(l, r)
            while ((neg in list_item)):
                neg = np.random.randint(l, r)
            neg_list.append(neg)
        return neg_list

    @classmethod
    def random_neq_N_seq_each(cls, l, r, list_item, N):
        neg_list = []
        for i in range(N):
            neg = np.random.randint(l, r)
            while (neg == list_item[i]):
                neg = np.random.randint(l, r)
            neg_list.append(neg)
        return neg_list


    @classmethod
    def load_dataset_train_batch_parse(cls, dataset, seq_max_len, item_count, cate_count):
        #print("load_dataset_train_batch_parse start...")
        #return res list
        user_list, seqItem_list, posItem_list, negItem_list, seqCate_list, posCate_list, negCate_list, time_ts_list, time_ts_prev_delta_list, time_ts_next_delta_list, seq_len_list = [], [], [], [], [], [], [], [], [], [], []

        #read data by line
        N = len(dataset)
        for index in range(N):
            #each line
            user_id, list_item, list_cate, list_time_ts = dataset[index]

            Tool.output("\nuse_id:", user_id, Config.flag_debug)
            Tool.output("list_item:", list_item, Config.flag_debug)
            Tool.output("list_time_ts:", list_time_ts, Config.flag_debug)
            #line req
            seq = {
                "item": np.zeros([seq_max_len], dtype=np.int32),
                "category": np.zeros([seq_max_len], dtype=np.int32)
            }

            pos = {
                "item": np.zeros([seq_max_len], dtype=np.int32),
                "category": np.zeros([seq_max_len], dtype=np.int32)
            }

            neg = {
                "item": np.zeros([seq_max_len], dtype=np.int32),
                "category": np.zeros([seq_max_len], dtype=np.int32)
            }

            time_ts_line = np.zeros([seq_max_len], dtype=np.int32)
            time_ts_prev_delta_line = np.zeros([seq_max_len], dtype=np.int32)
            time_ts_next_delta_line = np.zeros([seq_max_len], dtype=np.int32)

            seq_index = seq_max_len-1
            #list max len: T+1
            list_len = len(list_item)
            for list_index in range(list_len-1, 0, -1):
                if(seq_index < 0):
                    break
                #item
                seq["item"][seq_index] = list_item[list_index - 1]
                pos["item"][seq_index] = list_item[list_index]
                if(pos["item"][seq_index] != 0):
                    neg["item"][seq_index] = Dataset.random_neq(1, item_count + 1, list_item)
                #cate
                seq["category"][seq_index] = list_cate[list_index - 1]
                pos["category"][seq_index] = list_cate[list_index]
                if(pos["category"][seq_index] != 0):
                    neg["category"][seq_index] = Dataset.random_neq(1, cate_count + 1, list_cate)

                #time: cur timestamp
                time_ts_line[seq_index] = list_time_ts[list_index - 1]

                #time: prev time delta
                time_ts_prev_delta_line[seq_index] = 0
                if(list_index - 2 >= 0):
                    time_ts_prev_delta_line[seq_index] = Time.get_time_diff_bucket(list_time_ts[list_index - 2], list_time_ts[list_index - 1])

                #time: next time delta
                time_ts_next_delta_line[seq_index] = Time.get_time_diff_bucket(list_time_ts[list_index-1], list_time_ts[list_index])

                #next
                seq_index -= 1

            #user
            user_list.append(user_id)
            #item
            seqItem_list.append(seq["item"])
            posItem_list.append(pos["item"])
            negItem_list.append(neg["item"])

            Tool.output("seq_[item]:", seq["item"], Config.flag_debug)
            Tool.output("pos_[item]:", pos["item"], Config.flag_debug)
            Tool.output("neg_[item]:", neg["item"], Config.flag_debug)

            #cate
            seqCate_list.append(seq["category"])
            posCate_list.append(pos["category"])
            negCate_list.append(neg["category"])

            #time
            time_ts_list.append(time_ts_line)
            time_ts_prev_delta_list.append(time_ts_prev_delta_line)
            time_ts_next_delta_list.append(time_ts_next_delta_line)

            #seq len
            seq_len_list.append(list_len)

        #print("load_dataset_train_batch_parse end!")
        return (np.array(user_list), np.array(seqItem_list), np.array(posItem_list), np.array(negItem_list), np.array(seqCate_list), np.array(posCate_list), np.array(negCate_list), np.array(time_ts_list), np.array(time_ts_prev_delta_list), np.array(time_ts_next_delta_list), np.array(seq_len_list))



    @classmethod
    def load_dataset_test_batch_parse(cls, dataset, seq_max_len, item_count, cate_count, test_neg_N_item, test_neg_N_cate, flagIsValid):
        #print("load_dataset_test_batch_parse start...")
        Tool.output("flagIsValid:", flagIsValid, Config.flag_debug)
        #return res list
        user_list, seqItem_list, posItem_list, negItem_list, seqCate_list, posCate_list, negCate_list, time_ts_list, time_ts_prev_delta_list, time_ts_next_delta_list, seq_len_list = [], [], [], [], [], [], [], [], [], [], []

        #read data by line
        N = len(dataset)
        for index in range(N):
            #each line
            [user_id, (hist_item, cur_item, neg_list_N_item_test), (hist_cate, cur_cate, neg_list_N_cate_test), (hist_time_ts, cur_time_ts)] = dataset[index]

            Tool.output("\nuser_id:", user_id, Config.flag_debug)

            # line req
            seq = {
                "item": np.zeros([seq_max_len], dtype=np.int32),
                "category": np.zeros([seq_max_len], dtype=np.int32)
            }

            pos = {
                "item": np.zeros([seq_max_len], dtype=np.int32),
                "category": np.zeros([seq_max_len], dtype=np.int32)
            }

            neg = {
                "item": np.zeros([seq_max_len], dtype=np.int32),
                "category": np.zeros([seq_max_len], dtype=np.int32)
            }

            time_ts_line = np.zeros([seq_max_len], dtype=np.int32)
            time_ts_prev_delta_line = np.zeros([seq_max_len], dtype=np.int32)
            time_ts_next_delta_line = np.zeros([seq_max_len], dtype=np.int32)



            # hist max len: T
            hist_len = len(hist_item)
            seq["item"][-hist_len:] = np.array(hist_item)
            seq["category"][-hist_len:] = np.array(hist_cate)
            time_ts_line[-hist_len:] = np.array(hist_time_ts)

            seq_index = seq_max_len - 1
            for hist_index in range(hist_len - 1, -1, -1):
                if(seq_index < 0):
                    break
                # time: prev time delta
                time_ts_prev_delta_line[seq_index] = 0
                if (hist_index - 1 >= 0):
                    time_ts_prev_delta_line[seq_index] = Time.get_time_diff_bucket(hist_time_ts[hist_index - 1],
                                                                                   hist_time_ts[hist_index])
                # time: next time delta
                next_ts = cur_time_ts
                if(hist_index + 1 <= hist_len-1):
                    next_ts = hist_time_ts[hist_index+1]
                time_ts_next_delta_line[seq_index] = Time.get_time_diff_bucket(hist_time_ts[hist_index],
                                                                               next_ts)
                # next
                seq_index -= 1

            # user
            user_list.append(user_id)
            # item
            seqItem_list.append(seq["item"])
            posItem_list.append([cur_item])



            #valid
            if(flagIsValid):
                cur_negItemList = Dataset.random_neq_N(1, item_count + 1, hist_item, test_neg_N_item)
            #test
            else:
                cur_negItemList = neg_list_N_item_test
            negItem_list.append(np.array(cur_negItemList))

            Tool.output("hist_item:", hist_item, Config.flag_debug)
            Tool.output("seq_item:",  seq["item"], Config.flag_debug)
            Tool.output("cur_item:", cur_item, Config.flag_debug)
            Tool.output("neg_list_N_item_test:", neg_list_N_item_test, Config.flag_debug)
            Tool.output("cur_negItemList:", cur_negItemList, Config.flag_debug)

            # cate
            seqCate_list.append(seq["category"])
            posCate_list.append([cur_cate])
            negCate_list.append(np.array(neg_list_N_cate_test))

            # time
            time_ts_list.append(time_ts_line)
            time_ts_prev_delta_list.append(time_ts_prev_delta_line)
            time_ts_next_delta_list.append(time_ts_next_delta_line)

            # seq len
            seq_len_list.append(hist_len)

        #print("load_dataset_test_batch_parse end!")
        return (np.array(user_list), np.array(seqItem_list), np.array(posItem_list), np.array(negItem_list),
                    np.array(seqCate_list), np.array(posCate_list), np.array(negCate_list), np.array(time_ts_list),
                    np.array(time_ts_prev_delta_list), np.array(time_ts_next_delta_list), np.array(seq_len_list))

    @classmethod
    def load_dataset_train_batch(cls, dataset, seq_max_len, batch_size, item_count, cate_count):
        (user_list, seqItem_list, posItem_list, negItem_list, seqCate_list, posCate_list, negCate_list, time_ts_list,
         time_ts_prev_delta_list, time_ts_next_delta_list, seq_len_list) = Dataset.load_dataset_train_batch_parse(dataset, seq_max_len, item_count, cate_count)
        dataset = tf.data.Dataset.from_tensor_slices((user_list, seqItem_list, posItem_list, negItem_list, seqCate_list, posCate_list, negCate_list, time_ts_list, time_ts_prev_delta_list, time_ts_next_delta_list, seq_len_list)).shuffle(buffer_size=100000).batch(batch_size)
        return dataset

    @classmethod
    def load_dataset_test_batch(cls, dataset, seq_max_len, batch_size, item_count, cate_count, test_neg_N_item, test_neg_N_cate, flagIsValid):
        (user_list, seqItem_list, posItem_list, negItem_list, seqCate_list, posCate_list, negCate_list, time_ts_list,
         time_ts_prev_delta_list, time_ts_next_delta_list, seq_len_list) = Dataset.load_dataset_test_batch_parse(dataset, seq_max_len, item_count, cate_count, test_neg_N_item, test_neg_N_cate, flagIsValid)
        dataset = tf.data.Dataset.from_tensor_slices((user_list, seqItem_list, posItem_list, negItem_list, seqCate_list, posCate_list, negCate_list, time_ts_list,
         time_ts_prev_delta_list, time_ts_next_delta_list, seq_len_list)).batch(batch_size)
        return dataset

    @classmethod
    def trans_itemList_to_cateList(cls, itemList, dictItemCate):
        return [dictItemCate[item] for item in itemList]

    # @classmethod
    # def trans_dataset_to_SASRec(cls):
    #
