import sys
import copy
import random
import numpy as np
from collections import defaultdict
import tensorflow as tf

sys.path.append("..")
from Config import *
from utils.Dataset import *
from utils.Metrics import *
from utils.Tool import *

def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


def evaluate(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(xrange(1, usernum + 1), 10000)
    else:
        users = xrange(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.max_seq_len], dtype=np.int32)
        idx = args.max_seq_len - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]

        #print("-predictions:"), predictions
        #print type(predictions)
        #print predictions.shape

        rank = predictions.argsort().argsort()[0]

        #print("rank:", rank)

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print '.',
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(xrange(1, usernum + 1), 10000)
    else:
        users = xrange(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.max_seq_len], dtype=np.int32)
        idx = args.max_seq_len - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print '.',
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_model(model, train, valid, test, args, sess, dict_user_testNegN, flag_test_mini, flag_test_mini_cnt):
    #print("evaluate_model...")
    NDCG = 0.0
    Hit = 0.0
    valid_user = 0.0


    for user_id in test:
        u = user_id

        seq = np.zeros([args.max_seq_len], dtype=np.int32)
        idx = args.max_seq_len - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        item_idx = [test[u][0]]

        user_test_negN = dict_user_testNegN[u][0]
        item_idx.extend(user_test_negN)

        predictions = model.predict(sess, [u], [seq], item_idx)
        N, NDCG_K, Hit_K = Metrics.NDCG_HIT(predictions, Config.metrics_K)

        valid_user += 1
        NDCG += NDCG_K
        Hit += Hit_K

        if(flag_test_mini and valid_user >= flag_test_mini_cnt):
            break

    return NDCG / valid_user, Hit / valid_user



def evaluate_model_valid(model, train, valid, test, args, sess, dict_user_testNegN, itemnum, flag_test_mini, flag_test_mini_cnt):
    #print("evaluate_model_valid...")
    NDCG = 0.0
    Hit = 0.0
    valid_user = 0.0


    for user_id in valid:
        u = user_id

        seq = np.zeros([args.max_seq_len], dtype=np.int32)
        idx = args.max_seq_len - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        item_idx = [valid[u][0]]

        #rand neg
        rated = set(train[u])
        rated.add(0)
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        #test neg
        # user_test_negN = dict_user_testNegN[u]
        # item_idx.extend(user_test_negN)

        predictions = model.predict(sess, [u], [seq], item_idx)
        N, NDCG_K, Hit_K = Metrics.NDCG_HIT(predictions, Config.metrics_K)

        assert(N==1)

        valid_user += N
        NDCG += NDCG_K
        Hit += Hit_K

        if(flag_test_mini and valid_user >= flag_test_mini_cnt):
            break

    return NDCG / valid_user, Hit / valid_user





def evaluate_model_dataset(model, test_set, args, sess, item_count, cate_count, flag_test_mini, flag_test_mini_cnt, test_neg_N_item, test_neg_N_cate, flagIsValid=False):
    Tool.output("evaluate_model_dataset:", None, Config.flag_debug)
    Tool.output("flagIsValid:", flagIsValid, Config.flag_debug)
    if(flagIsValid == False):
        Tool.output("Test....flagIsValid:", flagIsValid, Config.flag_debug)
    #print("evaluate_model...")
    NDCG = np.array([0.0, 0.0])
    Hit = np.array([0.0, 0.0])
    valid_user = 0.0

    # train data
    dataset = Dataset.load_dataset_test_batch(test_set, args.max_seq_len, args.batch_size, item_count, cate_count, test_neg_N_item, test_neg_N_cate, flagIsValid)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    sess.run(iterator.initializer)
    while (True):
        try:
            (
            user_list, seqItem_list, posItem_list, negItem_list, seqCate_list, posCate_list, negCate_list, time_ts_list,
            time_ts_prev_delta_list, time_ts_next_delta_list, seq_len_list) \
                = sess.run(next_element)

            Tool.output("user_list:", user_list, Config.flag_debug)
            Tool.output("seqItem_list:", seqItem_list, Config.flag_debug)
            Tool.output("posItem_list:", posItem_list, Config.flag_debug)
            Tool.output("negItem_list:", negItem_list, Config.flag_debug)

            # seqItem_list: [B, T]
            # seq_list: [B, type, T]
            # input seq
            seq_type = ["item", "category", "time_ts", "time_ts_prev_delta", "time_ts_next_delta"]
            seq_type_len = len(seq_type)

            # [type, B, T]
            seq_list = np.stack(
                (seqItem_list, seqCate_list, time_ts_list, time_ts_prev_delta_list, time_ts_next_delta_list), axis=0)
            pos_list = np.stack((posItem_list, posCate_list), axis=0)
            neg_list = np.stack((negItem_list, negCate_list), axis=0)

            #[B, 1+Neg]
            test_item_list = np.concatenate((posItem_list, negItem_list), axis=-1)
            test_cate_list = np.concatenate((posCate_list, negCate_list), axis=-1)


            predictions = model.predict(sess, user_list, seq_list, test_item_list, test_cate_list)
            predictions_items, predictions_categories = predictions[0], predictions[1]

            N_item, NDCG_K_item, Hit_K_item = Metrics.NDCG_HIT(predictions_items, Config.metrics_K)
            N_cate, NDCG_K_cate, Hit_K_cate = Metrics.NDCG_HIT(predictions_categories, Config.metrics_K)

            # print("N_item:%d, NDCG_K_item:%f, Hit_K_item:%f" % (N_item, NDCG_K_item, Hit_K_item))
            # print("N_cate:%d, NDCG_K_cate:%f, Hit_K_cate:%f" % (N_cate, NDCG_K_cate, Hit_K_cate))
            valid_user += N_item
            NDCG += [NDCG_K_item, NDCG_K_cate]
            Hit += [Hit_K_item, Hit_K_cate]

            if (flag_test_mini and valid_user >= flag_test_mini_cnt):
                break

        except tf.errors.OutOfRangeError:
            break
    return np.array(NDCG) / float(valid_user), np.array(Hit) / float(valid_user)
