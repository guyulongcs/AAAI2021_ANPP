import numpy as np
import sys

from Tool import *
sys.path.append("..")
from Config import *



class Metrics():
    @classmethod
    #input a: np.array(Batch, 1 pos + N neg) score
    def NDCG_HIT(cls, a, K):
        Tool.output("NDCG_HIT", None, Config.flag_debug)
        Tool.output("a.shape", a.shape, Config.flag_debug)
        Tool.output("a", a, Config.flag_debug)
        rank = (-a).argsort(axis=-1).argsort(axis=-1)[:, 0]
        N = a.shape[0]
        rank_K = rank[rank < K]

        Tool.output("rank_K", rank_K, Config.flag_debug)

        NDCG_K = np.sum(1 / np.log2(rank_K + 2))
        Hit_K = len(rank_K)
        return N, NDCG_K, Hit_K


    @classmethod
    def NDCG_Hit_Prob(cls, prob_preds, label_true, K):
        N = len(label_true)
        rank_all = (-prob_preds).argsort(axis=-1).argsort(axis=-1)
        rank = np.zeros(N)
        for i in range(N):
            rank[i]=rank_all[i, label_true[i]]

        rank_K = rank[rank < K]
        print("rank_K:", rank_K)
        NDCG_K = np.sum(1 / np.log2(rank_K + 2))
        Hit_K = len(rank_K)
        return N, NDCG_K, Hit_K






    @classmethod
    def save_best_result(cls, epoch, t_valid, t_test, best_epoch, best_valid, best_test):
        if(t_valid[0] > best_valid[0]):
            best_epoch = epoch
            best_valid = t_valid
            best_test = t_test
        return best_epoch, best_valid, best_test

    @classmethod
    def save_best_result_multiple(cls, epoch, valid_NDCG,valid_Hit,test_NDCG,test_Hit,best_epoch,best_valid_NDCG,best_valid_Hit,best_test_NDCG,best_test_Hit):
        if(valid_NDCG[0] > best_valid_NDCG[0]):
            best_epoch = epoch
            best_valid_NDCG = valid_NDCG
            best_valid_Hit = valid_Hit
            best_test_NDCG = test_NDCG
            best_test_Hit = test_Hit
        return best_epoch, best_valid_NDCG, best_valid_Hit, best_test_NDCG, best_test_Hit



    @classmethod
    def MAE(cls, time_preds, time_true):
        return np.mean(np.abs(time_preds - time_true))

    @classmethod
    def ACC(cls, prob_preds, label_true):
        N = len(label_true)
        N_equal = np.sum(prob_preds.argmax(axis=-1) == label_true)
        return N_equal / N

    @classmethod
    #label_preds: [B, V], label_true: [B]
    def Rank(cls, prob_preds, label_true, K):
        acc = Metrics.ACC(prob_preds, label_true)
        N, NDCG_K_N, Hit_K_N = Metrics.NDCG_Hit_Prob(prob_preds, label_true, K)
        return (N, acc, NDCG_K_N/float(N), Hit_K_N/float(N))

    @classmethod
    def NDCG_Hit_Prob_checkValid(cls, prob_preds, label_true, K):
        #print("NDCG_Hit_Prob...")
        B, T, N = prob_preds.shape[0], prob_preds.shape[1], prob_preds.shape[2]
        B1, T1 = label_true.shape[0], label_true.shape[1]

        rank_all = (-prob_preds).argsort(axis=-1).argsort(axis=-1)
        #print("rank_all:", rank_all.shape)
        #print("B:{}, T:{}, B1:{}, T1:{}".format(B, T, B1, T1))
        rank = []
        is_valid = label_true > 0
        for i in range(B):
            for j in range(T):
                if (is_valid[i][j] and (label_true[i, j] - 1 < N)):
                    # print("label_true[i, j]-1]:", label_true[i, j]-1)
                    rank.append(rank_all[i, j, label_true[i, j] - 1])
        rank = np.array(rank)
        rank_K = rank[rank < K]
        #print("rank_K:", rank_K)
        NDCG_K = np.sum(1 / np.log2(rank_K + 2))
        Hit_K = len(rank_K)

        N = len(rank)
        NDCG_K = NDCG_K / float(N)
        Hit_K = Hit_K / float(N)

        return N, NDCG_K, Hit_K