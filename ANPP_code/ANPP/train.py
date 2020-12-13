import os
import numpy as np
import time
import argparse
import tensorflow as tf
from sampler import WarpSampler
from model import Model
from tqdm import tqdm
from util import *
import sys
sys.path.append("..")
from Config import *
from utils.Dataset import *
from build_dataset import *



def process_args():
    # Network parameters

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--folder_dataset', required=True)
    parser.add_argument('--folder_dataset_model', required=True)
    parser.add_argument('--test_neg_N_item', required=True)
    parser.add_argument('--train_type', required=True)
    parser.add_argument('--weight_loss', default='0.5,0.5')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--max_seq_len', default=50, type=int)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)

    parser.add_argument('--display_freq', default=100, type=int)
    parser.add_argument('--eval_freq', default=1000, type=int)
    parser.add_argument('--eval_rank_freq', default=3000, type=int)
    parser.add_argument('--eval_rank_epoch_freq', default=5, type=int)

    # parse args
    args = parser.parse_args()
    args.max_seq_len = Config.max_seq_len
    args.num_epochs = Config.train_num_epochs
    args.test_neg_N_item = Config.test_neg_N_item
    args.test_neg_N_cate = Config.test_neg_N_cate
    weight_loss = Config.weight_loss.split(',')
    args.weight_loss = np.array(weight_loss, float)
    args.l2_emb = args.weight_loss_regularization=float(Config.weight_loss_regularization)



    # write args
    file_arg = "args.txt_%s" % args.train_type
    with open(os.path.join(args.folder_dataset_model, file_arg), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    f.close()

    return args

def load_dataset(model_name):
    # loading data
    # load file: dataset_model.pickle
    dataset_file_name = os.path.join(args.folder_dataset, Config.dict_pkl_dataset[args.model])
    print "load file :", dataset_file_name
    (train_set, valid_set, test_set, user_count, item_count, cate_count, dict_item_cate) = \
        Dataset.load_dataset(model_name, dataset_file_name)
    return (train_set, valid_set, test_set, user_count, item_count, cate_count, dict_item_cate)

def train(args):
    (train_set, valid_set, test_set, user_count, item_count, cate_count, dict_item_cate) = load_dataset(args.model)

    file_log = "log.txt_%s" %  args.train_type
    f = open(os.path.join(args.folder_dataset_model, file_log), 'w')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    model = Model(user_count, [item_count, cate_count], args)

    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())

    T = 0.0
    t0 = time.time()

    best_epoch=0.
    #[item, cate]

    best_valid_NDCG=[0., 0.]
    best_valid_Hit=[0., 0.]
    best_test_NDCG=[0., 0.]
    best_test_Hit=[0., 0.]


    with sess.as_default():
        for epoch in range(1, args.num_epochs + 1):
            # train data
            dataset = Dataset.load_dataset_train_batch(train_set, args.max_seq_len, args.batch_size, item_count,
                                                       cate_count)
            iterator = dataset.make_initializable_iterator()
            next_element = iterator.get_next()

            # one epoch
            epoch_loss = []
            avg_loss = 0.
            sess.run(iterator.initializer)
            while (True):
                try:
                    (user_list, seqItem_list, posItem_list, negItem_list, seqCate_list, posCate_list, negCate_list, time_ts_list,
         time_ts_prev_delta_list, time_ts_next_delta_list, seq_len_list) \
                        = sess.run(next_element)

                    #seqItem_list: [B, T]
                    #seq_list: [B, type, T]
                    # input seq
                    seq_type = ["item", "category", "time_ts", "time_ts_prev_delta", "time_ts_next_delta"]
                    seq_type_len = len(seq_type)

                    #[type, B, T]
                    seq_list = np.stack((seqItem_list, seqCate_list, time_ts_list, time_ts_prev_delta_list, time_ts_next_delta_list),axis=0)
                    pos_list = np.stack((posItem_list, posCate_list), axis=0)
                    neg_list = np.stack((negItem_list, negCate_list), axis=0)

                    #train
                    auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                                {model.u: user_list, model.input_seq: seq_list,
                                                 model.pos: pos_list, model.neg: neg_list,
                                                 model.is_training: True})

                    epoch_loss.append(loss)

                    avg_loss += loss


                except tf.errors.OutOfRangeError:
                    break

            epoch_loss_avg = np.average(epoch_loss)

            if epoch % Config.eval_every_num_epochs == 0:
                t1 = time.time() - t0
                T += t1
                print '\tEvaluating',
                valid_NDCG, valid_Hit = evaluate_model_dataset(model, valid_set, args, sess, item_count, cate_count, Config.flag_test_mini, Config.flag_test_mini_cnt, Config.test_neg_N_item, Config.test_neg_N_cate, flagIsValid=True)
                test_NDCG, test_Hit = evaluate_model_dataset(model, test_set, args, sess, item_count, cate_count, Config.flag_test_mini, Config.flag_test_mini_cnt, Config.test_neg_N_item, Config.test_neg_N_cate, flagIsValid=False)
                str = "\tEpoch:%d, Epoch loss avg: %f, time: %f(s), valid Item (NDCG@10: %.4f, HR@10: %.4f), test Item (NDCG@10: %.4f, HR@10: %.4f), valid Cate (NDCG@10: %.4f, HR@10: %.4f), test Cate (NDCG@10: %.4f, HR@10: %.4f)\n" %\
                      (epoch, epoch_loss_avg, T, valid_NDCG[0], valid_Hit[0], test_NDCG[0], test_Hit[0], valid_NDCG[1], valid_Hit[1], test_NDCG[1], test_Hit[1])

                print(str)

                f.write(str)
                f.flush()
                t0 = time.time()

                best_epoch, best_valid_NDCG, best_valid_Hit, best_test_NDCG, best_test_Hit = Metrics.save_best_result_multiple(epoch, valid_NDCG, valid_Hit, test_NDCG, test_Hit, best_epoch, best_valid_NDCG, best_valid_Hit, best_test_NDCG, best_test_Hit)


    str="Best epoch:%d, Item:[ best_test_ndcg_K:%f, best_test_hit_K:%f ], Cate[: best_test_ndcg_K:%f, Cate best_test_hit_K:%f ], K:%d\n" % \
        (best_epoch, best_test_NDCG[0], best_test_Hit[0], best_test_NDCG[1], best_test_Hit[1], Config.metrics_K)
    print(str)
    f.write(str)
    f.close()

    print("Done")

if __name__ == "__main__":
    args = process_args()
    train(args)

