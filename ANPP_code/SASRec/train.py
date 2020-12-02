import os
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


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


# Network parameters

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--folder_dataset', required=True)
parser.add_argument('--folder_dataset_model', required=True)
parser.add_argument('--test_neg_N_item', required=True)
parser.add_argument('--train_type', required=True)


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



args = parser.parse_args()
args.max_seq_len = Config.max_seq_len
args.num_epochs = Config.train_num_epochs

file_arg = "args.txt_%s" % args.train_type
with open(os.path.join(args.folder_dataset_model, file_arg), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

# loading data
# load file: dataset_model.pickle
dataset_file_name = os.path.join(args.folder_dataset, Config.dict_pkl_dataset[args.model])
print "load file :", dataset_file_name
(user_train_item, user_valid_item, user_test_item, user_train_cate, user_valid_cate, user_test_cate, cate_list, user_count, item_count, cate_count, dict_item_cate,
            dict_user_test_negN) = \
    load_dataset_SASRec(dataset_file_name)

usernum = user_count



user_train = []
user_valid = []
user_test = []
if(args.train_type == Config.item_type_list[0]):
    user_train = user_train_item
    user_valid = user_valid_item
    user_test = user_test_item
    itemnum = item_count
    dict_user_test_negN_index = 0
if(args.train_type == Config.item_type_list[1]):
    user_train = user_train_cate
    user_valid = user_valid_cate
    user_test = user_test_cate
    itemnum = cate_count
    dict_user_test_negN_index = 1

num_batch = len(user_train) / args.batch_size
print "num_batch:", num_batch
cc = 0.0

# for u in user_train:
#     cc += len(user_train[u])
# print 'average sequence length: %.2f' % (cc / len(user_train))

file_log = "log.txt_%s" %  args.train_type
f = open(os.path.join(args.folder_dataset_model, file_log), 'w')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.max_seq_len, n_workers=3)
model = Model(usernum, itemnum, args)
sess.run(tf.initialize_all_variables())


T = 0.0
t0 = time.time()


best_epoch=0.
best_valid=[0., 0.]
best_test=[0., 0.]

for epoch in range(1, args.num_epochs + 1):

    epoch_loss = []
    for batch in range(num_batch):
        u, seq, pos, neg = sampler.next_batch()
        auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                 model.is_training: True})
        # if(batch % 200 == 0):
        #     print("Epoch:%d/%d, Batch:%d/%d" % (epoch, args.num_epochs, batch+1, num_batch))
        epoch_loss.append(loss)

    epoch_loss_avg = np.average(epoch_loss)
    print("Epoch:%d/%d, Epoch loss avg: %f " % (epoch, args.num_epochs, epoch_loss_avg))

    if epoch % Config.eval_every_num_epochs == 0:

        t1 = time.time() - t0
        T += t1
        print '\tEvaluating',
        t_valid = evaluate_model_valid(model, user_train, user_valid, user_test, args, sess, dict_user_test_negN, itemnum, Config.flag_test_mini, Config.flag_test_mini_cnt)
        t_test = evaluate_model(model, user_train, user_valid, user_test, args, sess, dict_user_test_negN, Config.flag_test_mini, Config.flag_test_mini_cnt, dict_user_test_negN_index)
        str = "\tepoch:%d, Epoch loss avg: %f, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)\n" % (epoch, epoch_loss_avg, T, t_valid[0], t_valid[1], t_test[0], t_test[1])
        print(str)

        f.write(str)
        f.flush()
        t0 = time.time()

        best_epoch, best_valid, best_test = Metrics.save_best_result(epoch, t_valid, t_test, best_epoch, best_valid, best_test)


str="Best epoch:%d, best_test_ndcg_K:%f, best_test_hit_K:%f, K:%d\n" % (best_epoch, best_test[0], best_test[1], Config.metrics_K)
print(str)
f.write(str)
f.close()

print("Done")
