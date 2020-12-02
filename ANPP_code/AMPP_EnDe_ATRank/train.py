import json
import os
import random
import sys
import time
from collections import OrderedDict

import numpy as np
import pickle

import tensorflow as tf
sys.path.append("..")
from Config import *
from utils.Dataset import *
from utils.Metrics import *

from input import DataInput, DataInputTest
from model import Model


random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

arr = sys.argv
assert(len(arr) == 1+6)
p_dataset, p_model, folder_dataset, folder_dataset_model, test_neg_N_item, train_type = arr[1], arr[2],arr[3],arr[4], int(arr[5]), arr[6]

print "p_dataset:", p_dataset
print "p_model:", p_model
print "train_type:", train_type


# Network parameters
tf.app.flags.DEFINE_integer('hidden_units', 128, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('num_blocks', 1, 'Number of blocks in each attention')
tf.app.flags.DEFINE_integer('num_heads', 8, 'Number of heads in each attention')
tf.app.flags.DEFINE_float('dropout', 0.0, 'Dropout probability(0.0: no dropout)')
tf.app.flags.DEFINE_float('regulation_rate', 0.00005, 'L2 regulation rate')

tf.app.flags.DEFINE_integer('itemid_embedding_size', 64, 'Item id embedding size')
tf.app.flags.DEFINE_integer('cateid_embedding_size', 64, 'Cate id embedding size')

tf.app.flags.DEFINE_boolean('concat_time_emb', True, 'Concat time-embedding instead of Add')

# Training parameters
tf.app.flags.DEFINE_boolean('from_scratch', True, 'Romove model_dir, and train from scratch, default: False')
tf.app.flags.DEFINE_string('model_dir', 'save_path', 'Path to save model checkpoints')
tf.app.flags.DEFINE_string('optimizer', 'sgd', 'Optimizer for training: (adadelta, adam, rmsprop,sgd*)')
tf.app.flags.DEFINE_float('learning_rate', 1, 'Learning rate')
tf.app.flags.DEFINE_float('max_gradient_norm', 5.0, 'Clip gradients to this norm')

tf.app.flags.DEFINE_integer('train_batch_size', 32, 'Training Batch size')
tf.app.flags.DEFINE_integer('test_batch_size', 128, 'Testing Batch size')
tf.app.flags.DEFINE_integer('max_epochs', 10, 'Maximum # of training epochs')

tf.app.flags.DEFINE_integer('display_freq', 100, 'Display training status every this iteration')
tf.app.flags.DEFINE_integer('eval_freq', 1000, 'Display training status every this iteration')
tf.app.flags.DEFINE_integer('eval_rank_freq', 3000, 'Display training status every this iteration')
tf.app.flags.DEFINE_integer('eval_rank_epoch_freq', 5, 'Display training status every this epoch')

# Runtime parameters
tf.app.flags.DEFINE_string('cuda_visible_devices', '0,1,2,3', 'Choice which GPU to use')
tf.app.flags.DEFINE_float('per_process_gpu_memory_fraction', 0.0, 'Gpu memory use fraction, 0.0 for allow_growth=True')


tf.app.flags.DEFINE_string('train_type', '', '')

FLAGS = tf.app.flags.FLAGS

FLAGS.model_dir = os.path.join(folder_dataset_model, FLAGS.model_dir)

FLAGS.max_epochs = Config.train_num_epochs

FLAGS.train_type = train_type


def create_model(sess, config, dict_item_cate):

  print(json.dumps(config, indent=4))
  model = Model(config, dict_item_cate)

  print('All global variables:')
  for v in tf.global_variables():
    if v not in tf.trainable_variables():
      print('\t', v)
    else:
      print('\t', v, 'trainable')

  ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print('Reloading model parameters..')
    model.restore(sess, ckpt.model_checkpoint_path)
  else:
    if not os.path.exists(FLAGS.model_dir):
      os.makedirs(FLAGS.model_dir)
    print('Created new model parameters..')
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

  return model

def _eval(sess, test_set, model):

  auc_sum = 0.0
  for _, uij in DataInputTest(test_set, FLAGS.test_batch_size):
    auc_sum += model.eval(sess, uij) * len(uij[0])
  test_auc = auc_sum / len(test_set)

  model.eval_writer.add_summary(
      summary=tf.Summary(
          value=[tf.Summary.Value(tag='Eval AUC', simple_value=test_auc)]),
      global_step=model.global_step.eval())

  return test_auc


def _eval_negN(sess, test_set, model, item_count):

  auc_sum = 0.0
  U = 0
  NDCG=0.
  Hit=0.
  for _, uij in DataInputTest(test_set, FLAGS.test_batch_size):
    N, NDCG_K, Hit_K = model.eval_negN(sess, uij, item_count, test_neg_N_item, Config.test_neg_N_cate)
    U += N
    NDCG += NDCG_K
    Hit += Hit_K

  NDCG_K_avg = NDCG / U
  Hit_K_avg = Hit / U

  model.eval_writer.add_summary(
      summary=tf.Summary(
          value=[tf.Summary.Value(tag='Eval NDCG_K', simple_value= NDCG_K_avg ),
                 tf.Summary.Value(tag='Hit_K', simple_value=Hit_K_avg)]),
      global_step=model.global_step.eval())

  return (NDCG_K_avg, Hit_K_avg)


def train():
  start_time = time.time()

  if FLAGS.from_scratch:
    if tf.gfile.Exists(FLAGS.model_dir):
      tf.gfile.DeleteRecursively(FLAGS.model_dir)
    tf.gfile.MakeDirs(FLAGS.model_dir)


  #load file: dataset_model.pickle
  dataset_file_name = os.path.join(folder_dataset, Config.dict_pkl_dataset[p_model])
  print "load file :", dataset_file_name

  (train_set, valid_set, test_set, user_count, item_count, cate_count, dict_item_cate) =\
    Dataset.load_dataset(p_model, dataset_file_name)

  # Config GPU options
  if FLAGS.per_process_gpu_memory_fraction == 0.0:
    gpu_options = tf.GPUOptions(allow_growth=True)
  elif FLAGS.per_process_gpu_memory_fraction == 1.0:
    gpu_options = tf.GPUOptions()
  else:
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=FLAGS.per_process_gpu_memory_fraction)

  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_visible_devices

  # Build Config
  #config = OrderedDict(sorted(FLAGS.__flags.items()))
  config = OrderedDict(tf.app.flags.FLAGS.flag_values_dict().items())
  # for k, v in config.items():
  #   config[k] = v.value
  config['user_count'] = user_count
  config['item_count'] = item_count
  config['cate_count'] = cate_count

  #print("config:", config)
  # write args
  file_arg = "args.txt_%s" % config["train_type"]
  with open(os.path.join(config["model_dir"], file_arg), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(config).items(), key=lambda x: x[0])]))
  f.close()

  # Initiate TF session
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    # Create a new model or reload existing checkpoint
    model = create_model(sess, config, dict_item_cate)
    print('Init finish.\tCost time: %.2fs' % (time.time()-start_time))

    # Eval init AUC
    #print('Init AUC: %.4f' % _eval(sess, test_set, model))

    (NDCG_K, Hit_K) = _eval_negN(sess, test_set, model, item_count)
    print('Init NDCG_K: %.4f, Hit_K: %.4f' % (NDCG_K, Hit_K))

    # Start training
    lr = FLAGS.learning_rate
    epoch_size = round(len(train_set) / FLAGS.train_batch_size)
    print('Training..\tmax_epochs: %d\tepoch_size: %d' %
          (FLAGS.max_epochs, epoch_size))


    start_time, avg_loss, epoch_loss, best_auc = time.time(), 0.0, 0.0, 0.0
    best_epoch, best_NDCG_K, best_Hit_K = 0, 0., 0.
    best_valid = [0., 0.]
    best_test = [0., 0.]

    T = 0.0
    t0 = time.time()
    for epoch in range(FLAGS.max_epochs):
      epoch_loss = 0.
      random.shuffle(train_set)
      batch_cnt = 0
      for _, uij in DataInput(train_set, FLAGS.train_batch_size):

        add_summary = bool(model.global_step.eval() % FLAGS.display_freq == 0)
        step_loss = model.train(sess, uij, lr, add_summary)
        avg_loss += step_loss
        epoch_loss += step_loss
        batch_cnt += 1
        if model.global_step.eval() % FLAGS.eval_freq == 0:
            print('Epoch %d Global_step %d\tTrain_loss: %.4f\t' %
              (model.global_epoch_step.eval(), model.global_step.eval(),
               avg_loss / FLAGS.eval_freq))
            avg_loss = 0.0

      epoch_loss_avg = float(epoch_loss) / batch_cnt
      print("Epoch %d DONE\tepoch_loss_avg:%.4f\tCost time: %.2f(s)" %
        (model.global_epoch_step.eval(), epoch_loss_avg, time.time() - start_time))


      #if model.global_step.eval() % FLAGS.eval_freq == 0:
        # test_auc = _eval(sess, test_set, model)
        # print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_AUC: %.4f' %
        #     (model.global_epoch_step.eval(), model.global_step.eval(),
        #      avg_loss / FLAGS.eval_freq, test_auc))

      # if model.global_epoch_step.eval() % FLAGS.eval_rank_epoch_freq == 0:
      #   test_auc = _eval(sess, test_set, model)
      #   print('Epoch %d Global_step %d\tEval_AUC: %.4f' %
      #         (model.global_epoch_step.eval(), model.global_step.eval(),
      #          test_auc))
      #
      #   if  test_auc > best_auc:
      #     best_auc = test_auc
      #     if(best_auc > 0.8):
      #           model.save(sess)

          #if model.global_step.eval() % FLAGS.eval_rank_freq == 0:

      if epoch % Config.eval_every_num_epochs == 0:
        t1 = time.time() - t0
        T += t1

        t_valid = _eval_negN(sess, valid_set, model, item_count)
        t_test = _eval_negN(sess, test_set, model, item_count)

        str_info = "\tEpoch:%d, Global_step %d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)\n" % (
          epoch, model.global_step.eval(), T, t_valid[0], t_valid[1], t_test[0], t_test[1])
        print(str_info)

        t0 = time.time()
        best_epoch, best_valid, best_test = Metrics.save_best_result(epoch, t_valid, t_test, best_epoch, best_valid,
                                                                     best_test)

      #
      if model.global_step.eval() == 336000:
        lr = 0.1

      # print('Epoch %d DONE\tCost time: %.2f(s)' %
      #       (model.global_epoch_step.eval(), time.time()-start_time))
      model.global_epoch_step_op.eval()
    model.save(sess)
    #print('best test_auc:', best_auc)
    str_info = "Best epoch:%d, best_test_ndcg_K:%f, best_test_hit_K:%f, K:%d\n" % ( best_epoch, best_test[0], best_test[1], Config.metrics_K)
    print(str_info)
    print('Finished')


def main(_):
  train()

if __name__ == '__main__':
  tf.app.run()
