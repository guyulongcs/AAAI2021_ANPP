import os
import time
import json
import pickle
import random
from collections import OrderedDict

import numpy as np
import tensorflow as tf

from input import DataInput, DataInputTest
from model import Model

random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

# pylint: disable=line-too-long
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
tf.app.flags.DEFINE_float('learning_rate', 1.0, 'Learning rate')
tf.app.flags.DEFINE_float('max_gradient_norm', 5.0, 'Clip gradients to this norm')

tf.app.flags.DEFINE_integer('train_batch_size', 32, 'Training Batch size')
tf.app.flags.DEFINE_integer('test_batch_size', 512, 'Testing Batch size')
tf.app.flags.DEFINE_integer('max_epochs', 50, 'Maximum # of training epochs')

tf.app.flags.DEFINE_integer('display_freq', 100, 'Display training status every this iteration')
tf.app.flags.DEFINE_integer('eval_freq', 1000, 'Display training status every this iteration')

# Runtime parameters
tf.app.flags.DEFINE_string('cuda_visible_devices', '1', 'Choice which GPU to use')
tf.app.flags.DEFINE_float('per_process_gpu_memory_fraction', 0.0, 'Gpu memory use fraction, 0.0 for allow_growth=True')
# pylint: enable=line-too-long

FLAGS = tf.app.flags.FLAGS

def create_model(sess, config, cate_list):
  print(json.dumps(config, indent=4))
  model = Model(config, cate_list)

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


def train():
  start_time = time.time()

  if FLAGS.from_scratch:
    if tf.gfile.Exists(FLAGS.model_dir):
      tf.gfile.DeleteRecursively(FLAGS.model_dir)
    tf.gfile.MakeDirs(FLAGS.model_dir)

  # Loading data
  print('Loading data..')
  with open('dataset.pkl', 'rb') as f:
    train_set = pickle.load(f)
    test_set = pickle.load(f)
    cate_list = pickle.load(f)
    user_count, item_count, cate_count = pickle.load(f)

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
  config = OrderedDict(tf.app.flags.FLAGS.flag_values_dict().items())
  # for k, v in config.items():
  #   config[k] = v.value
  config['user_count'] = user_count
  config['item_count'] = item_count
  config['cate_count'] = cate_count


  # Initiate TF session
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    # Create a new model or reload existing checkpoint
    model = create_model(sess, config, cate_list)
    print('Init finish.\tCost time: %.2fs' % (time.time()-start_time))

    # Eval init AUC
    print('Init AUC: %.4f' % _eval(sess, test_set, model))

    # Start training
    lr = FLAGS.learning_rate
    epoch_size = round(len(train_set) / FLAGS.train_batch_size)
    print('Training..\tmax_epochs: %d\tepoch_size: %d' %
          (FLAGS.max_epochs, epoch_size))

    start_time, avg_loss, best_auc = time.time(), 0.0, 0.0
    for _ in range(FLAGS.max_epochs):

      random.shuffle(train_set)

      for _, uij in DataInput(train_set, FLAGS.train_batch_size):

        add_summary = bool(model.global_step.eval() % FLAGS.display_freq == 0)
        step_loss = model.train(sess, uij, lr, add_summary)
        avg_loss += step_loss

        if model.global_step.eval() % FLAGS.eval_freq == 0:
          test_auc = _eval(sess, test_set, model)
          print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_AUC: %.4f' %
                (model.global_epoch_step.eval(), model.global_step.eval(),
                 avg_loss / FLAGS.eval_freq, test_auc))
          avg_loss = 0.0

          if test_auc > 0.86 and test_auc > best_auc:
            best_auc = test_auc
            model.save(sess)

        if model.global_step.eval() == 336000:
          lr = 0.1

      print('Epoch %d DONE\tCost time: %.2f' %
            (model.global_epoch_step.eval(), time.time()-start_time))
      model.global_epoch_step_op.eval()

    model.save(sess)
    print('best test_auc:', best_auc)
    print('Finished')



def main(_):
  train()

if __name__ == '__main__':
  tf.app.run()
