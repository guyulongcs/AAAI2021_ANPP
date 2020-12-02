#!/usr/bin/env python
import tempfile
import argparse
#import click
import tensorflow as tf
import rmtpp_core
import utils_tpp
import rmtpp_core
import os
import sys

sys.path.append("..")
from Config import *
from utils.Dataset import *
from utils.Metrics import *

def_opts = rmtpp_core.def_opts


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--folder_dataset', required=True)
parser.add_argument('--folder_dataset_model', required=True)
#parser.add_argument('--test_neg_N_item', required=True)
parser.add_argument('--bptt', default=def_opts["bptt"], type=int, required=True)

parser.add_argument('--max_seq_len', default=50, type=int)
parser.add_argument('--summary_dir', default="summary_path")
parser.add_argument('--save_dir', default="save_path")
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--restart', default=False, help='Can restart from a saved model from the summary folder, if available.')
parser.add_argument('--train-eval', default=False, help='Should evaluate the model on training data?')
parser.add_argument('--test-eval', default=True, help='Should evaluate the model on test data?')
parser.add_argument('--scale', default=1.0, type=float, help='Constant to scale the time fields by')
parser.add_argument('--batch-size', default=def_opts["batch_size"], type=int)
parser.add_argument('--learning_rate', default=def_opts["learning_rate"], type=float)

parser.add_argument('--cpu-only', default=def_opts["cpu_only"], help='Use only the CPU.')

parser.add_argument('--display_freq', default=100, type=int)
parser.add_argument('--eval_freq', default=1000, type=int)
parser.add_argument('--eval_rank_freq', default=3000, type=int)
parser.add_argument('--eval_rank_epoch_freq', default=5, type=int)

args = parser.parse_args()
args.bptt = args.bptt
args.learning_rate = Config.learning_rate
args.epochs = Config.train_num_epochs
args.batch_size = Config.batch_size
args.scale = Config.time_scale
args.summary_dir = os.path.join(args.folder_dataset_model, args.summary_dir)
args.save_dir =  os.path.join(args.folder_dataset_model, args.save_dir)

with open(os.path.join(args.folder_dataset_model, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()



def cmd(event_train, event_test, time_train, time_test,
        summary_dir, save_dir, num_epochs, restart, train_eval, test_eval, scale,
        batch_size, bptt, learning_rate, cpu_only):
    """Read data from EVENT_TRAIN_FILE, TIME_TRAIN_FILE and try to predict the values in EVENT_TEST_FILE, TIME_TEST_FILE."""
    # data = utils_tpp.read_data(
    #     event_train_file=event_train_file,
    #     event_test_file=event_test_file,
    #     time_train_file=time_train_file,
    #     time_test_file=time_test_file
    # )

    data = utils_tpp.read_data_pkl(event_train, event_test, time_train, time_test, args, pad=True, paddingStyle='post')



    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    sess = tf.Session(config=config)

    utils_tpp.data_stats(data)

    rmtpp_mdl = rmtpp_core.RMTPP(
        sess=sess,
        num_categories=data['num_categories'],
        summary_dir=summary_dir if summary_dir is not None else tempfile.mkdtemp(),
        save_dir=save_dir,
        batch_size=batch_size,
        bptt=bptt,
        learning_rate=learning_rate,
        cpu_only=cpu_only
    )

    # TODO: The finalize here has to be false because tf.global_variables()
    # creates a new graph node (why?). Hence, need to be extra careful while
    # saving the model.
    rmtpp_mdl.initialize(finalize=False)
    rmtpp_mdl.train(training_data=data, restart=restart,
                    with_summaries=summary_dir is not None,
                    num_epochs=num_epochs, with_evals=False, flag_train_mini=Config.flag_train_mini)

    if train_eval:
        print('\nEvaluation on training data:')
        train_time_preds, train_event_preds = rmtpp_mdl.predict_train(data=data)
        rmtpp_mdl.eval(train_time_preds, data['train_time_out_seq'],
                       train_event_preds, data['train_event_out_seq'], time_scale=data['time_scale'], K=Config.metrics_K)
        print()

    # if test_eval:
    #     print('\nEvaluation on testing data:')
    #     test_time_preds, test_event_preds = rmtpp_mdl.predict_test(data=data)
    #     rmtpp_mdl.eval(test_time_preds, data['test_time_out_seq'],
    #                    test_event_preds, data['test_event_out_seq'])

    if test_eval:
        print('\nEvaluation on testing data:')
        test_time_preds, test_event_preds = rmtpp_mdl.predict_test(data=data)
        rmtpp_mdl.eval(test_time_preds, data['test_time_out_seq'],
                       test_event_preds, data['test_event_out_seq'], time_scale=data['time_scale'], K=Config.metrics_K)

        # rmtpp_mdl.eval_last(test_time_preds, data['test_time_out_seq'],
        #                test_event_preds, data['test_event_out_seq'], Config.metrics_K)

    print("done")


if __name__ == '__main__':
    # loading data
    # load file: dataset_model.pickle
    dataset_file_name = os.path.join(args.folder_dataset, Config.dict_pkl_dataset[args.model])
    print "load file :", dataset_file_name
    (event_train, event_test, time_train, time_test) = Dataset.load_dataset(args.model, dataset_file_name)

    #train model
    cmd(event_train, event_test, time_train, time_test,
        args.summary_dir, args.save_dir, args.epochs, args.restart, args.train_eval, args.test_eval, args.scale,
        args.batch_size, args.bptt, args.learning_rate, args.cpu_only)
