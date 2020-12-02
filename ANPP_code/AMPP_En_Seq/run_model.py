#!/usr/bin/env python
import tempfile
import argparse
#import click
import tensorflow as tf
import model_core
import utils_tpp
import os
import sys
import numpy as np

sys.path.append("..")
from Config import *
from utils.Dataset import *
from utils.Metrics import *
from utils.DatasetSeq import *
import pickle


def process_args():
    # Network parameters

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--folder_dataset', required=True)
    parser.add_argument('--folder_dataset_model', required=True)
    parser.add_argument('--weight_loss', default='1,1', help="event,time")
    # parser.add_argument('--test_neg_N_item', required=True)
    # parser.add_argument('--bptt', default=def_opts["bptt"], type=int, required=True)

    parser.add_argument('--summary_dir', default="summary.AMPP_En_Seq")
    parser.add_argument('--save_dir', default="save.AMPP_En_Seq")
    parser.add_argument('--restart', default=False,
                        help='Can restart from a saved model from the summary folder, if available.')
    parser.add_argument('--train-eval', default=False, help='Should evaluate the model on training data?')
    parser.add_argument('--test-eval', default=True, help='Should evaluate the model on test data?')

    parser.add_argument('--cpu-only', default=False, help='Use only the CPU.')
    parser.add_argument('--device_gpu', default="/gpu:0")
    parser.add_argument('--device_cpu', default="/cpu:0")

    parser.add_argument('--display_freq', default=100, type=int)
    parser.add_argument('--eval_every_num_epochs', default=1, type=int)
    parser.add_argument('--eval_rank_freq', default=3000, type=int)
    parser.add_argument('--eval_rank_epoch_freq', default=5, type=int)

    parser.add_argument('--hidden_units', default=16, type=int)

    parser.add_argument('--float_type', default=tf.float32)
    parser.add_argument('--seed', default=42)

    parser.add_argument('--scale', default=1.0, type=float, help='Constant to scale the time fields by')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--max_T', default=20, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--decay_steps', default=100, type=int)
    parser.add_argument('--decay_rate', default=0.001, type=float)


    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--l2_penalty', default=0.0, type=float)
    parser.add_argument('--emb_size', default=4)
    parser.add_argument('--hidden_layer_size', default=16)


    parser.add_argument('--time_encode_method', default="pos", help='pos:encode by pos in seq; time_direct: map time to embedding directly; time_delta_prev_direct: map delta prev to embedding directly')
    parser.add_argument('--time_bucket_dim', default=10, type=int)
    parser.add_argument('--time_parse_log', default=False, help="parse time t to log(1+t)")
    parser.add_argument('--loss_time_method', default="intensity", help="intensity, mse, gaussian")


    # parse args
    args = parser.parse_args()
    args.seed = Config.seed_num
    args.epochs = Config.train_num_epochs
    args.batch_size = Config.batch_size
    args.learning_rate = Config.learning_rate
    args.scale = Config.time_scale
    args.time_method = Config.time_method
    args.time_bucket_dim = Config.time_bucket_dim
    args.eval_every_num_epochs = Config.eval_every_num_epochs
    args.l2_penalty =  float(Config.weight_loss_regularization)
    args.time_encode_method = Config.time_encode_method
    args.time_parse_log = Config.time_parse_log
    args.time_flag_parse_seq_to_startZero = Config.time_flag_parse_seq_to_startZero
    args.time_flagMinMax = Config.flagMinMax
    args.time_flagScale = Config.flagScale
    args.loss_time_method = Config.loss_time_method
    args.flag_train_mini = Config.flag_train_mini
    args.one_batch = False
    args.weight_loss = Config.weight_loss
    args.weight_loss_arr = np.array(args.weight_loss.split(','), float)
    args.weight_loss_event, args.weight_loss_time = args.weight_loss_arr[0], args.weight_loss_arr[1]
    args.sigma_square = Config.sigma_square

    args.num_blocks = Config.num_blocks
    args.num_heads = Config.num_heads
    args.hidden_units = Config.hidden_units
    args.hidden_layer_size = args.hidden_units


    # write args
    file_arg = "args.txt"
    with open(os.path.join(args.folder_dataset_model, file_arg), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    f.close()

    return args


def process_attentions(attention_score_blocks, event_in, time_in):
    print("process_attentions...")
    #[N, Blocks, heads, T_q, T_k]
    print("attention_score_blocks:", attention_score_blocks.shape)
    print("event_in:", event_in.shape)
    print("time_in:", time_in.shape)

    print(attention_score_blocks[0])
    print(event_in[0])
    print(time_in[0])

    file_attention = os.path.join(args.folder_dataset_model, "attention_%s_%s.pkl" % (str(args.dataset), str(args.max_T)))
    
    print("Attention score file: ", file_attention)
    with open(file_attention, 'wb') as f:
        pickle.dump(attention_score_blocks, f)

def cmd(event_train, event_test, time_train, time_test, args):
    #data = utils_tpp.read_data_pkl(event_train, event_test, time_train, time_test, args, pad=True, paddingStyle='post')

    data = DatasetSeq(settings={
            "eventTrain": event_train,
            "eventTest": event_test,
            "timeTrain": time_train,
            "timeTest": time_test,
            "args": args,
            },
            pad=True, paddingStyle='post')

    data.build_dataset()
    data.data_stats()
    #tils_tpp.data_stats(data)

    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    sess = tf.Session(config=config)
    seq_mdl = model_core.AMPP_En_Seq(
        sess=sess,
        num_categories=data.num_categories,
        square_std=data.square_std,
        args=args
    )

    # TODO: The finalize here has to be false because tf.global_variables()
    # creates a new graph node (why?). Hence, need to be extra careful while
    # saving the model.
    seq_mdl.initialize(finalize=False)
    seq_mdl.train(training_data=data, with_evals_train=False, with_evals_eval=True)

    #print("Restore best model...")
    #seq_mdl.restore_best_model()

    if args.train_eval:
        print('\nEvaluation on training data:')
        train_time_preds, train_event_preds = seq_mdl.predict_train(data=data)
        seq_mdl.eval(train_time_preds, data.data["train"]['time_out_seq'],
                       train_event_preds, data.data["train"]['event_out_seq'], time_scale=data.time_scale, K=Config.metrics_K, time_parse_log=args.time_parse_log)
        print()

    if args.test_eval:
        print('\nEvaluation on testing data:')

        test_time_preds, test_event_preds, all_attention_scores_blocks = seq_mdl.predict_test(data=data)
        seq_mdl.eval(test_time_preds, data.data["test"]['time_out_seq'],
                       test_event_preds, data.data["test"]['event_out_seq'], time_scale=data.time_scale, K=Config.metrics_K, time_parse_log=args.time_parse_log)
        #process_attentions(all_attention_scores_blocks, data.data["test"]['event_in_seq'], data.data["test"]['time_in_seq'])

    print("done")


def load_dataset(args):
    # load file: dataset_model.pickle
    dataset_file_name = os.path.join(args.folder_dataset_model, Config.dict_pkl_dataset[args.model])
    print "load file :", dataset_file_name
    (event_train, event_test, time_train, time_test) = Dataset.load_dataset(args.model, dataset_file_name)
    return (event_train, event_test, time_train, time_test)
if __name__ == '__main__':
    args = process_args()

    # loading data
    (event_train, event_test, time_train, time_test) = load_dataset(args)

    #train model
    cmd(event_train, event_test, time_train, time_test, args)
