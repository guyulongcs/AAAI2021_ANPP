import multiprocessing as MP
import os
from modules import *

import numpy as np
import tensorflow as tf
from scipy.integrate import quad
import shutil

from utils_tpp import create_dir, variable_summaries, MAE, MAE_norm, ACC, MAE_last, Rank_last

import sys
sys.path.append("..")
from Config import *
from utils.Dataset import *
from utils.Metrics import *

seed_num=Config.seed_num
np.random.seed(seed_num)
tf.random.set_random_seed(seed_num)

def softplus(x):
    """Numpy counterpart to tf.nn.softplus"""
    return np.log1p(np.exp(x))


def quad_func(t, c, w):
    """This is the t * f(t) function calculating the mean time to next event,
    given c, w."""
    return c * t * np.exp(-w * t + (c / w) * (np.exp(-w * t) - 1))


class AMPP_En_Seq:
    """Class implementing the AMPP_En_Seq model."""

    def __init__(self, sess, num_categories, square_std, args, reuse=None):
        self.sess = sess
        self.num_categories = num_categories

        #args
        self.summary_dir = args.summary_dir
        self.save_dir = args.save_dir
        self.restart = args.restart
        self.train_eval = args.train_eval
        self.test_eval = args.test_eval
        self.cpu_only = args.cpu_only
        self.device_gpu = args.device_gpu
        self.device_cpu = args.device_cpu
        self.display_freq = args.display_freq
        self.eval_every_num_epochs = args.eval_every_num_epochs
        self.eval_rank_freq = args.eval_rank_freq
        self.eval_rank_epoch_freq = args.eval_rank_epoch_freq
        self.hidden_units = args.hidden_units
        self.float_type = args.float_type
        self.seed = args.seed
        self.scale = args.scale
        self.epochs = args.epochs
        self.max_T = args.max_T
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.decay_steps = args.decay_steps
        self.decay_rate = args.decay_rate
        self.num_blocks = args.num_blocks
        self.num_heads = args.num_heads
        self.dropout_rate = args.dropout_rate
        self.l2_penalty = args.l2_penalty
        self.emb_size = args.emb_size
        self.hidden_layer_size = args.hidden_layer_size
        self.flag_train_mini = args.flag_train_mini
        self.one_batch = args.one_batch
        self.time_encode_method = args.time_encode_method
        self.time_bucket_dim = args.time_bucket_dim
        #
        self.check_nans = False
        #self.with_summaries = self.summary_dir is not None
        self.with_summaries = False

        self.last_epoch = 0
        self.rs = np.random.RandomState(self.seed)
        self.is_training = tf.placeholder(tf.bool, shape=())

        self.weight_loss = args.weight_loss
        self.weight_loss_event = args.weight_loss_event
        self.weight_loss_time = args.weight_loss_time

        self.loss_time_method = args.loss_time_method
        self.sigma_square = args.sigma_square
        self.sigma_square = square_std

        self.save_dir = os.path.join(args.folder_dataset_model, self.save_dir + self.weight_loss)
        self.summary_dir = os.path.join(args.folder_dataset_model, self.summary_dir + self.weight_loss)



        with tf.variable_scope("input"):
            with tf.device(self.device_gpu if not self.cpu_only else self.device_cpu):
                # Make input variables
                self.events_in = tf.placeholder(tf.int32, [None, self.max_T], name='events_in')
                self.times_in = tf.placeholder(self.float_type, [None, self.max_T], name='times_in')

                self.events_out = tf.placeholder(tf.int32, [None, self.max_T], name='events_out')
                self.times_out = tf.placeholder(self.float_type, [None, self.max_T], name='times_out')

                self.times_in_delta_prev = tf.placeholder(self.float_type, [None, self.max_T], name='times_in_delta_prev')

                self.times_in_delta_prev_bucket = tf.placeholder(tf.int32, [None, self.max_T],
                                                          name='times_in_delta_prev')

                self.batch_num_events = tf.placeholder(self.float_type, [], name='batch_num_events')

                self.inf_batch_size = tf.shape(self.events_in)[0]

                self.mask = tf.expand_dims(tf.to_float(tf.not_equal(self.events_in[:, :], 0)), -1)

                # Make variables
                with tf.variable_scope('hidden_state'):
                    self.Wt = tf.get_variable(name='Wt',
                                              shape=(1, self.hidden_units),
                                              dtype=self.float_type,
                                              initializer=tf.constant_initializer(
                                                  np.ones((1, self.hidden_units)) * 1e-3))

                #self attention
                with tf.variable_scope("Self_Attention", reuse=reuse):
                    # sequence embedding, item embedding table

                    # emb: item, category
                    self.seq_emb = {}
                    self.emb_table = {}

                    print("self.l2_penalty:", self.l2_penalty)
                    print("self.num_categories:", self.num_categories)
                    print("self.hidden_units:", self.hidden_units)

                    self.seq_emb, self.emb_table = embedding(self.events_in,
                                                                       vocab_size=self.num_categories,
                                                                       num_units=self.hidden_units,
                                                                       zero_pad=True,
                                                                       scale=True,
                                                                       l2_reg=self.l2_penalty,
                                                                       scope="input_embeddings_event",
                                                                       with_t=True,
                                                                       reuse=reuse
                                                                       )

                    self.t = None
                    with tf.variable_scope("time_encoding"):
                        # Positional Encoding
                        self.time_encode_pos, pos_emb_table = embedding(
                            # [B, T]
                            tf.tile(tf.expand_dims(tf.range(tf.shape(self.events_in)[-1]), 0),
                                    [tf.shape(self.events_in)[0], 1]),
                            vocab_size=self.max_T,
                            num_units=self.hidden_units,
                            zero_pad=False,
                            scale=False,
                            l2_reg=self.l2_penalty,
                            scope="dec_pos",
                            reuse=reuse,
                            with_t=True
                        )

                        # time direct Encoding
                        self.time_encode_value_direct_encode = tf.tensordot(tf.expand_dims(self.times_in,axis=-1), self.Wt, axes=[-1,0])

                        # time delta prev Encoding
                        self.time_encode_delta_prev_direct_encode = tf.tensordot(tf.expand_dims(self.times_in_delta_prev,axis=-1), self.Wt, axes=[-1,0])

                        # time delta prev bucket Encoding
                        self.time_encode_delta_prev_bucket_encode, self.emb_table_time_bucket = embedding(
                                                                 self.times_in_delta_prev_bucket,
                                                                 vocab_size=self.time_bucket_dim,
                                                                 num_units=self.hidden_units,
                                                                 zero_pad=False,
                                                                 scale=True,
                                                                 l2_reg=self.l2_penalty,
                                                                 scope="input_embeddings_time_bucket",
                                                                 with_t=True,
                                                                 reuse=reuse
                                                                 )


                        #set time emedding
                        if(self.time_encode_method == "time_direct"):
                            self.t = self.time_encode_value_direct_encode
                        if (self.time_encode_method == "time_delta_prev_direct"):
                            self.t = self.time_encode_delta_prev_direct_encode
                        if (self.time_encode_method == "time_delta_prev_bucket"):
                            self.t = self.time_encode_delta_prev_bucket_encode
                        else:
                            self.t = self.time_encode_pos

                    # input embedding merge

                    self.seq = tf.add(self.seq_emb, self.t)

                    # Dropout
                    self.seq = tf.layers.dropout(self.seq,
                                                 rate=self.dropout_rate,
                                                 training=tf.convert_to_tensor(self.is_training))
                    self.seq *= self.mask

                    # Build blocks
                    #[Batch, Blocks, heads, T_q, T_k]
                    # self.attention_scores_blocks=tf.zeros([self.inf_batch_size, self.num_blocks, self.num_heads,self.max_T,self.max_T], tf.int32)
                    list_attention_scores_blocks= []
                    for i in range(self.num_blocks):
                        with tf.variable_scope("num_blocks_%d" % i):
                            # Self-attention
                            #attention_scores:[Batch, heads, T_q, T_k]
                            self.seq, self.attention_scores = multihead_attention(queries=normalize(self.seq),
                                                           keys=self.seq,
                                                           num_units=self.hidden_units,
                                                           num_heads=self.num_heads,
                                                           dropout_rate=self.dropout_rate,
                                                           is_training=self.is_training,
                                                           causality=True,
                                                           scope="self_attention",
                                                           with_att_score=True
                                                           )
                            list_attention_scores_blocks.append(self.attention_scores)
                            # Feed forward
                            self.seq = feedforward(normalize(self.seq),
                                                   num_units=[self.hidden_units, self.hidden_units],
                                                   dropout_rate=self.dropout_rate, is_training=self.is_training)
                            self.seq *= self.mask

                    self.attention_scores_blocks = tf.stack(list_attention_scores_blocks)
                    #[B, T, H]
                    self.seq = normalize(self.seq)


                with tf.variable_scope('output'):
                    self.wt = tf.get_variable(name='wt', shape=(1, 1),
                                              dtype=self.float_type,
                                              initializer=tf.constant_initializer(1.0))

                    self.Wy = tf.get_variable(name='Wy', shape=(self.emb_size, self.hidden_layer_size),
                                              dtype=self.float_type,
                                              initializer=tf.constant_initializer(np.ones((self.emb_size, self.hidden_layer_size)) * 0.0))

                    # The first column of Vy is merely a placeholder (will not be trained).
                    self.Vy = tf.get_variable(name='Vy', shape=(self.hidden_layer_size, self.num_categories),
                                              dtype=self.float_type,
                                              initializer=tf.constant_initializer(np.ones((self.hidden_layer_size, self.num_categories)) * 0.001))
                    self.Vt = tf.get_variable(name='Vt', shape=(self.hidden_layer_size, 1),
                                              dtype=self.float_type,
                                              initializer=tf.constant_initializer(np.ones((self.hidden_layer_size, 1)) * 0.001))
                    self.bt = tf.get_variable(name='bt', shape=(1, 1),
                                              dtype=self.float_type,
                                              initializer=tf.constant_initializer(np.log(1.0)))
                    self.bk = tf.get_variable(name='bk', shape=(1, self.num_categories),
                                              dtype=self.float_type,
                                              initializer=tf.constant_initializer(np.ones((1, self.num_categories)) * 0.0))

                self.all_vars = [self.Wt, self.wt, self.Wy, self.Vy, self.Vt, self.bt, self.bk]

                # Add summaries for all (trainable) variables
                with tf.device(self.device_cpu):
                    for v in self.all_vars:
                        variable_summaries(v)

                # Initial state for GRU cells
                self.initial_state = state = tf.zeros([self.inf_batch_size, self.hidden_layer_size],
                                                      dtype=self.float_type,
                                                      name='initial_state')


                self.initial_time = last_time = tf.zeros((self.inf_batch_size,),
                                                         dtype=self.float_type,
                                                         name='initial_time')

                self.loss = 0.0
                ones_2d = tf.ones((self.inf_batch_size, 1), dtype=self.float_type)
                # ones_1d = tf.ones((self.inf_batch_size,), dtype=self.float_type)

                self.hidden_states = []
                self.event_preds = []
                self.time_preds = []

                self.time_LLs = []
                self.mark_LLs = []
                self.log_lambdas = []
                # self.delta_ts = []
                self.times = []


                with tf.name_scope('max_T'):
                    for i in range(self.max_T):

                        time = self.times_in[:, i]
                        time_next = self.times_out[:, i]

                        delta_t_prev = tf.expand_dims(time - last_time, axis=-1)
                        delta_t_next = tf.expand_dims(time_next - time, axis=-1)

                        last_time = time

                        time_2d = tf.expand_dims(time, axis=-1)

                        # output, state = RNNcell(events_embedded, state)

                        # TODO Does TF automatically broadcast? Then we'll not
                        # need multiplication with tf.ones
                        type_delta_t = True

                        with tf.name_scope('loss_calc'):
                            state = self.seq[:, i, :]
                            base_intensity = tf.matmul(ones_2d, self.bt)
                            # wt_non_zero = tf.sign(self.wt) * tf.maximum(1e-9, tf.abs(self.wt))
                            wt_soft_plus = tf.nn.softplus(self.wt)

                            log_lambda_ = (tf.matmul(state, self.Vt) +
                                           (-delta_t_next * wt_soft_plus) +
                                           base_intensity)

                            lambda_ = tf.exp(tf.minimum(10.0, log_lambda_), name='lambda_')

                            log_f_star = (log_lambda_ -
                                          (1.0 / wt_soft_plus) * tf.exp(tf.minimum(50.0, tf.matmul(state, self.Vt) + base_intensity)) +
                                          (1.0 / wt_soft_plus) * lambda_)

                            time_preds = None
                            if (self.loss_time_method == "intensity"):
                                time_LL = log_f_star
                            else:
                                # Intensity RNN: MSE
                                # [B, 1]
                                time_preds = tf.matmul(state, self.Vt) + base_intensity
                                time_loss_mse = -tf.subtract(time_preds, delta_t_next)
                                time_loss_mse = -tf.pow(time_loss_mse, 2)

                                time_loss_gaussian = tf.multiply(
                                    tf.divide(1, tf.sqrt(2 * tf.constant(np.pi) * tf.maximum(0.001, self.sigma_square))), tf.exp(
                                        tf.minimum(30.0,
                                        tf.divide(tf.pow(tf.subtract(time_preds, delta_t_next), 2),
                                                  -2 * tf.maximum(0.001, self.sigma_square)))))

                                time_loss_gaussian = tf.log(tf.maximum(0.001, time_loss_gaussian))

                                time_preds = tf.squeeze(time_preds, axis=-1)

                                if (self.loss_time_method == "mse"):
                                    time_LL = time_loss_mse
                                if (self.loss_time_method == "gaussian"):
                                    time_LL = time_loss_gaussian

                            events_pred = tf.nn.softmax(
                                tf.minimum(50.0,
                                           tf.matmul(state, self.Vy) + ones_2d * self.bk),
                                name='Pr_events'
                            )


                            mark_LL = tf.expand_dims(
                                tf.log(
                                    tf.maximum(
                                        1e-6,
                                        tf.gather_nd(
                                            events_pred,
                                            tf.concat([
                                                tf.expand_dims(tf.range(self.inf_batch_size), -1),
                                                tf.expand_dims(tf.mod(self.events_out[:, i] - 1, self.num_categories), -1)
                                            ], axis=1, name='Pr_next_event'
                                            )
                                        )
                                    )
                                ), axis=-1, name='log_Pr_next_event'
                            )
                            #step_LL = time_LL + mark_LL
                            step_LL = tf.add(tf.multiply(time_LL, self.weight_loss_time), tf.multiply(mark_LL, self.weight_loss_event))


                            # In the batch some of the sequences may have ended before we get to the
                            # end of the seq. In such cases, the events will be zero.
                            # TODO Figure out how to do this with RNNCell, LSTM, etc.
                            num_events = tf.reduce_sum(tf.where(self.events_in[:, i] > 0,
                                                       tf.ones(shape=(self.inf_batch_size,), dtype=self.float_type),
                                                       tf.zeros(shape=(self.inf_batch_size,), dtype=self.float_type)),
                                                       name='num_events')

                            self.loss -= tf.reduce_sum(
                                tf.where(self.events_in[:, i] > 0,
                                         tf.squeeze(step_LL) / self.batch_num_events,
                                         tf.zeros(shape=(self.inf_batch_size,)))
                            )


                        self.time_LLs.append(time_LL)
                        self.mark_LLs.append(mark_LL)
                        self.log_lambdas.append(log_lambda_)

                        self.hidden_states.append(state)
                        self.event_preds.append(events_pred)
                        if (self.loss_time_method != "intensity"):
                            self.time_preds.append(time_preds)

                        # self.delta_ts.append(tf.clip_by_value(delta_t, 0.0, np.inf))
                        self.times.append(time)

                self.final_state = self.hidden_states[-1]

                with tf.device(self.device_cpu):
                    # Global step needs to be on the CPU (Why?)
                    self.global_step = tf.Variable(0, name='global_step', trainable=False)

                self.learning_rate = tf.train.inverse_time_decay(self.learning_rate,
                                                                 global_step=self.global_step,
                                                                 decay_steps=self.decay_steps,
                                                                 decay_rate=self.decay_rate)
                # self.global_step is incremented automatically by the
                # optimizer.

                # self.increment_global_step = tf.assign(
                #     self.global_step,
                #     self.global_step + 1,
                #     name='update_global_step'
                # )

                # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

                # Capping the gradient before minimizing.
                # update = optimizer.minimize(loss)

                # Performing manual gradient clipping.
                self.gvs = self.optimizer.compute_gradients(self.loss)
                # update = optimizer.apply_gradients(gvs)

                # capped_gvs = [(tf.clip_by_norm(grad, 100.0), var) for grad, var in gvs]
                grads, vars_ = list(zip(*self.gvs))

                self.norm_grads, self.global_norm = tf.clip_by_global_norm(grads, 10.0)
                capped_gvs = list(zip(self.norm_grads, vars_))

                with tf.device(self.device_cpu):
                    tf.contrib.training.add_gradients_summaries(self.gvs)
                    # for g, v in zip(grads, vars_):
                    #     variable_summaries(g, name='grad-' + v.name.split('/')[-1][:-2])

                    variable_summaries(self.loss, name='loss')
                    variable_summaries(self.hidden_states, name='agg-hidden-states')
                    variable_summaries(self.event_preds, name='agg-event-preds-softmax')
                    variable_summaries(self.time_LLs, name='agg-time-LL')
                    variable_summaries(self.mark_LLs, name='agg-mark-LL')
                    variable_summaries(self.time_LLs + self.mark_LLs, name='agg-total-LL')
                    # variable_summaries(self.delta_ts, name='agg-delta-ts')
                    variable_summaries(self.times, name='agg-times')
                    variable_summaries(self.log_lambdas, name='agg-log-lambdas')
                    variable_summaries(tf.nn.softplus(self.wt), name='wt-soft-plus')

                    self.tf_merged_summaries = tf.summary.merge_all()

                self.update = self.optimizer.apply_gradients(capped_gvs,
                                                             global_step=self.global_step)

                self.tf_init = tf.global_variables_initializer()
                # self.check_nan = tf.add_check_numerics_ops()

    def initialize(self, finalize=False):
        """Initialize the global trainable variables."""
        self.sess.run(self.tf_init)

        if finalize:
            # This prevents memory leaks by disallowing changes to the graph
            # after initialization.
            self.sess.graph.finalize()


    def make_feed_dict(self, training_data, batch_idxes, bptt_idx, args,
                       init_hidden_state=None):
        """Creates a batch for the given batch_idxes starting from bptt_idx.
        The hidden state will be initialized with all zeros if no such state is
        provided.
        """

        if init_hidden_state is None:
            cur_state = np.zeros((len(batch_idxes), self.hidden_layer_size))
        else:
            cur_state = init_hidden_state

        train_event_in_seq = training_data['train_event_in_seq']
        train_time_in_seq = training_data['train_time_in_seq']
        train_event_out_seq = training_data['train_event_out_seq']
        train_time_out_seq = training_data['train_time_out_seq']

        batch_event_train_in = train_event_in_seq[batch_idxes, :]
        batch_event_train_out = train_event_out_seq[batch_idxes, :]
        batch_time_train_in = train_time_in_seq[batch_idxes, :]
        batch_time_train_out = train_time_out_seq[batch_idxes, :]

        bptt_range = range(bptt_idx, (bptt_idx + self.max_T))
        bptt_event_in = batch_event_train_in[:, bptt_range]
        bptt_event_out = batch_event_train_out[:, bptt_range]
        bptt_time_in = batch_time_train_in[:, bptt_range]
        bptt_time_out = batch_time_train_out[:, bptt_range]

        if bptt_idx > 0:
            initial_time = batch_time_train_in[:, bptt_idx - 1]
        else:
            initial_time = np.zeros(batch_time_train_in.shape[0])

        feed_dict = {
            self.initial_state: cur_state,
            self.initial_time: initial_time,
            self.events_in: bptt_event_in,
            self.events_out: bptt_event_out,
            self.times_in: bptt_time_in,
            self.times_out: bptt_time_out,
            self.batch_num_events: np.sum(batch_event_train_in > 0)
        }

        return feed_dict

    def train(self, training_data, with_evals_train=False, with_evals_eval=False):
        """Train the model given the training data.

        If with_evals is an integer, then that many elements from the test set
        will be tested.
        """

        create_dir(self.save_dir)
        ckpt = tf.train.get_checkpoint_state(self.save_dir)

        saver = tf.train.Saver(tf.global_variables())

        if self.with_summaries:
            train_writer = tf.summary.FileWriter(self.summary_dir + '/train',
                                                 self.sess.graph)


        if ckpt and self.restart:
            print('Restoring from {}'.format(ckpt.model_checkpoint_path))
            saver.restore(self.sess, ckpt.model_checkpoint_path)

        train_event_in_seq = training_data.data["train"]['event_in_seq']
        train_time_in_seq = training_data.data["train"]['time_in_seq']
        train_event_out_seq = training_data.data["train"]['event_out_seq']
        train_time_out_seq = training_data.data["train"]['time_out_seq']
        train_time_in_seq_delta_prev = training_data.data["train"]["time_in_seq_delta_prev"]
        train_time_in_seq_delta_prev_bucket = training_data.data["train"]["time_in_seq_delta_prev_bucket"]


        idxes = list(range(len(train_event_in_seq)))
        n_batches = len(idxes) // self.batch_size

        best_MAE, best_NDCG, best_Hit = 0., 0., 0.
        best_valid_loss = sys.float_info.max
        stopping_step = 0
        for epoch in range(self.last_epoch, self.last_epoch + self.epochs):
            self.rs.shuffle(idxes)

            print("\nStarting epoch %d..." % epoch)
            total_loss = 0.0

            for batch_idx in range(n_batches):
                # TODO: This is horribly inefficient. Move this to a separate
                # thread using FIFOQueues.
                # However, the previous state from max_T still needs to be
                # passed to the next max_T batch. To make this efficient, we
                # will need to set and preserve the previous state in a
                # tf.Variable.
                #  - Sounds like a job for tf.placeholder_with_default?
                #  - Or, of a variable with optinal default?

                batch_idxes = idxes[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
                batch_event_train_in = train_event_in_seq[batch_idxes, :]
                batch_event_train_out = train_event_out_seq[batch_idxes, :]
                batch_time_train_in = train_time_in_seq[batch_idxes, :]
                batch_time_train_out = train_time_out_seq[batch_idxes, :]
                batch_train_time_in_seq_delta_prev = train_time_in_seq_delta_prev[batch_idxes, :]
                batch_train_time_in_seq_delta_prev_bucket = train_time_in_seq_delta_prev_bucket[batch_idxes, :]


                cur_state = np.zeros((self.batch_size, self.hidden_layer_size))
                batch_loss = 0.0

                batch_num_events = np.sum(batch_event_train_in > 0)
                for bptt_idx in range(0, len(batch_event_train_in[0]) - self.max_T, self.max_T):
                    bptt_range = range(bptt_idx, (bptt_idx + self.max_T))
                    bptt_event_in = batch_event_train_in[:, bptt_range]
                    bptt_event_out = batch_event_train_out[:, bptt_range]
                    bptt_time_in = batch_time_train_in[:, bptt_range]
                    bptt_time_out = batch_time_train_out[:, bptt_range]
                    bptt_time_in_delta_prev = batch_train_time_in_seq_delta_prev[:, bptt_range]
                    bptt_time_in_delta_prev_bucket = batch_train_time_in_seq_delta_prev_bucket[:, bptt_range]



                    if np.all(bptt_event_in[:, 0] == 0):
                        # print('Breaking at bptt_idx {} / {}'
                        #       .format(bptt_idx // self.max_T,
                        #               (len(batch_event_train_in[0]) - self.max_T) // self.max_T))
                        break

                    if bptt_idx > 0:
                        initial_time = batch_time_train_in[:, bptt_idx - 1]
                    else:
                        initial_time = np.zeros(batch_time_train_in.shape[0])

                    feed_dict = {
                        self.initial_state: cur_state,
                        self.initial_time: initial_time,
                        self.events_in: bptt_event_in,
                        self.events_out: bptt_event_out,
                        self.times_in: bptt_time_in,
                        self.times_out: bptt_time_out,
                        self.times_in_delta_prev: bptt_time_in_delta_prev,
                        self.times_in_delta_prev_bucket: bptt_time_in_delta_prev_bucket,
                        self.batch_num_events: batch_num_events,
                        self.is_training: True
                    }

                    if self.check_nans:
                        raise NotImplemented('tf.add_check_numerics_ops is '
                                             'incompatible with tf.cond and '
                                             'tf.while_loop.')
                        # _, _, cur_state, loss_ = \
                        #     self.sess.run([self.check_nan, self.update,
                        #                    self.final_state, self.loss],
                        #                   feed_dict=feed_dict)
                    else:
                        if self.with_summaries:
                            _, summaries, cur_state, loss_, step = \
                                self.sess.run([self.update,
                                               self.tf_merged_summaries,
                                               self.final_state,
                                               self.loss,
                                               self.global_step],
                                              feed_dict=feed_dict)
                            #train_writer.add_summary(summaries, step)
                        else:
                            _, cur_state, loss_ = \
                                self.sess.run([self.update,
                                               self.final_state, self.loss],
                                              feed_dict=feed_dict)
                    batch_loss += loss_

                total_loss += batch_loss
                if batch_idx % 10 == 0:
                    print("\tLoss during Epoch {} / {}, batch {} / {}: batch loss = {:.5f}, Epoch loss: {:.5f}, "
                      "lr = {:.5f}".format(epoch, self.epochs, batch_idx, n_batches, batch_loss, total_loss, self.sess.run(self.learning_rate)))

                if(self.flag_train_mini and batch_idx > 1000):
                    break

                if self.one_batch:
                    print('Breaking after just one batch.')
                    break

            #epoch done
            # self.sess.run(self.increment_global_step)
            print("\tLoss on last epoch = {:.4f}, new lr = {:.5f}, global_step = {}"
                  .format(total_loss / n_batches,
                          self.sess.run(self.learning_rate),
                          self.sess.run(self.global_step)))

            #eval on eval data
            # if (with_evals_eval and epoch % self.eval_every_num_epochs == 0):
            #     print('\tRunning evaluation on evaluation data: ...')
            #     # eval_time_preds, eval_event_preds = self.predict_eval(data=training_data)
            #     # self.eval(eval_time_preds, training_data['eval_time_out_seq'], eval_event_preds, training_data['eval_event_out_seq'], time_scale=training_data['time_scale'],
            #     #          K=Config.metrics_K)
				# #
            #     # print '\tEvaluating',
            #     valid_MAE, valid_NDCG, valid_Hit, valid_loss = self.validation_model_dataset(data=training_data)
            #     print("\tvalid_loss:%f" % valid_loss)
            #     if(valid_loss < best_valid_loss):
            #         self.save_best_model()
            #         best_valid_loss = valid_loss
			#
            #         #early stopping
            #         stopping_step = 0
            #     else:
            #         stopping_step += 1
            #         #print("valid_loss, best_valid_loss:%f, %f" % (valid_loss, best_valid_loss))
            #     if stopping_step >= Config.early_stopping_epochs:
            #         print("Early stopping is trigger at step: {} loss:{}".format(self.sess.run(self.global_step), valid_loss))
            #         break

        # checkpoint_path = os.path.join(self.save_dir, 'model.ckpt')
        # saver.save(self.sess, checkpoint_path, global_step=self.global_step)
        # print('Model saved at {}'.format(checkpoint_path))

        # Remember how many epochs we have trained.
        self.last_epoch += self.epochs

        if(with_evals_train):
            if isinstance(with_evals_train, int):
                batch_size = with_evals_train
            else:
                batch_size = len(training_data.data["train"]['event_in_seq'])

            print('Running evaluation on training data: ...')
            train_time_preds, train_event_preds = self.predict_train(training_data, batch_size=batch_size)
            self.eval(train_time_preds[0:batch_size], train_time_out_seq[0:batch_size],
                      train_event_preds[0:batch_size], train_event_out_seq[0:batch_size])

    def save_best_model(self):
        saver = tf.train.Saver(tf.global_variables())
        checkpoint_path = os.path.join(self.save_dir, 'best_model.ckpt')
        saver.save(self.sess, checkpoint_path, global_step=self.global_step)
        print('Best model saved at {}'.format(checkpoint_path))

    def restore_best_model(self):
        """Restore the model from saved state."""
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(self.save_dir)
        print('Loading best model from {}'.format(ckpt.model_checkpoint_path))
        saver.restore(self.sess, ckpt.model_checkpoint_path)

    def restore(self, args):
        """Restore the model from saved state."""
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(self.save_dir)
        print('Loading the model from {}'.format(ckpt.model_checkpoint_path))
        saver.restore(self.sess, ckpt.model_checkpoint_path)



    def predict(self, event_in_seq, time_in_seq, time_in_seq_delta_prev, time_in_seq_delta_prev_bucket, flagReturnAttention=False, flagIsValidation=False, single_threaded=False, MP_Pool=100):
        """Treats the entire dataset as a single batch and processes it."""

        all_hidden_states = []
        all_event_preds = []
        all_time_preds = []
        #[Batch, Blocks, heads, T_q, T_k]
        all_attention_scores_blocks = []
        cur_state = np.zeros((len(event_in_seq), self.hidden_layer_size))


        #print("event_in_seq:", len(event_in_seq), len(event_in_seq[0]))
        for bptt_idx in range(0, len(event_in_seq[0]) - self.max_T, self.max_T):
            bptt_range = range(bptt_idx, (bptt_idx + self.max_T))
            bptt_event_in = event_in_seq[:, bptt_range]
            bptt_time_in = time_in_seq[:, bptt_range]
            bptt_time_in_delta_prev = time_in_seq_delta_prev[:, bptt_range]
            bptt_time_in_delta_prev_bucket = time_in_seq_delta_prev_bucket[:, bptt_range]

            if bptt_idx > 0:
                initial_time = event_in_seq[:, bptt_idx - 1]
            else:
                initial_time = np.zeros(bptt_time_in.shape[0])

            feed_dict = {
                self.initial_state: cur_state,
                self.initial_time: initial_time,
                self.events_in: bptt_event_in,
                self.times_in: bptt_time_in,
                self.times_in_delta_prev: bptt_time_in_delta_prev,
                self.times_in_delta_prev_bucket: bptt_time_in_delta_prev_bucket,
                self.is_training: False
            }

            if(flagIsValidation):
                bptt_hidden_states, bptt_events_pred, cur_state, bptt_loss = self.sess.run(
                    [self.hidden_states, self.event_preds, self.final_state, self.loss],
                    feed_dict=feed_dict
                )

                all_hidden_states.extend(bptt_hidden_states)
                all_event_preds.extend(bptt_events_pred)

            else:
                bptt_hidden_states, bptt_events_pred, cur_state, bptt_time_preds, bptt_attention_scores_blocks = self.sess.run(
                    [self.hidden_states, self.event_preds, self.final_state, self.time_preds, self.attention_scores_blocks],
                    feed_dict=feed_dict
                )

                all_hidden_states.extend(bptt_hidden_states)
                all_event_preds.extend(bptt_events_pred)

                #before: [Blocks, Batch, heads, T_q, T_k], after:[Batch, Blocks, heads, T_q, T_k]
                bptt_attention_scores_blocks = bptt_attention_scores_blocks.swapaxes(0,1)
                #print("bptt_attention_scores_blocks:", bptt_attention_scores_blocks.shape)
                all_attention_scores_blocks.extend(bptt_attention_scores_blocks)

                if (self.loss_time_method != "intensity"):
                    all_time_preds.extend(bptt_time_preds)


        if (self.loss_time_method == "intensity"):
            print("predict time intensity...")
            # TODO: This calculation is completely ignoring the clipping which
            # happens during the inference step.
            [Vt, bt, wt]  = self.sess.run([self.Vt, self.bt, self.wt])
            wt = softplus(wt)

            global _quad_worker
            def _quad_worker(params):
                idx, h_i = params
                preds_i = []
                C = np.exp(np.dot(h_i, Vt) + bt).reshape(-1)

                for c_, t_last in zip(C, time_in_seq[:, idx]):
                    args = (c_, wt)
                    upbound = np.inf
                    #upbound = Config.split_sequence_time_gap_max
                    val, _err = quad(quad_func, 0, upbound, args=args)
                    preds_i.append(t_last + val)

                return preds_i

            if single_threaded:
                all_time_preds = [_quad_worker((idx, x)) for idx, x in enumerate(all_hidden_states)]
            else:
                pool = MP.Pool(MP_Pool)
                all_time_preds = pool.map(_quad_worker, enumerate(all_hidden_states))
                #with MP.Pool(10) as pool:
                #    all_time_preds = pool.map(_quad_worker, enumerate(all_hidden_states))

        #print("all_time_preds", np.asarray(all_time_preds).shape)
        #print("all_event_preds", np.asarray(all_event_preds).shape)

        all_time_preds = np.asarray(all_time_preds).T
        all_event_preds = np.asarray(all_event_preds).swapaxes(0, 1)
        #[N, Blocks, heads, T_q, T_k]
        all_attention_scores_blocks = np.asarray(all_attention_scores_blocks)
        #print("all_time_preds", all_time_preds.shape)
        #print("all_event_preds", all_event_preds.shape)
        print("all_attention_scores_blocks:", all_attention_scores_blocks.shape)

        if(flagReturnAttention):
            return all_time_preds, all_event_preds, all_attention_scores_blocks
        else:
            return all_time_preds, all_event_preds

    def validation(self, event_in_seq, time_in_seq, time_in_seq_delta_prev, time_in_seq_delta_prev_bucket, event_out_seq, time_out_seq, single_threaded=False, MP_Pool=100):
        """Treats the entire dataset as a single batch and processes it."""

        all_hidden_states = []
        all_event_preds = []
        cur_state = np.zeros((len(event_in_seq), self.hidden_layer_size))

        loss_arr=[]
        #print("event_in_seq:", len(event_in_seq), len(event_in_seq[0]))
        for bptt_idx in range(0, len(event_in_seq[0]) - self.max_T, self.max_T):
            bptt_range = range(bptt_idx, (bptt_idx + self.max_T))
            bptt_event_in = event_in_seq[:, bptt_range]
            bptt_time_in = time_in_seq[:, bptt_range]
            bptt_time_in_delta_prev = time_in_seq_delta_prev[:, bptt_range]
            bptt_time_in_seq_delta_prev_bucket = time_in_seq_delta_prev_bucket[:, bptt_range]
            bptt_event_out = event_out_seq[:, bptt_range]
            bptt_time_out = time_out_seq[:, bptt_range]

            if bptt_idx > 0:
                initial_time = event_in_seq[:, bptt_idx - 1]
            else:
                initial_time = np.zeros(bptt_time_in.shape[0])

            batch_num_events = np.sum(bptt_event_in > 0)
            feed_dict = {
                self.initial_state: cur_state,
                self.initial_time: initial_time,
                self.events_in: bptt_event_in,
                self.times_in: bptt_time_in,
                self.times_in_delta_prev: bptt_time_in_delta_prev,
                self.times_in_delta_prev_bucket: bptt_time_in_seq_delta_prev_bucket,
                self.events_out: bptt_event_out,
                self.times_out: bptt_time_out,
                self.batch_num_events: batch_num_events,
                self.is_training: False
            }

            bptt_hidden_states, bptt_events_pred, cur_state, bptt_loss = self.sess.run(
                [self.hidden_states, self.event_preds, self.final_state, self.loss],
                feed_dict=feed_dict
            )

            all_hidden_states.extend(bptt_hidden_states)
            all_event_preds.extend(bptt_events_pred)
            loss_arr.append(bptt_loss)


        # TODO: This calculation is completely ignoring the clipping which
        # happens during the inference step.
        [Vt, bt, wt]  = self.sess.run([self.Vt, self.bt, self.wt])
        wt = softplus(wt)

        global _quad_worker
        def _quad_worker(params):
            idx, h_i = params
            preds_i = []
            C = np.exp(np.dot(h_i, Vt) + bt).reshape(-1)

            for c_, t_last in zip(C, time_in_seq[:, idx]):
                args = (c_, wt)
                upbound = np.inf
                #upbound = Config.split_sequence_time_gap_max
                val, _err = quad(quad_func, 0, upbound, args=args)
                preds_i.append(t_last + val)

            return preds_i

        if single_threaded:
            all_time_preds = [_quad_worker((idx, x)) for idx, x in enumerate(all_hidden_states)]
        else:
            pool = MP.Pool(MP_Pool)
            all_time_preds = pool.map(_quad_worker, enumerate(all_hidden_states))


        all_time_preds = np.asarray(all_time_preds).T
        all_event_preds = np.asarray(all_event_preds).swapaxes(0, 1)

        #print("all_time_preds", all_time_preds.shape)
        #print("all_event_preds", all_event_preds.shape)

        loss_avg = np.average(np.array(loss_arr))
        return all_time_preds, all_event_preds, loss_avg

    def predict_last(self, event_in_seq, time_in_seq, single_threaded=False):
        """Treats the entire dataset as a single batch and processes it."""

        print("predict_last:")
        print("event_in_seq.shape:", event_in_seq.shape)
        all_hidden_states = []
        all_event_preds = []

        cur_state = np.zeros((len(event_in_seq), self.hidden_layer_size))

        #for bptt_idx in range(0, len(event_in_seq[0]) - self.max_T, self.max_T):


        for bptt_idx in [len(event_in_seq[0]) - self.max_T - 1]:
            bptt_range = range(bptt_idx, (bptt_idx + self.max_T))
            bptt_event_in = event_in_seq[:, bptt_range]
            bptt_time_in = time_in_seq[:, bptt_range]

            if bptt_idx > 0:
                initial_time = event_in_seq[:, bptt_idx - 1]
            else:
                initial_time = np.zeros(bptt_time_in.shape[0])

            feed_dict = {
                self.initial_state: cur_state,
                self.initial_time: initial_time,
                self.events_in: bptt_event_in,
                self.times_in: bptt_time_in,
                self.is_training: False
            }

            bptt_hidden_states, bptt_events_pred, cur_state = self.sess.run(
                [self.hidden_states, self.event_preds, self.final_state],
                feed_dict=feed_dict
            )

            all_hidden_states.extend(bptt_hidden_states)
            all_event_preds.extend(bptt_events_pred)

        # TODO: This calculation is completely ignoring the clipping which
        # happens during the inference step.
        [Vt, bt, wt]  = self.sess.run([self.Vt, self.bt, self.wt])
        wt = softplus(wt)

        global _quad_worker
        def _quad_worker(params):
            idx, h_i = params
            preds_i = []
            C = np.exp(np.dot(h_i, Vt) + bt).reshape(-1)

            for c_, t_last in zip(C, time_in_seq[:, idx]):
                args = (c_, wt)
                upbound = np.inf
                # upbound = Config.split_sequence_time_gap_max
                val, _err = quad(quad_func, 0, upbound, args=args)
                preds_i.append(t_last + val)

            return preds_i

        if single_threaded:
            all_time_preds = [_quad_worker((idx, x)) for idx, x in enumerate(all_hidden_states)]
        else:
            pool = MP.Pool(100)
            all_time_preds = pool.map(_quad_worker, enumerate(all_hidden_states))
            #with MP.Pool(10) as pool:
            #    all_time_preds = pool.map(_quad_worker, enumerate(all_hidden_states))

        return np.asarray(all_time_preds).T, np.asarray(all_event_preds).swapaxes(0, 1)

    def eval(self, time_preds, time_true, event_preds, event_true, time_scale=1, K=20, time_parse_log=False):
        """Prints evaluation of the model on the given dataset."""
        # Print test error once every epoch:
        mae, total_valid = MAE_norm(time_preds, time_true, event_true, time_scale, time_parse_log)

        print("\t** MAE = {:.3f}; ACC = {:.3f}; valid = {}".format(
            mae, ACC(event_preds, event_true), total_valid))
        N, NDCG, Hit = Metrics.NDCG_Hit_Prob_checkValid(event_preds, event_true, K)
        print("\t** NDCG:{:.3f}, Hit:{:.3f}, K:{}, valid:{}".format( NDCG, Hit, K, N))

        return MAE, NDCG, Hit

    def eval_last(self, time_preds, time_true, event_preds, event_true, K):
        """Prints evaluation of the model on the given dataset."""
        # Print test error once every epoch:
        #time_preds: [B, T]
        #event_preds: [B, T, V]
        MAE = MAE_last(time_preds[:, -1], time_true[:, -1])
        N, acc, NDCG_K, Hit_K = Rank_last(event_preds[:, -1], event_true[:, -1]-1, K)
        print("** N = %d, MAE = %.3f; ACC = %.3f;  NDCG_K = %.3f, Hit_K = %.3f" % (
            N, MAE, acc, NDCG_K, Hit_K))

    def predict_test(self, data, single_threaded=False):
        """Make (time, event) predictions on the test data."""
        test_time_preds, test_event_preds, test_attention_scores_blocks = self.predict(event_in_seq=data.data["test"]['event_in_seq'],
                            time_in_seq=data.data["test"]['time_in_seq'],
                            time_in_seq_delta_prev=data.data["test"]["time_in_seq_delta_prev"],
                            time_in_seq_delta_prev_bucket=data.data["test"]["time_in_seq_delta_prev_bucket"],
                            single_threaded=single_threaded, flagReturnAttention=True)
        return test_time_preds, test_event_preds, test_attention_scores_blocks

    def predict_eval(self, data, single_threaded=False):
        """Make (time, event) predictions on the test data."""
        val_time_preds, eval_event_preds, eval_loss = self.validation(event_in_seq=data.data["eval"]['event_in_seq'],
                            time_in_seq=data.data["eval"]['time_in_seq'],
                            time_in_seq_delta_prev=data.data["eval"]["time_in_seq_delta_prev"],
                            time_in_seq_delta_prev_bucket=data.data["test"]["time_in_seq_delta_prev_bucket"],
                            event_out_seq=data.data["eval"]['event_out_seq'],
                            time_out_seq=data.data["eval"]['time_out_seq'],
                            single_threaded=single_threaded)
        return val_time_preds, eval_event_preds, eval_loss



    def validation_model_dataset(self, data, single_threaded=False):
        eval_time_preds, eval_event_preds, eval_loss = self.predict_eval(data=data, single_threaded=single_threaded)
        MAE, NDCG, Hit = self.eval(eval_time_preds, data.data["eval"]['time_out_seq'], eval_event_preds,
                  data.data["eval"]['event_out_seq'], time_scale=data.time_scale,
                  K=Config.metrics_K)

        return MAE, NDCG, Hit, eval_loss

    def predict_model_dataset(self, data, single_threaded=False):
        test_time_preds, test_event_preds = self.predict_test(data=data)
        MAE, NDCG, Hit = self.eval(test_time_preds, data.data["test"]['time_out_seq'], test_event_preds,
                                   data.data["test"]['event_out_seq'], time_scale=data.time_scale,
                                   K=Config.metrics_K)

        return MAE, NDCG, Hit


    def predict_test_last(self, data, single_threaded=False):
        """Make (time, event) predictions on the test data."""
        return self.predict_last(event_in_seq=data.data["test"]['event_in_seq'],
                            time_in_seq=data.data["test"]['time_in_seq'],
                            single_threaded=single_threaded)

    def predict_train(self, data, single_threaded=False, batch_size=None):
        """Make (time, event) predictions on the training data."""
        if batch_size == None:
            batch_size = data.data["train"]['event_in_seq'].shape[0]

        return self.predict(event_in_seq=data.data["train"]['event_in_seq'][0:batch_size, :],
                            time_in_seq=data.data["train"]['time_in_seq'][0:batch_size, :],
                            time_in_seq_delta_prev=data.data["train"]["time_in_seq_delta_prev"],
                            time_in_seq_delta_prev_bucket=data.data["train"]["time_in_seq_delta_prev_bucket"],
                            single_threaded=single_threaded)
