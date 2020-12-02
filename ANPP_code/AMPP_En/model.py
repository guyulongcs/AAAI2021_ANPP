from modules import *


class Model():
    def __init__(self, usernum, itemnum, args, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())

        #user
        self.u = tf.placeholder(tf.int32, shape=(None))

        #input seq
        seq_type=["item", "category", "time_ts", "time_ts_prev_delta", "time_ts_next_delta"]
        seq_type_len = len(seq_type)
        self.input_seq = tf.placeholder(tf.int32, shape=(seq_type_len, None, args.max_seq_len))

        #pos, neg
        item_type_list = ["item", "cateogry"]
        item_type_len = len(item_type_list)
        self.pos = tf.placeholder(tf.int32, shape=(item_type_len, None, args.max_seq_len))
        self.neg = tf.placeholder(tf.int32, shape=(item_type_len, None, args.max_seq_len))

        [item_count, cate_count] = itemnum

        test_neg_N=[args.test_neg_N_item, args.test_neg_N_cate]

        #mask
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq[0, :, :], 0)), -1)

        with tf.variable_scope("AMPP_En", reuse=reuse):
            # sequence embedding, item embedding table

            #emb: item, category
            self.seq_emb={}
            self.emb_table = {}

            for i in range(item_type_len):
                self.seq_emb[i], self.emb_table[i] = embedding(self.input_seq[i, :, :],
                                                          vocab_size=itemnum[i],
                                                          num_units=args.hidden_units,
                                                          zero_pad=True,
                                                          scale=True,
                                                          l2_reg=args.l2_emb,
                                                          scope="input_embeddings_%s" % (item_type_list[i]),
                                                          with_t=True,
                                                          reuse=reuse
                                                          )

            # Positional Encoding
            self.t, pos_emb_table = embedding(
                #[B, T]
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[-1]), 0), [tf.shape(self.input_seq)[1], 1]),
                vocab_size=args.max_seq_len,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos",
                reuse=reuse,
                with_t=True
            )

            #input embedding merge


            self.seq = self.seq_emb[0] + self.seq_emb[1] + self.t

            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask

            # Build blocks
            for i in range(args.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):

                    # Self-attention
                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   num_units=args.hidden_units,
                                                   num_heads=args.num_heads,
                                                   dropout_rate=args.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    # Feed forward
                    self.seq = feedforward(normalize(self.seq), num_units=[args.hidden_units, args.hidden_units],
                                           dropout_rate=args.dropout_rate, is_training=self.is_training)
                    self.seq *= mask

            self.seq = normalize(self.seq)

        #seq emb: [BT, H]
        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[1] * args.max_seq_len, args.hidden_units])

        #pos, neg
        pos={}
        neg = {}
        pos_emb = {}
        neg_emb = {}

        # [B, 1+Neg]
        self.test_item_list = tf.placeholder(tf.int32, shape=(None, 1 + test_neg_N[0]))
        self.test_cate_list = tf.placeholder(tf.int32, shape=(None, 1 + test_neg_N[1]))

        # [B, 1+Neg, H]
        self.test_list_emb = [None, None]
        self.test_list_emb[0] = tf.nn.embedding_lookup(self.emb_table[0], self.test_item_list)
        self.test_list_emb[1] = tf.nn.embedding_lookup(self.emb_table[1], self.test_cate_list)

        seq_emb_tile = {}
        self.test_logits = {}

        self.pos_logits = {}
        self.neg_logits = {}

        self.auc = {}

        istarget = {}

        loss_arr = {}
        for i in range(2):
            #pos[i]: [BT]
            pos[i] = tf.reshape(self.pos[i, :, :], [tf.shape(self.input_seq)[1] * args.max_seq_len])
            neg[i] = tf.reshape(self.neg[i, :, :], [tf.shape(self.input_seq)[1] * args.max_seq_len])

            #pos_emb[i]: [BT, H]
            pos_emb[i] = tf.nn.embedding_lookup(self.emb_table[i], pos[i])
            neg_emb[i] = tf.nn.embedding_lookup(self.emb_table[i], neg[i])




            #[B, 1+Neg, H]: same as last H

            #seq_emb_T:[B, T, H]
            seq_emb_T = tf.reshape(seq_emb, [tf.shape(self.input_seq)[1], args.max_seq_len, args.hidden_units])

            #seq_emb_last:[B, H]
            seq_emb_last = seq_emb_T[:, -1, :]

            #seq_emb_tile: [B, 1+Neg, H]
            seq_emb_tile[i] = tf.reshape(tf.tile(seq_emb_last, [1, 1 + test_neg_N[i]]), [tf.shape(self.input_seq)[1], 1 + test_neg_N[i], args.hidden_units])

            #test_logits: [B, 1+Neg]
            self.test_logits[i] = tf.reduce_sum(seq_emb_tile[i] * self.test_list_emb[i], -1)

            # prediction layer
            #pos_logits[i]: [BT]
            self.pos_logits[i] = tf.reduce_sum(pos_emb[i] * seq_emb, -1)
            self.neg_logits[i] = tf.reduce_sum(neg_emb[i] * seq_emb, -1)

            # ignore padding items (0): [BT]
            istarget[i] = tf.reshape(tf.to_float(tf.not_equal(pos[i], 0)), [tf.shape(self.input_seq)[1] * args.max_seq_len])

            loss_arr[i] = tf.reduce_sum(
                - tf.log(tf.sigmoid(self.pos_logits[i]) + 1e-24) * istarget[i] -
                tf.log(1 - tf.sigmoid(self.neg_logits[i]) + 1e-24) * istarget[i]
        ) / tf.reduce_sum(istarget[i])
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        #loss
        self.loss = (args.weight_loss[0]) * loss_arr[0] + (args.weight_loss[1]) * loss_arr[1] + (args.weight_loss[2]) * sum(reg_losses)
        tf.summary.scalar('loss', self.loss)

        for i in range(2):
            self.auc[i] = tf.reduce_sum(
                ((tf.sign(self.pos_logits[i] - self.neg_logits[i]) + 1) / 2) * istarget[i]
                ) / tf.reduce_sum(istarget[i])

        if reuse is None:
            tf.summary.scalar('auc_item', self.auc[0])
            tf.summary.scalar('auc_category', self.auc[1])
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.global_epoch_step = \
                tf.Variable(0, trainable=False, name='global_epoch_step')
            self.global_epoch_step_op = \
                tf.assign(self.global_epoch_step, self.global_epoch_step + 1)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc_item', self.auc[0])
            tf.summary.scalar('test_auc_category', self.auc[1])

        self.merged = tf.summary.merge_all()

    def predict(self, sess, user_list, seq_list, test_item_list, test_cate_list):
        res = sess.run(self.test_logits,
                        {self.u: user_list, self.input_seq: seq_list, self.test_item_list: test_item_list, self.test_cate_list: test_cate_list, self.is_training: False})

        #print(">>test_logits:", res.shape)
        return res
