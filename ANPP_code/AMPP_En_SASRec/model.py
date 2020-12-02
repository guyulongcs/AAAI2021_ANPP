from modules import *


class Model():
    def __init__(self, usernum, itemnum, args, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.max_seq_len))
        self.pos = tf.placeholder(tf.int32, shape=(None, args.max_seq_len))
        self.neg = tf.placeholder(tf.int32, shape=(None, args.max_seq_len))
        pos = self.pos
        neg = self.neg
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)

        with tf.variable_scope("AMPP_En_SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )

            # Positional Encoding
            t, pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=args.max_seq_len,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos",
                reuse=reuse,
                with_t=True
            )
            self.seq += t

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

        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.max_seq_len])
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.max_seq_len])
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)
        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.max_seq_len, args.hidden_units])

        #[B, 1+Neg]

        self.test_item = tf.placeholder(tf.int32, shape=(None, 1 + args.test_neg_N_item))
        #[B, 1+Neg, H]
        test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)
        #test_item_emb = tf.reshape(test_item_emb , [args.hidden_units, -1])

        #[B, 1+Neg, H]: same as last H
        seq_emb_T = tf.reshape(seq_emb, [tf.shape(self.input_seq)[0], args.max_seq_len, args.hidden_units])
        seq_emb_last = seq_emb_T[:, -1, :]
        seq_emb_tile = tf.reshape(tf.tile(seq_emb_last, [1, 1 + args.test_neg_N_item]), [tf.shape(self.input_seq)[0], 1 + args.test_neg_N_item, args.hidden_units])

        #test_logits: [B, 1+Neg]
        self.test_logits = tf.reduce_sum(seq_emb_tile * test_item_emb, -1)

        # prediction layer
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        # ignore padding items (0)
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.max_seq_len])
        self.loss = tf.reduce_sum(
            - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
            tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)

        tf.summary.scalar('loss', self.loss)
        self.auc = tf.reduce_sum(
            ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)

        if reuse is None:
            tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.global_epoch_step = \
                tf.Variable(0, trainable=False, name='global_epoch_step')
            self.global_epoch_step_op = \
                tf.assign(self.global_epoch_step, self.global_epoch_step + 1)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, user_list, seqItem_list, posItem_list, negItem_list):
        test_item_list = np.concatenate((posItem_list, negItem_list),axis=-1)
        #print("test_item_list:", test_item_list)
        res = sess.run(self.test_logits,
                        {self.u: user_list, self.input_seq: seqItem_list, self.test_item: test_item_list, self.is_training: False})

        #print(">>test_logits:", res.shape)
        return res
