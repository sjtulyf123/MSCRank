import random
import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib.rnn import GRUCell, static_bidirectional_rnn, LSTMCell, MultiRNNCell
from tensorflow.python.ops.rnn import dynamic_rnn, bidirectional_dynamic_rnn
from rnn_version.rnn_layer import NewRNNCell
import copy
import time

class TwoLevelRNN(object):
    def __init__(self, name_map, feature_size, emb_dim, hidden_size, max_seq_len, cat_rnn='gru', item_rnn='gru',
                 interaction_mode='one_to_one', multi_task=False, use_cat2nd=True, use_cat3rd=True,
                 optimize_ratio=False, use_regular=False, use_new_rnn=False, use_initlist_attention=False, use_js=False,
                 ideal_ratio=1.0,kl_weight=1, sample_num=5, ranker_weight=1,behavior_length=12,soft_ranker_tau=1.0):
        tf.reset_default_graph()

        self.name_map = name_map
        self.sample_num = sample_num
        self.use_multi_task = multi_task
        self.behavior_length = behavior_length
        self.use_grad_norm = False
        self.gradnorm_alpha = 0.1
        self.use_new_rnn = use_new_rnn
        self.use_initlist_attention = use_initlist_attention
        self.ideal_ratio = ideal_ratio
        self.ad_occupancy, self.share_occupancy, self.nature_occupancy = 1./float(3.), 1./float(3.), 1./float(3.)
        self.tau = soft_ranker_tau
        self.use_js = use_js

        with tf.name_scope("inputs"):
            # for online inference and training
            self.seq_feat_ph = tf.placeholder(tf.float32, [None, max_seq_len, 27],
                                              name='seq_feat_ph')
            self.seq_len_ph = tf.placeholder(tf.int32, [None, ], name='seq_length_ph')
            self.init_list_ph = tf.placeholder(tf.float32, [None, 3,max_seq_len, 27 + 1],
                                                 name='init_list_ph')
            self.keep_prob = tf.placeholder(tf.float32, [1, ], name='keep_prob')

            # for multi_task
            if self.use_multi_task:
                self.pos_pair_pos_ph = tf.placeholder(tf.int32, [None, self.sample_num, 2], name='pos_pair_pos_ph')
                self.neg_pair_pos_ph = tf.placeholder(tf.int32, [None, self.sample_num, 2], name='neg_pair_pos_ph')
                self.ranker_label_ph = tf.placeholder(tf.float32, [None, max_seq_len], name='ranker_label_ph')

            # for training
            self.label_ph = tf.placeholder(tf.int32, [None, max_seq_len, 4], name='true_labels')
            self.lr = tf.placeholder(tf.float32, [], name='lr')
            self.reg_lambda = tf.placeholder(tf.float32, [], name='reg_lambda')

        # slice from seq_feat_ph
        self.cat1st_ph = tf.to_int32(self.seq_feat_ph[:, :, 4])

        # item
        self.item_id_ph = tf.to_int32(self.seq_feat_ph[:, :, 3])
        self.item_gender_ph = tf.to_int32(self.seq_feat_ph[:, :, 5])
        self.item_buy_ph = tf.to_int32(self.seq_feat_ph[:, :, 6])
        self.item_bc_ph = tf.to_int32(self.seq_feat_ph[:, :, 7])
        self.item_dense_ph = self.seq_feat_ph[:, :, 8:]

        # user
        self.user_age_ph = tf.to_int32(self.seq_feat_ph[:, 0, 0])
        self.use_gender_ph = tf.to_int32(self.seq_feat_ph[:, 0, 1])
        self.user_buy_ph = tf.to_int32(self.seq_feat_ph[:, 0, 2])

        self.max_seq_len = max_seq_len
        self.use_cat2nd = use_cat2nd
        self.use_cat3rd = use_cat3rd
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.optimize_ratio = optimize_ratio
        self.use_regular = use_regular

        if self.use_new_rnn:
            self.ad_list_ph = self.init_list_ph[:, 0, :,:]
            self.share_list_ph = self.init_list_ph[:, 1, :, :]
            self.nature_list_ph = self.init_list_ph[:, 2, :, :]

        self.original_weight = 1.0
        self.kl_weight = kl_weight
        self.ranker_weight = ranker_weight

        self.bid = self.seq_feat_ph[:, :, 17]

        # build embedding
        with tf.name_scope('embedding'):
            self.emb_mtx = [tf.get_variable('overall_emb_mtx_{}'.format(i), [self.feature_size[i], self.emb_dim],
                                           initializer=tf.truncated_normal_initializer) for i in range(len(feature_size))]

            cat1_emb = tf.nn.embedding_lookup(self.emb_mtx[4], self.cat1st_ph)
            self.cat_emb = cat1_emb

            # item embedding
            item_id_emb = tf.nn.embedding_lookup(self.emb_mtx[3], self.item_id_ph)
            item_gender_emb = tf.nn.embedding_lookup(self.emb_mtx[5], self.item_id_ph)
            item_buy_emb = tf.nn.embedding_lookup(self.emb_mtx[6], self.item_id_ph)
            item_bc_emb = tf.nn.embedding_lookup(self.emb_mtx[7], self.item_id_ph)
            self.item_emb = tf.concat([item_id_emb, item_gender_emb, item_buy_emb, item_bc_emb, self.item_dense_ph], axis=-1)

            # user specified embedding
            user_age_emb = tf.nn.embedding_lookup(self.emb_mtx[0], self.user_age_ph)
            user_gender_emb = tf.nn.embedding_lookup(self.emb_mtx[1], self.use_gender_ph)
            user_buy_emb = tf.nn.embedding_lookup(self.emb_mtx[2], self.user_buy_ph)
            self.user_emb = tf.concat([user_age_emb, user_gender_emb, user_buy_emb], axis=-1)


        if self.use_new_rnn:
            # build init list input
            self.length_emb_mtx = tf.get_variable('initlist_length_emb_mtx', [30, self.emb_dim],
                                                  initializer=tf.truncated_normal_initializer)
            init_list_concat = tf.concat([self.ad_list_ph, self.share_list_ph, self.nature_list_ph], axis=1)
            with tf.name_scope('new_rnn_embedding'):
                list_cat_emb = tf.nn.embedding_lookup(self.emb_mtx[4],
                                                      tf.to_int32(init_list_concat[:, :, 4]),
                                                      name='list_cat_emb')
                list_item_id_emb = tf.nn.embedding_lookup(self.emb_mtx[3], tf.to_int32(init_list_concat[:, :, 3]),
                                                          name='list_id_emb')
                list_length_emb = tf.nn.embedding_lookup(self.length_emb_mtx,
                                                         tf.to_int32(init_list_concat[:, :, 27]),
                                                         name='list_length_emb')
                list_value = tf.ones_like(init_list_concat[:,:,4],dtype=tf.float32)
                list_value = tf.expand_dims(list_value, axis=-1)
                list_value_emb = tf.layers.dense(list_value,self.emb_dim,activation=tf.nn.tanh)
                list_concat_emb = tf.concat([list_cat_emb, list_item_id_emb, list_length_emb,list_value_emb], axis=-1,
                                            name='list_concat_emb')
                ad_list_emb = list_concat_emb[:, 0:max_seq_len, :]
                share_list_emb = list_concat_emb[:, max_seq_len:2 * max_seq_len, :]
                nature_list_emb = list_concat_emb[:, 2 * max_seq_len:3 * max_seq_len, :]
                ad_list_length = list_length_emb[:,0,-1]
                share_list_length = list_length_emb[:, max_seq_len, -1]
                nature_list_length = list_length_emb[:, 2*max_seq_len, -1]

            # build l1,l2,l3 based on position weighting
            if self.use_initlist_attention:
                l_ad = self.multihead_window_attention(ad_list_emb,ad_list_emb,ad_list_length,half_window_size=5,num_heads=1,scope="ad_attention")
                l_share = self.multihead_window_attention(share_list_emb,share_list_emb,share_list_length,half_window_size=5,num_heads=1,scope="share_attention")
                l_nature = self.multihead_window_attention(nature_list_emb,nature_list_emb,nature_list_length,half_window_size=5,num_heads=1,scope="nature_attention")
            else:
                l_ad = tf.divide(tf.reduce_sum(ad_list_emb, axis=1, keepdims=True),tf.maximum(tf.reshape(ad_list_length,[-1,1,1]),1))
                l_share = tf.divide(tf.reduce_sum(share_list_emb, axis=1, keepdims=True),tf.maximum(tf.reshape(share_list_length,[-1,1,1]),1))
                l_nature = tf.divide(tf.reduce_sum(nature_list_emb, axis=1, keepdims=True),tf.maximum(tf.reshape(nature_list_length,[-1,1,1]),1))
                l_ad = tf.tile(l_ad, [1, self.max_seq_len, 1], name='l_ad')
                l_share = tf.tile(l_share, [1, self.max_seq_len, 1], name='l_share')
                l_nature = tf.tile(l_nature, [1, self.max_seq_len, 1], name='l_nature')

        # build category level rnn
        with tf.name_scope('category_rnn'):
            if self.use_new_rnn:
                xt_depth = self.cat_emb.shape[-1]
                list_depth = l_nature.shape[-1]
                cat_rnncell = NewRNNCell(hidden_size, xt_depth, list_depth, 3)
                self.cat_seq_ht, self.cat_seq_final_state = dynamic_rnn(cat_rnncell, inputs=tf.concat(
                    [self.cat_emb, l_ad, l_share, l_nature], axis=-1), sequence_length=self.seq_len_ph,
                                                                        dtype=tf.float32,
                                                                        scope='crnn')
            else:
                assert cat_rnn in ['gru', 'lstm'], "RNN should be gru or lstm!"
                if cat_rnn == 'gru':
                    cat_rnncell = GRUCell(hidden_size)
                elif cat_rnn == 'lstm':
                    cat_rnncell = LSTMCell(hidden_size)
                self.cat_seq_ht, self.cat_seq_final_state = dynamic_rnn(cat_rnncell, inputs=self.cat_emb,
                                                                        sequence_length=self.seq_len_ph,
                                                                        dtype=tf.float32,
                                                                        scope='crnn')

        # build item level rnn
        with tf.name_scope('item_rnn'):
            assert cat_rnn in ['gru', 'lstm'], "RNN should be gru or lstm!"
            if item_rnn == 'gru':
                item_rnncell = GRUCell(hidden_size)
            else:
                item_rnncell = LSTMCell(hidden_size)
            self.item_seq_ht, self.item_seq_final_state = dynamic_rnn(item_rnncell, inputs=self.item_emb,
                                                                      sequence_length=self.seq_len_ph,
                                                                      dtype=tf.float32,
                                                                      scope='irnn')

        self.build_interaction(interaction_mode=interaction_mode)
        self.build_loss()

    def build_interaction(self, interaction_mode='one_to_one'):
        # conduct interaction between category level and item level output
        assert interaction_mode in ['one_to_one', 'one_to_all', 'convolution']
        if interaction_mode == 'one_to_one':
            interact_res = tf.multiply(self.cat_seq_ht, self.item_seq_ht)
        elif interaction_mode == 'one_to_all':
            cat_sum = tf.reduce_sum(self.cat_seq_ht, axis=1, keep_dims=True)  # batch_size,1,feature_dim
            cat_sum = tf.tile(cat_sum, multiples=[1, self.max_seq_len, 1])  # batch_size, max_seq_len, feature_dim
            item_sum = tf.reduce_sum(self.item_seq_ht, axis=1, keep_dims=True)  # batch_size,1,feature_dim
            item_sum = tf.tile(item_sum, multiples=[1, self.max_seq_len, 1])  # batch_size, max_seq_len, feature_dim

            interact_res = tf.add(tf.multiply(self.cat_seq_ht, item_sum), tf.multiply(self.item_seq_ht, cat_sum))
        else:
            # convolution
            cat_conv = tf.layers.conv1d(self.cat_seq_ht, filters=self.emb_dim, kernel_size=3, strides=1, padding='same')
            item_conv = tf.layers.conv1d(self.item_seq_ht, filters=self.emb_dim, kernel_size=3, strides=1,
                                         padding='same')
            interact_res = tf.multiply(cat_conv, item_conv)

        # concat user specified info and add MLP
        bn1 = tf.layers.batch_normalization(inputs=interact_res, name='bn1')
        interact_out = tf.layers.dense(bn1, units=self.hidden_size, activation=tf.nn.tanh, name='fc1')
        interact_out = tf.nn.dropout(interact_out, keep_prob=self.keep_prob[0])

        # todo: normalize user and context embedding?
        user_emb = tf.expand_dims(self.user_emb, 1)
        user_emb = tf.tile(user_emb, [1, self.max_seq_len, 1])
        user_emb = tf.layers.batch_normalization(inputs=user_emb, name='user_bn')
        user_emb = tf.layers.dense(user_emb, units=self.hidden_size, activation=tf.nn.tanh, name='user_fc')

        self.shared_out = tf.concat([interact_out, user_emb], axis=-1, name='concat_out')

        seq_mask = tf.sequence_mask(self.seq_len_ph, maxlen=self.max_seq_len, dtype=tf.float32)
        self.seq_mask = tf.expand_dims(seq_mask, axis=2)
        no_mask_logit = tf.layers.dense(self.shared_out, units=4, activation=None, name='fc_out')
        labels = tf.to_float(self.label_ph)

        if self.use_multi_task:
            hidden1 = tf.layers.dense(self.shared_out, units=self.hidden_size, activation=tf.nn.relu, name='ranker_fc1')
            ranker_score = tf.layers.dense(hidden1, units=2, activation=None,
                                           name='ranker_fc2')  # (batch_size, max_len, 1)

            score = tf.nn.softmax(ranker_score)
            self.reshaped_score = tf.reshape(score[:, :, 0], [-1, self.max_seq_len], name='ranker_score')
            seq_mask = tf.sequence_mask(self.seq_len_ph, maxlen=self.max_seq_len, dtype=tf.float32)
            self.masked_score = self.reshaped_score * seq_mask

            sorted_pos = tf.argsort(self.masked_score * (self.bid), axis=-1, direction='DESCENDING',
                                    stable=True)
            _no_mask_logit = tf.batch_gather(no_mask_logit, sorted_pos)
            _labels = tf.batch_gather(labels, sorted_pos)
            no_mask_logit = (1.0-self.tau)*no_mask_logit+self.tau*_no_mask_logit
            labels = (1.0-self.tau)*labels+self.tau*_labels

        self.logits = tf.multiply(no_mask_logit, self.seq_mask, name='logits')
        _pred = tf.argmax(self.logits[:, :, :3], dimension=2)
        self.pred = tf.to_float(_pred, name='pred')

        mask_target = tf.multiply(labels, self.seq_mask)
        self.target = tf.reshape(mask_target, [-1, 4])
        self.reshape_logits = tf.reshape(self.logits, [-1, 4])

    def build_loss(self):

        self.original_loss = tf.losses.softmax_cross_entropy(self.target, self.reshape_logits)
        if self.optimize_ratio == False:
            self.kl_loss = tf.constant(0.0)
        else:
            if self.use_js:
                self.kl_loss = self.cal_js(self.logits)
            else:
                self.kl_loss = self.cal_kl(self.logits)

        if self.use_multi_task == False:
            self.ranker_loss = tf.constant(0.0)
        else:
            self.ranker_loss = self.cal_ranker_loss(self.masked_score, self.pos_pair_pos_ph, self.neg_pair_pos_ph)

        self.loss = self.original_weight * self.original_loss + self.kl_weight * self.kl_loss + self.ranker_weight * self.ranker_loss


        if self.use_regular:
            for v in tf.trainable_variables():
                if 'bias' not in v.name and 'emb' not in v.name:
                    self.loss += self.reg_lambda * tf.nn.l2_loss(v)
        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

        self.gradnorm_loss = tf.constant(0.0)
        self.update_op = [self.original_weight, self.kl_weight, self.ranker_weight]

    def train(self, sess, padding_batch_data, seq_len, padding_init_list,lr, reg_lambda, keep_prob=0.8):

        padding_batch_data[np.isnan(padding_batch_data)] = 0.0
        padding_init_list[np.isnan(padding_init_list)]=0.0

        original_label = padding_batch_data[:, :, -1].tolist()
        cat = padding_batch_data[:, :, 5].tolist()
        feat_col = list(range(1, 28))
        seq_feat_data = padding_batch_data[:, :, feat_col]
        seq_feat_init_list = padding_init_list[:, :, :,list(range(1, 28))+[29]]

        # BCE loss 4*n label
        modified_label = np.zeros((len(original_label), len(original_label[0]), 4), dtype=np.float32)
        for bidx in range(len(original_label)):
            for iidx in range(len(original_label[0])):
                if int(original_label[bidx][iidx]) == 1:
                    cur_cat = int(cat[bidx][iidx])
                    modified_label[bidx, iidx, cur_cat] = 1.0
                else:
                    modified_label[bidx, iidx, 3] = 1.0
        ranker_loss = 0.0
        if self.use_multi_task:
            value = padding_batch_data[:, :, -1]
            loss, kl_loss, ranker_loss, _ = sess.run(
                [self.original_loss, self.kl_loss, self.ranker_loss, self.train_step],
                feed_dict={
                    self.seq_feat_ph: seq_feat_data,
                    self.label_ph: modified_label,
                    self.seq_len_ph: seq_len,
                    self.init_list_ph: seq_feat_init_list,
                    self.ranker_label_ph: value,
                    self.lr: lr,
                    self.reg_lambda: reg_lambda,
                    self.keep_prob: [keep_prob]
                })

        else:
            loss, kl_loss, _ = sess.run([self.original_loss, self.kl_loss, self.train_step],
                                        feed_dict={
                                            self.seq_feat_ph: seq_feat_data,
                                            self.label_ph: modified_label,
                                            self.seq_len_ph: seq_len,
                                            self.init_list_ph: seq_feat_init_list,
                                            self.lr: lr,
                                            self.reg_lambda: reg_lambda,
                                            self.keep_prob: [keep_prob]
                                        })
        return loss, kl_loss, ranker_loss

    def eval(self, sess, padding_batch_data, seq_len, padding_init_list,reg_lambda, keep_prob=1):
        padding_batch_data[np.isnan(padding_batch_data)] = 0.0
        padding_init_list[np.isnan(padding_init_list)] = 0.0

        original_label = padding_batch_data[:, :, -1].tolist()
        cat = padding_batch_data[:, :, 5].tolist()
        feat_col = list(range(1, 28))
        seq_feat_data = padding_batch_data[:, :, feat_col]
        seq_feat_init_list = padding_init_list[:, :, :,list(range(1, 28))+[29]]

        # BCE loss 4*n label
        modified_label = np.zeros((len(original_label), len(original_label[0]), 4), dtype=np.float32)
        for bidx in range(len(original_label)):
            for iidx in range(len(original_label[0])):
                if int(original_label[bidx][iidx]) == 1:
                    cur_cat = int(cat[bidx][iidx])
                    modified_label[bidx, iidx, cur_cat] = 1.0
                else:
                    modified_label[bidx, iidx, 3] = 1.0
        ranker_loss = 0.0
        if self.use_multi_task:
            value = padding_batch_data[:, :, -1]
            pred, label, loss, kl_loss, ranker_loss = sess.run(
                [self.pred, self.label_ph, self.original_loss, self.kl_loss, self.ranker_loss], feed_dict={
                    self.seq_feat_ph: seq_feat_data,
                    self.label_ph: modified_label,
                    self.seq_len_ph: seq_len,
                    self.init_list_ph: seq_feat_init_list,
                    self.ranker_label_ph: value,
                    self.reg_lambda: reg_lambda,
                    self.keep_prob: [keep_prob]
                })
        else:
            pred, label, loss, kl_loss = sess.run(
                [self.pred, self.label_ph, self.original_loss, self.kl_loss], feed_dict={
                    self.seq_feat_ph: seq_feat_data,
                    self.label_ph: modified_label,
                    self.seq_len_ph: seq_len,
                    self.init_list_ph: seq_feat_init_list,
                    self.reg_lambda: reg_lambda,
                    self.keep_prob: [keep_prob]
                })

        return pred, modified_label.tolist(), loss, kl_loss, ranker_loss

    def cal_kl(self, output_logits):
        reshaped_logits = tf.slice(tf.reshape(output_logits, [-1, 4]), begin=[0, 0], size=[-1, 3])
        p = tf.nn.softmax(reshaped_logits)
        y_hard = tf.cast(tf.equal(p, tf.reduce_max(p, 1, keep_dims=True)), p.dtype)
        gumbel_logits = tf.stop_gradient(y_hard - p) + p

        seq_mask = tf.sequence_mask(self.seq_len_ph, maxlen=self.max_seq_len, dtype=tf.float32)
        seq_mask = tf.reshape(seq_mask, [-1, 1])

        dist = gumbel_logits * seq_mask

        dist = tf.reshape(dist, [-1, 30, 3])
        dist1 = tf.reshape(dist[:, :10, :], [-1, 3])
        sum_p = tf.reduce_sum(dist1, axis=0)
        self.sump = sum_p
        valid_num = sum_p[0] + sum_p[1] + sum_p[2]

        kl = sum_p[0] / valid_num * tf.log(sum_p[0] / (self.ad_occupancy * valid_num)) + sum_p[1] / valid_num * tf.log(
            sum_p[1] / (self.share_occupancy * valid_num)) + sum_p[2] / valid_num * tf.log(
            sum_p[2] / (self.nature_occupancy * valid_num))

        dist2 = tf.reshape(dist[:, 10:20, :], [-1, 3])
        sum_p = tf.reduce_sum(dist2, axis=0)
        valid_num = sum_p[0] + sum_p[1] + sum_p[2]

        kl += sum_p[0] / valid_num * tf.log(sum_p[0] / (self.ad_occupancy * valid_num)) + sum_p[1] / valid_num * tf.log(
            sum_p[1] / (self.share_occupancy * valid_num)) + sum_p[2] / valid_num * tf.log(
            sum_p[2] / (self.nature_occupancy * valid_num))

        dist3 = tf.reshape(dist[:, 20:30, :], [-1, 3])
        sum_p = tf.reduce_sum(dist3, axis=0)
        valid_num = sum_p[0] + sum_p[1] + sum_p[2]

        kl += sum_p[0] / valid_num * tf.log(sum_p[0] / (self.ad_occupancy * valid_num)) + sum_p[1] / valid_num * tf.log(
            sum_p[1] / (self.share_occupancy * valid_num)) + sum_p[2] / valid_num * tf.log(
            sum_p[2] / (self.nature_occupancy * valid_num))

        kl = kl / 3
        return kl

    def cal_ranker_loss(self, score, pos_pair, neg_pair):
        logloss = tf.losses.log_loss(self.ranker_label_ph, score)
        return logloss

    def cal_js(self, output_logits):
        reshaped_logits = tf.slice(tf.reshape(output_logits, [-1, 4]), begin=[0, 0], size=[-1, 3])
        p = tf.nn.softmax(reshaped_logits)
        y_hard = tf.cast(tf.equal(p, tf.reduce_max(p, 1, keep_dims=True)), p.dtype)
        gumbel_logits = tf.stop_gradient(y_hard - p) + p

        seq_mask = tf.sequence_mask(self.seq_len_ph, maxlen=self.max_seq_len, dtype=tf.float32)
        seq_mask = tf.reshape(seq_mask, [-1, 1])

        dist = gumbel_logits * seq_mask

        dist = tf.reshape(dist, [-1, 30, 3])
        dist1 = tf.reshape(dist[:, :10, :], [-1, 3])
        sum_p = tf.reduce_sum(dist1, axis=0)
        self.sump = sum_p
        valid_num = sum_p[0] + sum_p[1]

        left_kl = sum_p[0] / valid_num * tf.log(sum_p[0] / ((self.ad_occupancy+sum_p[0]/valid_num)/2 * valid_num)) + sum_p[1] / valid_num * tf.log(
            sum_p[1] / ((self.share_occupancy+sum_p[1]/valid_num)/2 * valid_num)) + sum_p[2] / valid_num * tf.log(
            sum_p[2] / ((self.nature_occupancy+sum_p[2]/valid_num)/2 * valid_num))
        right_kl = self.ad_occupancy * tf.log(self.ad_occupancy/((self.ad_occupancy+sum_p[0]/valid_num)/2)) + self.share_occupancy * tf.log(
             self.share_occupancy/((self.share_occupancy+sum_p[1]/valid_num)/2))+ self.nature_occupancy * tf.log(
             self.nature_occupancy/((self.nature_occupancy+sum_p[2]/valid_num)/2))
        js = 0.5*left_kl+0.5*right_kl

        dist2 = tf.reshape(dist[:, 10:20, :], [-1, 3])
        sum_p = tf.reduce_sum(dist2, axis=0)
        valid_num = sum_p[0] + sum_p[1]

        left_kl = sum_p[0] / valid_num * tf.log(sum_p[0] / ((self.ad_occupancy+sum_p[0]/valid_num)/2 * valid_num)) + sum_p[1] / valid_num * tf.log(
            sum_p[1] / ((self.share_occupancy+sum_p[1]/valid_num)/2 * valid_num)) + sum_p[2] / valid_num * tf.log(
            sum_p[2] / ((self.nature_occupancy+sum_p[2]/valid_num)/2 * valid_num))
        right_kl = self.ad_occupancy * tf.log(self.ad_occupancy/((self.ad_occupancy+sum_p[0]/valid_num)/2)) + self.share_occupancy * tf.log(
             self.share_occupancy/((self.share_occupancy+sum_p[1]/valid_num)/2))+ self.nature_occupancy * tf.log(
             self.nature_occupancy/((self.nature_occupancy+sum_p[2]/valid_num)/2))
        js = js+0.5*left_kl+0.5*right_kl

        dist3 = tf.reshape(dist[:, 20:30, :], [-1, 3])
        sum_p = tf.reduce_sum(dist3, axis=0)
        valid_num = sum_p[0] + sum_p[1]

        left_kl = sum_p[0] / valid_num * tf.log(sum_p[0] / ((self.ad_occupancy+sum_p[0]/valid_num)/2 * valid_num)) + sum_p[1] / valid_num * tf.log(
            sum_p[1] / ((self.share_occupancy+sum_p[1]/valid_num)/2 * valid_num)) + sum_p[2] / valid_num * tf.log(
            sum_p[2] / ((self.nature_occupancy+sum_p[2]/valid_num)/2 * valid_num))
        right_kl = self.ad_occupancy * tf.log(self.ad_occupancy/((self.ad_occupancy+sum_p[0]/valid_num)/2)) + self.share_occupancy * tf.log(
             self.share_occupancy/((self.share_occupancy+sum_p[1]/valid_num)/2))+ self.nature_occupancy * tf.log(
             self.nature_occupancy/((self.nature_occupancy+sum_p[2]/valid_num)/2))
        js = js+0.5*left_kl+0.5*right_kl

        js = js / 3
        return js

    def multihead_window_attention(self,
                            queries,
                            keys,
                            seq_real_length,
                            half_window_size=2,
                            num_units=None,
                            num_heads=1,
                            scope="multihead_attention",
                            reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            if num_units is None:
                num_units = queries.get_shape().as_list()[-1]

            Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            key_masks = tf.to_int32(tf.sequence_mask(seq_real_length,self.max_seq_len))
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)
            if half_window_size>0:
                # use window attention
                window_masks = np.zeros((1,self.max_seq_len,self.max_seq_len))
                for idx in range(self.max_seq_len):
                    window_masks[:,idx,max(0,idx-half_window_size):min(1+self.max_seq_len,1+idx+half_window_size)]=1
                tf_window_masks = tf.convert_to_tensor(window_masks,dtype=tf.int32)
                tf_window_masks = tf.tile(tf_window_masks,[tf.shape(key_masks)[0],1,1])
                outputs = tf.where(tf.equal(tf_window_masks, 0), paddings, outputs)
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

            query_masks = tf.to_float(tf.sequence_mask(seq_real_length, self.max_seq_len))
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)
            outputs = tf.nn.dropout(outputs, self.keep_prob[0])
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        return outputs

    def save_pb(self, sess, to_path, name_list):
        """
        :param sess: run session
        :param to_path: .pb file
        :param name_list: out name, should be 'pred' in thi session
        :return: none
        """
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            name_list)

        with tf.gfile.GFile(to_path, "wb") as f:
            f.write(constant_graph.SerializeToString())
        print("save {} ops within graph.".format(len(constant_graph.node)))
