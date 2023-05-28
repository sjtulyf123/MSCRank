# -*- coding:utf-8 -*-
import os
from time import time
import random
import numpy as np
import glob
import argparse
from tensorflow.contrib.rnn import GRUCell, static_bidirectional_rnn, LSTMCell, MultiRNNCell
from tensorflow.python.ops.rnn import dynamic_rnn, bidirectional_dynamic_rnn
from collections import deque


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

try:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.ERROR)
except ImportError:
    import tensorflow as tf

    tf.logging.set_verbosity(tf.logging.ERROR)
finally:
    pass

DEFAULT_MODEL_NAME = "category_ranker"


def _calc_ad_freq(target_ratio):
    y = 1. / (1. + target_ratio)
    return 1. - y


class DQNAgent(object):
    def __init__(self, paras):
        print("tf.__version__: ", tf.__version__)
        self.paras = paras
        tf.random.set_random_seed(paras.get("seed", 3407))
        graph = tf.Graph()
        self.graph = graph
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config, graph=graph)

        self.buff = deque(maxlen=paras["replay_size"])
        self.threshold = 10000

        self.state_dim = 28
        self.rnn_time_step = 10
        self.gru_dim = 128
        self.action_dim = 3
        self.gamma = paras["gamma"]
        self.learning_rate = paras["learning_rate"]
        self.batch_size = paras["batch_size"]
        self.hidden_layers = paras["hidden_layers"]
        self.epsilon_start, self.epsilon_end = paras["epsilon"]
        self.reg_lambda = paras.get("l2_reg_lambda", 0.00005)
        self.kl_weight = paras["kl_weight"]
        self.t_ratio = 1. / float(3)
        self.epsilon = self.epsilon_start
        self.tau = paras.get("tau", 0.005)

        self.feature_size = paras["feature_size"]
        self.emb_dim = paras.get("embedding_dim", 64)
        self.behavior_feat_size = paras["behavior_feat_size"]

        self.use_double_q = paras["use_double_q"]

        self.out_graph_path = paras.get("out_graph_path")
        self.train_cnt = 0

        self.kl_record = deque(maxlen=100)
        self.target_q_record = deque(maxlen=5)
        self.freq_record = deque(maxlen=5)
        self.norm_freq_record = deque(maxlen=5)

        with self.graph.as_default():
            with tf.variable_scope("inputs"):
                self.state = tf.placeholder(
                    "float", [None, 2, self.rnn_time_step, self.state_dim], name="seq_feat_ph"
                )
                self.next_state = tf.placeholder(
                    "float", [None, 2, self.rnn_time_step, self.state_dim], name="next_seq_feat_ph"
                )
                self.action = tf.placeholder(tf.int32, [None, ], name="action")
                self.reward = tf.placeholder(tf.float32, [None, ], name="reward")
                self.done = tf.placeholder(tf.bool, [None, ], name="done")
                self.seq_len = tf.placeholder(tf.int32, [None, ], name="seq_length_ph")
                self.keep_prob = tf.placeholder(tf.float32, [1], name='keep_prob')
                self.state_len = tf.placeholder(tf.int32, [None, 4], name="state_length_ph")

            # cur state
            # slice from seq_feat_ph
            self.cat1st_ph = tf.to_int32(self.state[:, :, :, 4])

            # item
            self.item_id_ph = tf.to_int32(self.state[:, :, :, 3])
            self.item_gender_ph = tf.to_int32(self.state[:, :, :, 5])
            self.item_buy_ph = tf.to_int32(self.state[:, :, :, 6])
            self.item_bc_ph = tf.to_int32(self.state[:, :, :, 7])
            self.item_dense_ph = self.state[:, :, :, 8:-1]

            # user
            self.user_age_ph = tf.to_int32(self.state[:, :, 0, 0])
            self.use_gender_ph = tf.to_int32(self.state[:, :, 0, 1])
            self.user_buy_ph = tf.to_int32(self.state[:, :, 0, 2])

            # next state
            # slice from seq_feat_ph
            self.n_cat1st_ph = tf.to_int32(self.next_state[:, :, :, 4])

            # item
            self.n_item_id_ph = tf.to_int32(self.next_state[:, :, :, 3])
            self.n_item_gender_ph = tf.to_int32(self.next_state[:, :, :, 5])
            self.n_item_buy_ph = tf.to_int32(self.next_state[:, :, :, 6])
            self.n_item_bc_ph = tf.to_int32(self.next_state[:, :, :, 7])
            self.n_item_dense_ph = self.next_state[:, :, :, 8:-1]

            # user
            self.n_user_age_ph = tf.to_int32(self.next_state[:, :, 0, 0])
            self.n_use_gender_ph = tf.to_int32(self.next_state[:, :, 0, 1])
            self.n_user_buy_ph = tf.to_int32(self.next_state[:, :, 0, 2])

            with tf.variable_scope("embedding"):
                self.emb_mtx = [tf.get_variable('overall_emb_mtx_{}'.format(i), [self.feature_size[i], self.emb_dim],
                                                initializer=tf.truncated_normal_initializer) for i in
                                range(len(self.feature_size))]

                cat1_emb = tf.nn.embedding_lookup(self.emb_mtx[4], self.cat1st_ph)

                # item embedding
                item_id_emb = tf.nn.embedding_lookup(self.emb_mtx[3], self.item_id_ph)
                item_gender_emb = tf.nn.embedding_lookup(self.emb_mtx[5], self.item_id_ph)
                item_buy_emb = tf.nn.embedding_lookup(self.emb_mtx[6], self.item_id_ph)
                item_bc_emb = tf.nn.embedding_lookup(self.emb_mtx[7], self.item_id_ph)

                # user specified embedding
                user_age_emb = tf.nn.embedding_lookup(self.emb_mtx[0], self.user_age_ph)
                user_gender_emb = tf.nn.embedding_lookup(self.emb_mtx[1], self.use_gender_ph)
                user_buy_emb = tf.nn.embedding_lookup(self.emb_mtx[2], self.user_buy_ph)
                user_emb = tf.expand_dims(tf.concat([user_age_emb, user_gender_emb, user_buy_emb], axis=-1), 2)
                user_emb = tf.tile(user_emb, [1, 1, self.rnn_time_step, 1])

                reward_emb = self.state[:, :, :, -1]
                reward_emb = tf.expand_dims(reward_emb, axis=-1)
                reward_emb = tf.layers.dense(reward_emb, units=self.emb_dim, activation=tf.nn.relu,
                                             kernel_initializer="truncated_normal",
                                             bias_initializer=tf.constant_initializer(0.01), name="reward_emb")
                total_s_emb = tf.concat(
                    [cat1_emb, item_id_emb, item_gender_emb, item_buy_emb, item_bc_emb, user_emb, self.item_dense_ph,
                     reward_emb], axis=3)
                total_s_emb = tf.reshape(total_s_emb, [-1, 2, self.rnn_time_step, (self.state_dim-19) * self.emb_dim+19])
                print("total_s_emb", total_s_emb)
                total_positive_s_emb = total_s_emb[:, 0, :, :]
                total_negative_s_emb = total_s_emb[:, 1, :, :]

                # next emb
                n_cat1_emb = tf.nn.embedding_lookup(self.emb_mtx[4], self.n_cat1st_ph)

                # item embedding
                n_item_id_emb = tf.nn.embedding_lookup(self.emb_mtx[3], self.n_item_id_ph)
                n_item_gender_emb = tf.nn.embedding_lookup(self.emb_mtx[5], self.n_item_id_ph)
                n_item_buy_emb = tf.nn.embedding_lookup(self.emb_mtx[6], self.n_item_id_ph)
                n_item_bc_emb = tf.nn.embedding_lookup(self.emb_mtx[7], self.n_item_id_ph)

                # user specified embedding
                n_user_age_emb = tf.nn.embedding_lookup(self.emb_mtx[0], self.n_user_age_ph)
                n_user_gender_emb = tf.nn.embedding_lookup(self.emb_mtx[1], self.n_use_gender_ph)
                n_user_buy_emb = tf.nn.embedding_lookup(self.emb_mtx[2], self.n_user_buy_ph)
                n_user_emb = tf.expand_dims(tf.concat([n_user_age_emb, n_user_gender_emb, n_user_buy_emb], axis=-1), 2)
                n_user_emb = tf.tile(n_user_emb, [1, 1, self.rnn_time_step, 1])

                n_reward_emb = self.next_state[:, :, :, -1]
                n_reward_emb = tf.expand_dims(n_reward_emb, axis=-1)
                n_reward_emb = tf.layers.dense(n_reward_emb, units=self.emb_dim, activation=tf.nn.relu,
                                               kernel_initializer="truncated_normal",
                                               bias_initializer=tf.constant_initializer(0.01), name="n_reward_emb")
                _ns_emb = tf.concat(
                    [n_cat1_emb, n_item_id_emb, n_item_gender_emb, n_item_buy_emb, n_item_bc_emb, n_user_emb,
                     self.n_item_dense_ph, n_reward_emb], axis=3)
                _ns_emb = tf.reshape(_ns_emb, [-1, 2, self.rnn_time_step, (self.state_dim-19) * self.emb_dim+19])
                print("_ns_emb: ", _ns_emb)
                total_positive_ns_emb = _ns_emb[:, 0, :, :]
                total_negative_ns_emb = _ns_emb[:, 1, :, :]

            with tf.variable_scope("eval"):
                self.q_predict = self.build_q_net(total_positive_s_emb, total_negative_s_emb, self.state_len[:, 0],
                                                  self.state_len[:, 1])

            print("self.q_predict: ", self.q_predict)
            self.logits = tf.identity(self.q_predict, name="logits")
            print("self.logits: {}".format(self.logits))

            # [-1, 8, 3] --> [-1, 8]
            _pred = tf.argmax(self.q_predict, axis=-1)
            self.cal_action = tf.to_float(_pred, name='pred')
            print("self.cal_action:", self.cal_action)

            with tf.variable_scope("target"):
                q_target = self.build_q_net(total_positive_ns_emb, total_negative_ns_emb, self.state_len[:, 2],
                                            self.state_len[:, 3])

            self.q_target = tf.stop_gradient(q_target)
            print("self.q_target: ", self.q_target)

            q_val = tf.reduce_sum(
                tf.multiply(self.q_predict, tf.one_hot(self.action, self.action_dim)), -1
            )

            if not self.use_double_q:
                q_next = tf.where(
                    self.done,
                    self.reward,
                    self.reward + self.gamma * tf.reduce_max(self.q_target, axis=-1),
                )
            else:
                one_hot_max_a = tf.one_hot(_pred, self.action_dim)
                q_next = tf.where(
                    self.done,
                    self.reward,
                    self.reward + self.gamma * tf.reduce_sum(
                        tf.multiply(one_hot_max_a, self.q_target), -1),
                )

            print("predict q_val: {}\ttarget q_next: {}".format(q_val, q_next))

            self.loss = tf.losses.mean_squared_error(q_val, q_next)
            self.kl_loss, self.sum_freq, self.norm_freq = self.calc_kl(
                self.q_predict, ad_freq=self.t_ratio)

            self.loss += self.kl_weight * self.kl_loss
            # self.kl_loss, self.sum_freq, self.norm_freq = tf.constant(0.0),tf.constant(0.0),tf.constant(0.0)

            for v in tf.trainable_variables():
                if 'bias' not in v.name:
                    self.loss += self.reg_lambda * tf.nn.l2_loss(v)

            # todo: penalty for target values, ei, batch predict, ad/share ~== 1

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_step = self.optimizer.minimize(self.loss)

            _eval_paras = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="eval"
            )
            _target_paras = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="target"
            )
            print("\n_eval_paras\n", _eval_paras)

            with tf.variable_scope("soft_replace"):
                self.update_target_q = [
                    tf.assign(t, e * self.tau + (1 - self.tau) * t) for t, e in zip(_target_paras, _eval_paras)
                ]

            self.debug_paras = [_target_paras, _eval_paras]
            if self.out_graph_path:
                tf.summary.FileWriter(self.out_graph_path, self.sess.graph)

            self.saver = tf.train.Saver()

            self.sess.run(tf.global_variables_initializer())

    @staticmethod
    def calc_kl(logits, ad_freq):
        p = tf.nn.softmax(tf.reshape(logits, [-1, 3]))
        y_hard = tf.cast(tf.equal(p, tf.reduce_max(p, axis=1, keepdims=True)), p.dtype)
        gumbel_logits = tf.stop_gradient(y_hard - p) + p

        sum_freq_all = tf.reduce_sum(gumbel_logits, axis=0)

        sum_ad_share = sum_freq_all[0] + sum_freq_all[1] + sum_freq_all[2]
        norm_freq = sum_freq_all / (sum_ad_share + 1e-6)

        kl = norm_freq[0] * tf.log(norm_freq[0] / ad_freq+ 1e-6) + \
             norm_freq[1] * tf.log(norm_freq[1] / ad_freq+ 1e-6) + \
             norm_freq[2] * tf.log(norm_freq[2] / ad_freq+ 1e-6)

        return kl, sum_freq_all, norm_freq

    def add_buff(self, state, action, reward, next_state, done, state_len):
        self.buff.append((state, action, reward, next_state, done, state_len))

    def build_q_net(self, positive_inputs, negative_inputs, positive_state_len, negative_state_len):
        pos_rnncell = GRUCell(self.gru_dim)
        neg_rnncell = GRUCell(self.gru_dim)
        positive_seq_ht, positive_seq_final_state = dynamic_rnn(pos_rnncell, inputs=positive_inputs,
                                                                sequence_length=positive_state_len,
                                                                dtype=tf.float32,
                                                                scope='pos_rnn')
        negative_seq_ht, negative_seq_final_state = dynamic_rnn(neg_rnncell, inputs=negative_inputs,
                                                                sequence_length=negative_state_len,
                                                                dtype=tf.float32,
                                                                scope='neg_rnn')
        output = tf.concat([positive_seq_final_state, negative_seq_final_state], axis=-1)
        for i, hidden in enumerate(self.hidden_layers):
            output = tf.layers.dense(
                output,
                hidden,
                activation=tf.nn.relu,
                kernel_initializer="truncated_normal",
                bias_initializer=tf.constant_initializer(0.01),
                name="hidden_{}".format(i),
            )

        q_pred = tf.layers.dense(
            output,
            self.action_dim,
            activation=None,
            kernel_initializer="truncated_normal",
            bias_initializer=tf.constant_initializer(0.01),
            name="q_predict",
        )

        return tf.reshape(q_pred, [-1, self.action_dim])

    def buff_ready(self):
        return len(self.buff) > self.threshold

    def _run_train_step(self, states, actions, rewards, next_states, dones, state_len):
        with self.graph.as_default():
            loss_val, _, kl_val, sv, fv = self.sess.run(
                [self.loss, self.train_step, self.kl_loss, self.sum_freq, self.norm_freq],
                feed_dict={
                    self.state: states,
                    self.action: actions,
                    self.reward: rewards,
                    self.next_state: next_states,
                    self.done: dones,
                    self.state_len: state_len,
                },
            )

            if self.train_cnt % 60 == 0:
                self.sess.run(self.update_target_q)

        self.kl_record.append(kl_val)
        self.freq_record.append(sv)
        self.norm_freq_record.append(fv)
        return loss_val

    def train(self):
        self.train_cnt += 1
        mini_batch = random.sample(self.buff, self.batch_size)
        states, actions, rewards, next_states, dones, state_len = map(list, zip(*mini_batch))
        actions = np.squeeze(actions, axis=-1)
        rewards = np.squeeze(rewards, axis=-1)
        loss_val = self._run_train_step(states, actions, rewards, next_states, dones, state_len)

        return loss_val

    def train_with_transition(self, states, actions, rewards, next_states, dones, state_len):
        self.train_cnt += 1
        actions = np.squeeze(actions, axis=-1)
        rewards = np.squeeze(rewards, axis=-1)
        loss_val = self._run_train_step(states, actions, rewards, next_states, dones, state_len)
        return loss_val

    def save(self, to_path, step=None):
        if not os.path.exists(to_path):
            os.makedirs(to_path)
        print("saver checkpoint to path: {}".format(to_path))

        model_path = os.path.join(to_path, DEFAULT_MODEL_NAME)
        with self.graph.as_default():
            self.saver.save(self.sess, model_path, global_step=step or self.train_cnt)

    def load(self, from_path, step=-1):
        if step < 0:
            model_id = tf.train.latest_checkpoint(from_path)
            meta_file = '{}.meta'.format(model_id)
        else:
            model_id = os.path.join(from_path, '{}-{}'.format(DEFAULT_MODEL_NAME, step))
            meta_file = os.path.join(from_path, '{}-{}.meta'.format(DEFAULT_MODEL_NAME, step))

        print("load weight with model id: {} \tmeta: {}".format(model_id, meta_file))
        with self.graph.as_default():
            new_saver = tf.train.import_meta_graph(meta_file)
            new_saver.restore(self.sess, model_id)

    def save_pb(self, to_path, name_list):
        constant_graph = tf.graph_util.convert_variables_to_constants(
            self.sess,
            self.sess.graph_def,
            name_list)

        with tf.gfile.GFile(to_path, "wb") as f:
            f.write(constant_graph.SerializeToString())
        print("save {} ops within graph.".format(len(constant_graph.node)))

    def predict(self, data, state_len):
        with self.graph.as_default():
            pred_labels = self.sess.run(self.cal_action, feed_dict={self.state: data, self.state_len: state_len})
        return pred_labels

    def infer_action(self, state, state_len):
        """select action with maximum q value."""
        a = self.sess.run(self.cal_action, feed_dict={self.state: state, self.state_len: state_len})
        return a[0]


def load_pb_test(pb_file, data):
    graph = tf.Graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config, graph=graph)

    with graph.as_default():
        with open(pb_file, "rb") as gf:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(gf.read())
            ph_feat, ph_seq_len, keep_prob, pred, logits = tf.import_graph_def(
                graph_def,
                return_elements=[
                    "inputs/seq_feat_ph:0",
                    "inputs/seq_length_ph:0",
                    "inputs/keep_prob:0",
                    "pred:0",
                    "logits:0"
                ])
            ret_pred, ret_logits = sess.run([pred, logits],
                                            feed_dict={
                                                ph_feat: data[0],
                                                ph_seq_len: data[1],
                                                keep_prob: data[2]})
        return ret_pred, ret_logits


def save_agent_pb(agent, to_path, pb_name="model.pb"):
    if not os.path.exists(to_path):
        os.makedirs(to_path)

    print("save pb to path: {}".format(to_path))

    save_pb_path = os.path.join(to_path, pb_name)
    agent.save_pb(to_path=save_pb_path, name_list=["inputs/seq_feat_ph",
                                                   "inputs/seq_length_ph",
                                                   "inputs/keep_prob",
                                                   "pred", "logits"])
