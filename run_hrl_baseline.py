# -*- coding:utf-8 -*-

import os
import glob
import shutil
from time import time
from datetime import datetime
import argparse
import pprint
from collections import deque, defaultdict
from functools import partial
import random
import pandas as pd
import numpy as np
from multiprocessing import Pool
import common.config_task as conf
from hrl_baseline.policy_model import DQNAgent, save_agent_pb
import pickle

from hrl_baseline.eval_utils import eval_policy
from hrl_baseline.construct_transition_data import  construct_sarsd_hrl

from rnn_version.utils import process_baseline_data, cal_baseline_criterion

dummy_custom = {
    'hidden_layers': [320, 1024, 100],
    'gamma': 0.9,
    'epoch_num':  20,
    'learning_rate': 0.00001,
    'tau': 0.005,

    'used_train_days': 7,

    # for preprocess dataset
    'prepare_data_parallel_num': 4,
    'exp_name': "train_eval_inference",

    # within train stage, construct session data
    'construct_session_parallel_num': 15,
    'kl_weight': 1.0,

    'if_save_checkpoint': False,
    'if_save_pb': True,
    'if_save_metric_to_file': True,

    'verbose': False,
    'data_list_id':   'cds_111124244',
}


class BestMetric(object):
    def __init__(self, k, seed, predict_ratio, test_files=None, best_topk=20,
                 min_epoch=2, var_threshold=0.01):
        self._top_k = k
        self._seed = seed
        self._p_ratio = predict_ratio
        self.maximum_revenue = -9999.0
        self.best_train_cnt = 0
        self.min_epoch = min_epoch
        self._v_ths = var_threshold

        self.raw_eval_list = list()
        self.raw_cols_name = ["stage", "scope", "ndcg", "revenue",
                              "real_ad_cnt", "real_share_cnt", "real_nature_cnt", "model_ad_cnt", "model_share_cnt","model_nature_cnt","alpha_ndcg", "alpha_revenue"]

        self.summary_eval_list = list()
        self.sum_cols_name = ["stage", "topK", "NDCG", "REVENUE", "real_RATIO", "predict_RATIO","alpha NDCG", "alpha REVENUE"]

    def _best(self, purified_val, epoch):
        if epoch < self.min_epoch:
            return False

        # prior max revenue
        cur_rev = [item[3] for item in purified_val if item[1] == self._top_k]
        if len(cur_rev) > 1:
            raise ValueError("not support multi value: {}".format(cur_rev))

        # used the real_ratio to check convergence
        select_pr = [item[4] for item in self.summary_eval_list[-5:] if item[1] == self._top_k]
        select_pr = np.array(select_pr)

        # last point to 0.04 * 5 range  np.sum(np.abs(ratio - target_ratio))
        if cur_rev[0] > self.maximum_revenue and np.var(select_pr) < self._v_ths:
            self.maximum_revenue = cur_rev[0]
            return True

        return False

    def record_if_best(self, raw, purified, cur_epoch):
        self.record(raw, purified)
        return self._best(purified, cur_epoch)

    def record(self, raw, purified):
        self.raw_eval_list.extend(raw)
        self.summary_eval_list.extend(purified)

    def verbose(self):
        summary_df = pd.DataFrame(self.summary_eval_list, columns=self.sum_cols_name)
        print(summary_df.to_dict(orient="list"))

    def save(self, raw_to_file, purified_to_file):
        raw_df = pd.DataFrame(self.raw_eval_list, columns=self.raw_cols_name)
        raw_df["seed"] = self._seed

        summary_df = pd.DataFrame(self.summary_eval_list, columns=self.sum_cols_name)
        summary_df["seed"] = self._seed

        if not os.path.exists(os.path.dirname(raw_to_file)):
            os.makedirs(os.path.dirname(raw_to_file))

        print("save results to file: {}".format(raw_to_file))
        raw_df.to_hdf(raw_to_file, conf.default_pd_ds, mode="w")
        summary_df.to_hdf(purified_to_file, conf.default_pd_ds, mode="w")
        summary_df.to_csv(
            os.path.join(args.target_dir,
                         "summary_metric_seed{}_ratio{}.csv".format(args.seed, args.ratio)),
            index=False)


def _exp_on_one_file(agent, hdf5_item, parallel_worker, processes_num,
                     bm, exp_output_path, e_prefix, _verbose,
                     eval_data_file, epoch):
    train_skip_item_cnt = 0

    partial_job = partial(construct_sarsd_hrl, max_time_length=10,max_interaction_length=30)

    _bsize_record = deque(maxlen=100)
    _loss_record = deque(maxlen=100)

    time_info = defaultdict(float)
    pkl_stime = time()

    f = open(hdf5_item, "rb")
    df_raw = pickle.load(f)
    f.close()

    # sort
    random.shuffle(df_raw)

    time_info["read_sort"] = time() - pkl_stime

    sess_nums, to_eval_flag = len(df_raw), False

    batch_max_n = processes_num
    train_sess_val = list([None for k in range(batch_max_n)])
    d_cur_p = 0

    for sess_idx, value in enumerate(df_raw):
        if len(value[2]) < conf.SESSION_LEN_MIN:
            # record the skip cnt
            train_skip_item_cnt += len(value[2])
            continue

        if not to_eval_flag and sess_idx in (int(sess_nums // 2), sess_nums - 1):
            to_eval_flag = True

        if sess_idx + 1 != sess_nums and d_cur_p < batch_max_n:
            train_sess_val[d_cur_p] = value
            d_cur_p += 1
            continue

        _t = time()
        batch_data = parallel_worker.map(partial_job, train_sess_val[:d_cur_p])
        bs = np.concatenate([b[0] for b in batch_data], axis=0)
        ba = np.concatenate([b[1] for b in batch_data], axis=0)
        br = np.concatenate([b[2] for b in batch_data], axis=0)
        bs_n = np.concatenate([b[3] for b in batch_data], axis=0)
        bd = np.concatenate([b[4] for b in batch_data], axis=0)
        b_len = np.concatenate([b[5] for b in batch_data], axis=0)

        _bsize_record.append(bs.shape[0])
        time_info["construct"] += time() - _t

        _t = time()
        loss = agent.train_with_transition(bs, ba, br, bs_n, bd, b_len)
        _loss_record.append(loss)
        time_info["train"] += time() - _t
        d_cur_p = 0

        if agent.train_cnt % 1000 == 1:
            print("{}train_cnt:{:>6}, b_size:{:6.1f}, loss:{:8.6f}, kl_loss:{:8.6f}".format(
                e_prefix, agent.train_cnt, np.nanmean(_bsize_record),
                np.nanmean(_loss_record), np.nanmean(agent.kl_record)))

    time_info["loop"] = time() - pkl_stime

    print("train: {} with time:{}".format(hdf5_item, dict(time_info)))

    if train_skip_item_cnt > 0:
        print("TRAIN Skip item(len<{}) cnt: {} within each epoch!".format(
            conf.SESSION_LEN_MIN, train_skip_item_cnt))

    print("train_cnt: {} train bsize mean: {}, likes:{}".format(
        agent.train_cnt, np.nanmean(_bsize_record),
        np.random.choice(_bsize_record, 5, replace=False)))
    print("loss list: mean-{}, likes: {}".format(
        np.mean(_loss_record), np.random.choice(_loss_record, 5, replace=False)))


def _save_model(agent, custom_configs, exp_output_path):
    if custom_configs.get("if_save_pb", True):
        save_agent_pb(agent,
                      os.path.join(exp_output_path, "output_pb"),
                      "model.pb")

    if custom_configs.get("if_save_checkpoint", False):
        agent.save(to_path=os.path.join(
            exp_output_path, "output_checkpoint"))


def main():
    target_dir = args.target_dir

    epoch_num = args.epochs

    to_exp_name = 'train'
    exp_output_path = os.path.join(
        target_dir, "exp_outputs",
        "RLT{}".format(datetime.now().strftime("%y%m%d_%H%M%S")))

    _verbose = False
    if _verbose:
        online_feat_map = os.path.join(args.target_dir, "featureMap.featureMap")
        if not os.path.exists(online_feat_map):
            shutil.copy(os.path.join(target_dir, "feature", "featureMap.featureMap"),
                        online_feat_map)

    h5_file_list = glob.glob(
        os.path.join(args.file_path,"clean_data", "train_part*.pkl"))
    h5_file_list = sorted(h5_file_list, reverse=False)

    test_file_list = glob.glob(
        os.path.join(args.file_path, "clean_data", "test_part*.pkl"))
    test_file_list = sorted(test_file_list, reverse=False)

    print(" >>> START [{}] experiment >>>".format(to_exp_name))

    # read feature size
    f = open(os.path.join(args.file_path, "clean_data", "map_dicts.pkl"), "rb")
    map_dicts = pickle.load(f)
    feature_size = [len(map_dicts[1][0].keys()) + 1, len(map_dicts[1][1].keys()) + 1, len(map_dicts[1][2].keys()) + 1,
                    len(map_dicts[2][0].keys()) + 1, len(map_dicts[2][1].keys()) + 1, len(map_dicts[2][2].keys()) + 1,
                    len(map_dicts[2][3].keys()) + 1,
                    len(map_dicts[2][4].keys()) + 1]
    print(feature_size)
    _feat_encode_size = feature_size

    # unify the behavior feature size to single emb, 101*2+11*2 for TOTAL/each feature
    _behavior_feat_size = (101 * 1 + 11 * 1) * 1

    # number 3 point to ‘hour,city,device_name’, and 4 point to multi_feat
    configs = dict(
        state_dim=5 * conf.STEP_LEN + 3 + 4,
        list_length=conf.STEP_LEN,
        action_dim=3,
        hidden_layers=[256,128,64],
        gamma=0.9,
        learning_rate=args.lr,
        epsilon=[0.5, 0.01],
        batch_size=args.batch_size,
        replay_size=int(1e5),
        seed=8888,
        ad_share_ratio=0.5,
        kl_weight=args.kl_weight,
        feature_size=_feat_encode_size,
        behavior_feat_size=_behavior_feat_size,
        use_double_q=False,
        out_graph_path="log_dqn" if _verbose else None,
    )
    print("Init Agent with paras:\n", pprint.pformat(configs, indent=0, width=1, ))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    agent = DQNAgent(configs)

    csp_n = 2
    parallel_worker = Pool(processes=csp_n)
    print("construct_session_parallel_num: {}".format(csp_n))

    bm = BestMetric(k=10, seed=args.seed, predict_ratio=args.ratio,
                    min_epoch=3, var_threshold=0.01)

    if "TRAIN" in str(to_exp_name).upper():
        one_eval_raw, metric_summary = eval_policy(
            agent, test_file_list, parallel_worker, args.batch_size,
            redo_per_job=8, sample_first_sess=-1,  # 600,
            scope_num_set=(10, 20), stage="eval_0",
            sort_position=True)
        bm.record(one_eval_raw, metric_summary)

        for epoch_i in range(1, epoch_num+1):
            e_prefix = "EPOCH[{}/{}]>>".format(epoch_i, epoch_num)
            for hdf5_item in h5_file_list:
                _exp_on_one_file(agent, hdf5_item, parallel_worker, args.batch_size,
                                bm, exp_output_path, e_prefix, _verbose,
                                eval_data_file=test_file_list, epoch=epoch_i)
            one_eval_raw, metric_summary = eval_policy(
                agent, test_file_list, parallel_worker, args.batch_size,
                redo_per_job=8, sample_first_sess=-1,  # 600,
                scope_num_set=(10, 20), stage="eval_0",
                sort_position=True)
            bm.record(one_eval_raw, metric_summary)
            metric_to_file = os.path.join(exp_output_path, "output_metric",
                                  "raw_metric_seed{}_ratio{}.h5".format(args.seed, args.ratio))
            summary_to_file = os.path.join(exp_output_path, "output_metric",
                                        "summary_metric_seed{}_ratio{}.h5".format(args.seed, args.ratio))
            bm.save(metric_to_file, summary_to_file)


    parallel_worker.close()
    parallel_worker.join()

    if _verbose:
        bm.verbose()
    metric_to_file = os.path.join(exp_output_path, "output_metric",
                                  "raw_metric_seed{}_ratio{}.h5".format(args.seed, args.ratio))
    summary_to_file = os.path.join(exp_output_path, "output_metric",
                                   "summary_metric_seed{}_ratio{}.h5".format(args.seed, args.ratio))
    bm.save(metric_to_file, summary_to_file)

    #eval_original(filename_list[-1:], top_k=[10, 20])



if __name__ == "__main__":
    st_time = time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str, help="target data dir")
    parser.add_argument("--file_path", type=str, help="data dir")
    parser.add_argument("--seed", type=int, default=3407, help="train seed")
    parser.add_argument("--ratio", type=float, default=0.5, help="target ratio")
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--gpu_num", type=str, default=0)

    args, _ = parser.parse_known_args()
    if _:
        print("get unknown args: {}".format(_))
    print(pprint.pformat(args, indent=0, width=1, ))

    main()
