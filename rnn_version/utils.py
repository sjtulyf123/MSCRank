import pickle
import numpy as np
import random
import pandas as pd

from collections import Counter, defaultdict
import os
import argparse
import time
import copy
import pprint
from common.sys_utils import get_initial_ranking_in_one_session, get_reranked_list, cal_session_alpha_ndcg
from functools import partial

name_map = {'uid': 0, 'u_age': 1, 'u_gender': 2, 'u_buy': 3, 'iid': 4, 'cat1d': 5, 'i_gender': 6, 'i_buy': 7, 'i_BC': 8,
            'price':18, 'label': -1}


def get_batch(data, seq_len, init_list, batch_size, batch_no):
    return data[batch_size * batch_no: batch_size * (batch_no + 1)], seq_len[batch_size * batch_no: batch_size * (
            batch_no + 1)], init_list[batch_size * batch_no: batch_size * (batch_no + 1)]

def construct_one_session(one_session):
    uid = one_session[0]
    u_feat = one_session[1]
    i_feat = one_session[2]
    dense1 = one_session[3]
    dense2 = one_session[4]
    label = one_session[5]
    v = []
    for pos in range(len(i_feat)):
        tmp = [uid] + u_feat + i_feat[pos] + dense1[pos] + dense2[pos] + [label[pos]]
        v.append(tmp)
    v = np.array(v, dtype=float)
    return v


def process_baseline_data(data_file_list, parallel_worker):
    data = []
    for hdf_item in data_file_list:
        f = open(hdf_item, "rb")
        raw_data = pickle.load(f)
        f.close()

        for one_session in raw_data:
            data.append(construct_one_session(one_session))
    print("session number: ", len(data))
    return data


def cal_baseline_criterion(baseline_data, scope_number):
    ndcg = []
    cnt_ad = 0
    cnt_share = 0
    cnt_nature = 0
    revenue = []
    alpha_ndcg = []
    alpha_revenue = []
    for session in range(len(baseline_data)):
        session_data = baseline_data[session]
        item_length = len(session_data)
        clicks = session_data[:,name_map['label']].astype(int).tolist()
        udf_cat = session_data[:,name_map['cat1d']].astype(int).tolist()
        bid = session_data[:,name_map['price']].tolist()

        # ratio
        cat_cnt = Counter(udf_cat[:scope_number])
        cnt_ad += cat_cnt[0]
        cnt_share += cat_cnt[1]
        cnt_nature += cat_cnt[2]

        # revenue
        tmp_revnue = 0.0
        tmp_alpha_revenue = 0.0
        cat_acc = [0, 0, 0]
        for item in range(min(len(clicks), scope_number)):
            tmp_revnue += clicks[item] * bid[item]
            tmp_alpha_revenue += (clicks[item] * bid[item]) * pow(1 - 0.5, cat_acc[udf_cat[item]])
            cat_acc[udf_cat[item]] += 1
        revenue.append(tmp_revnue)
        alpha_revenue.append(tmp_alpha_revenue)

        # ndcg
        if len(clicks) < 2:
            ndcg.append(sum(clicks))
            continue
        final = list(range(len(clicks)))
        gold = get_reranked_list(clicks)
        ideal_dcg = 0
        dcg = 0
        # define scope for calculation
        scope_final = final[:scope_number]
        scope_gold = gold[:scope_number]
        for _i, _f, _g in zip(range(1, scope_number + 1), scope_final, scope_gold):
            dcg += (pow(2, clicks[_f]) - 1) / (np.log2(_i + 1))
            ideal_dcg += (pow(2, clicks[_g]) - 1) / (np.log2(_i + 1))
        _ndcg = float(dcg) / ideal_dcg if ideal_dcg != 0 else 0.
        ndcg.append(_ndcg)
        alpha_ndcg.append(cal_session_alpha_ndcg(clicks, udf_cat, scope_number))
    total_item_num = cnt_ad+cnt_share+cnt_nature
    percentage = np.array([cnt_ad,cnt_share,cnt_nature],dtype=np.float)/float(total_item_num)
    kl_ratio =np.sum(percentage*np.log(percentage/float(0.333)))
    print("total category cnt over top {}: {}, {}, {}.".format(scope_number,cnt_ad,cnt_share,cnt_nature))
    # return np.mean(np.array(ndcg)), np.mean(np.array(revenue)), \
    #        (cnt_ad / float(cnt_share) + cnt_share / float(cnt_nature) + cnt_nature / float(cnt_ad)) / float(3), \
    #        np.mean(np.array(alpha_ndcg)), np.mean(np.array(alpha_revenue))
    return np.mean(np.array(ndcg)), np.mean(np.array(revenue)), \
           kl_ratio, \
           np.mean(np.array(alpha_ndcg)), np.mean(np.array(alpha_revenue))


def cal_round_robin_baseline_criterion(baseline_data, scope_number):
    ndcg = []
    cnt_ad = 0
    cnt_share = 0
    cnt_nature = 0
    revenue = []
    alpha_revenue = []
    alpha_ndcg = []
    for session in range(len(baseline_data)):
        session_data = baseline_data[session]
        item_length = len(session_data)
        if item_length <= 5:
            continue
        _clicks = session_data[:, name_map['label']].astype(int).tolist()
        _udf_cat = session_data[:, name_map['cat1d']].astype(int).tolist()
        _bid = session_data[:, name_map['price']].tolist()

        cat_idx = [[], [], []]
        for j in range(len(_udf_cat)):
            cat_idx[_udf_cat[j]].append(j)
        sorted_pos = []
        for j in range(max(max(len(cat_idx[0]), len(cat_idx[1])), len(cat_idx[2]))):
            if j < len(cat_idx[0]):
                sorted_pos.append(cat_idx[0][j])
            if j < len(cat_idx[1]):
                sorted_pos.append(cat_idx[1][j])
            if j < len(cat_idx[2]):
                sorted_pos.append(cat_idx[2][j])
        clicks = [_clicks[i] for i in sorted_pos]
        udf_cat = [_udf_cat[i] for i in sorted_pos]
        bid = [_bid[i] for i in sorted_pos]

        # ratio
        cat_cnt = Counter(udf_cat[:scope_number])
        cnt_ad += cat_cnt[0]
        cnt_share += cat_cnt[1]
        cnt_nature+=cat_cnt[2]

        # revenue
        tmp_revnue = 0.0
        tmp_alpha_revenue = 0.0
        cat_acc = [0, 0, 0]
        for item in range(min(len(clicks), scope_number)):
            tmp_revnue += clicks[item] * bid[item]
            tmp_alpha_revenue += (clicks[item] * bid[item]) * pow(1 - 0.5,cat_acc[udf_cat[item]])
            cat_acc[udf_cat[item]] += 1
        revenue.append(tmp_revnue)
        alpha_revenue.append(tmp_alpha_revenue)

        # ndcg
        if len(clicks) < 2:
            ndcg.append(sum(clicks))
            continue
        final = list(range(len(clicks)))
        gold = get_reranked_list(clicks)
        ideal_dcg = 0
        dcg = 0
        # define scope for calculation
        scope_final = final[:scope_number]
        scope_gold = gold[:scope_number]
        for _i, _f, _g in zip(range(1, scope_number + 1), scope_final, scope_gold):
            dcg += (pow(2, clicks[_f]) - 1) / (np.log2(_i + 1))
            ideal_dcg += (pow(2, clicks[_g]) - 1) / (np.log2(_i + 1))
        _ndcg = float(dcg) / ideal_dcg if ideal_dcg != 0 else 0.
        ndcg.append(_ndcg)
        alpha_ndcg.append(cal_session_alpha_ndcg(clicks, udf_cat, scope_number))
    total_item_num = cnt_ad+cnt_share+cnt_nature
    percentage = np.array([cnt_ad,cnt_share,cnt_nature],dtype=np.float)/float(total_item_num)
    kl_ratio =np.sum(percentage*np.log(percentage/float(0.333)))
    # return np.mean(np.array(ndcg)), np.mean(np.array(revenue)), \
    #        (cnt_ad / float(cnt_share) + cnt_share / float(cnt_nature) + cnt_nature / float(cnt_ad)) / float(3), \
    #        np.mean(np.array(alpha_ndcg)), np.mean(np.array(alpha_revenue))
    return np.mean(np.array(ndcg)), np.mean(np.array(revenue)), \
           kl_ratio, \
           np.mean(np.array(alpha_ndcg)), np.mean(np.array(alpha_revenue))


def gen_initial_ranking(data):
    ad_lists, share_lists, nature_lists = [], [], []
    for session in data:
        _ad_list, _share_list, _nature_list = get_initial_ranking_in_one_session(session, name_map)
        ad_lists.append(_ad_list)
        share_lists.append(_share_list)
        nature_lists.append(_nature_list)
    return ad_lists, share_lists, nature_lists


def gen_train_data(baseline_data, window_size=30):
    res_data = []
    res_init_list = []
    cnt = 0
    for session in baseline_data:
        init_list = get_initial_ranking_in_one_session(session, name_map)
        if np.sum(session[:, name_map['label']]) >= 1:
            res_data.append(session)
            res_init_list.append(init_list)
        else:
            cnt += 1
    print("remove no_click window: {}".format(cnt))
    return res_data, res_init_list


def gen_evaluate_data(baseline_data, window_size):
    return baseline_data


def gen_data(file_list, parallel_worker, window_size=30, data_mode='train', shuffle=False):
    assert data_mode in ['train', 'valid', 'test']
    baseline_data = process_baseline_data(file_list, parallel_worker)
    ad_lists, share_lists, nature_lists = gen_initial_ranking(baseline_data)
    if data_mode == 'train':
        res_data, res_init_list = gen_train_data(baseline_data, window_size)
        if shuffle:
            state = random.getstate()
            random.shuffle(res_data)
            random.setstate(state)
            random.shuffle(res_init_list)
    else:
        res_data = gen_evaluate_data(baseline_data, window_size)
        res_init_list = []
    return res_data, ad_lists, share_lists, nature_lists, res_init_list


def padding_all_data(data, init_list_data, window_size=30, behavior_length=12):
    c = time.time()
    data_length = len(data)
    feature_dim = data[0].shape[1]
    padding_batch_data = np.zeros((data_length, window_size, feature_dim), dtype=np.float32)
    seq_len = [data[i].shape[0] for i in range(data_length)]
    for list_idx in range(data_length):
        padding_batch_data[list_idx, :seq_len[list_idx], 1:] = data[list_idx][:, 1:]

    padding_init_list_data = np.zeros((data_length, 3, window_size, feature_dim + 1), dtype=np.float32)
    for list_idx in range(data_length):
        for cat_idx in [0, 1, 2]:
            cur_cat_len = init_list_data[list_idx][cat_idx].shape[0]
            if cur_cat_len == 0:
                continue
            padding_init_list_data[list_idx, cat_idx, :min(window_size, cur_cat_len), 1:feature_dim] = \
                init_list_data[list_idx][cat_idx][:min(window_size, cur_cat_len), 1:]
            padding_init_list_data[list_idx, cat_idx, :min(window_size, cur_cat_len), feature_dim] = cur_cat_len
    print(time.time() - c)
    return padding_batch_data, seq_len, padding_init_list_data


def padding_test_data(data, init_list_data, window_size=30, behavior_length=12):
    c = time.time()
    data_length = len(data)
    feature_dim = data[0].shape[1]
    padding_batch_data = np.zeros((data_length, window_size, feature_dim), dtype=np.float32)
    seq_len = [data[i].shape[0] for i in range(data_length)]
    for list_idx in range(data_length):
        padding_batch_data[list_idx, :seq_len[list_idx], 1:] = data[list_idx][:, 1:]

    padding_init_list_data = np.zeros((data_length, 3, window_size, feature_dim + 1), dtype=np.float32)
    for list_idx in range(data_length):
        for cat_idx in [0, 1, 2]:
            cur_cat_len = init_list_data[cat_idx][list_idx].shape[0]
            if cur_cat_len == 0:
                continue
            padding_init_list_data[list_idx, cat_idx, :min(window_size, cur_cat_len), 1:feature_dim] = \
                init_list_data[cat_idx][list_idx][:min(window_size, cur_cat_len), 1:]
            padding_init_list_data[list_idx, cat_idx, :min(window_size, cur_cat_len), feature_dim] = cur_cat_len
    print(time.time() - c)
    return padding_batch_data, seq_len, padding_init_list_data


def save_padded_data(data, path, file_name):
    assert 'pkl' in file_name
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, file_name), 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print("Saving data to {} with shape {}.".format(os.path.join(path, file_name), data.shape))


def load_padded_data(file):
    fr = open(file, 'rb')
    load_data = pickle.load(fr)
    fr.close()
    return load_data


def get_default_config():
    d = {'batch_size': 128,
         'epochs': 30,
         'lr': 0.00001,
         'hidden_size': 64,
         'multi_task': True,
         'optimize_ratio': True,
         'use_new_rnn': False,
         'use_initlist_attention': False,
         'kl_weight': 1.0,
         'ranker_weight': 1.0,
         'only_store_best': False,
         'update_padded_data': False,
         'behavior_length': 0,
         'used_train_days': 100,
         'data_list_id': 'cds_111124073',
         'evaluate_times_per_epoch': 5,
         'soft_ranker_tau': 1.0,
         'use_js': False,

         # often keep default params
         'seed': 8888,
         'embedding_dim': 64,
         'reg_lambda': 0.1,
         'label_decay': 0.8,
         'keep_prob': 0.8,
         'sample_pair_num': 5,
         'cat_rnn_type': 'gru',
         'item_rnn_type': 'gru',
         'interaction_mode': 'one_to_one',
         'window_size': 30,
         'use_2level_category': True,
         'use_3level_category': True,
         'use_l2_regular': False,
         'evaluate_cri_number': 20,
         'revenue_decrease_bound': 0.03,
         'ideal_ratio': 1.0
         }
    return d


def print_baselines(file_list, parallel_worker):
    evaluate_number = [3, 5, 10, 20]
    data = process_baseline_data(file_list, parallel_worker)
    res_dict = {'original': {'ndcg': [], 'revenue': [],
                             'ratio': []},
                'revenue_max': {'ndcg': [],
                                'revenue': [],
                                'ratio': []},
                'round_robin': {'ndcg': [],
                                'revenue': [],
                                'ratio': []},
                'ndcg_max': {'ndcg': [], 'revenue': [],
                             'ratio': []},
                'revenue_upper_bound': {'ndcg': [], 'revenue': [],
                                        'ratio': []}
                }
    _format = "{:^20}\t" * 6
    f_format = "{:^20}\t" + "{:^20.6f}\t" * 5

    print("original:")
    print(_format.format("topK", "NDCG", "REVENUE", "real RATIO", "alpha NDCG", "alpha REVENUE"))
    for n in evaluate_number:
        res_ndcg, res_revenue, res_ratio, res_alpha_ndcg, res_alpha_revenue = cal_baseline_criterion(data, n)
        print(f_format.format(n, res_ndcg, res_revenue, res_ratio, res_alpha_ndcg, res_alpha_revenue))
        res_dict['original']['ndcg'].append(round(float(res_ndcg), 4))
        res_dict['original']['revenue'].append(round(float(res_revenue), 4))
        res_dict['original']['ratio'].append(round(float(res_ratio), 4))

    print("round robin:")
    print(_format.format("topK", "NDCG", "REVENUE", "real RATIO","alpha NDCG","alpha REVENUE"))
    for n in evaluate_number:
        res_ndcg, res_revenue, res_ratio,res_alpha_ndcg,res_alpha_revenue = cal_round_robin_baseline_criterion(data, n)
        print(f_format.format(n, res_ndcg, res_revenue, res_ratio,res_alpha_ndcg,res_alpha_revenue))
        res_dict['round_robin']['ndcg'].append(round(float(res_ndcg), 4))
        res_dict['round_robin']['revenue'].append(round(float(res_revenue), 4))
        res_dict['round_robin']['ratio'].append(round(float(res_ratio), 4))
    return res_dict
