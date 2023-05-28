"""Eval policy."""
import os
import math
from functools import partial
from collections import defaultdict
from multiprocessing import Process, Pipe
from time import time
import numpy as np
import pandas as pd
from common.sys_utils import cal_one_session_criterion, get_initial_ranking_in_one_session
import common.config_task as conf
import pickle
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


def construct_state_hrl(sorted_sess,cur_evaluate_length=30,max_time_length=10):
    """sarsd with numpy."""
    data_len = sorted_sess.shape[0]

    # only fetch ‘floor’ round number
    state_num = min(data_len,cur_evaluate_length)

    s_feature = sorted_sess[0:state_num, :-2]
    s_reward = sorted_sess[0:state_num,-2:-1]
    all_state = np.concatenate([s_feature,s_reward],axis=1)
    all_label = sorted_sess[:state_num, -1:]

    state = np.zeros((1,2,max_time_length,all_state.shape[1]))
    all_state_len = np.zeros((1, 2))

    history_interact = all_state[:state_num,:]
    history_label = all_label[:state_num,0]
    positive_state = history_interact[np.where(history_label ==1.0)]
    negative_state = history_interact[np.where(history_label ==0.0)]
    state[0,0,:min(positive_state.shape[0], max_time_length),:] = positive_state[max(0, positive_state.shape[0] - max_time_length):positive_state.shape[0], :]
    state[0, 1, :min(negative_state.shape[0], max_time_length), :] = negative_state[max(0, negative_state.shape[
        0] - max_time_length):negative_state.shape[0], :]
    all_state_len[0, 0] = min(positive_state.shape[0], max_time_length)
    all_state_len[0, 1] = min(negative_state.shape[0], max_time_length)

    state_len = np.concatenate([all_state_len, all_state_len], axis=1)
    return state, state_len

def rank_sess_with_cate_hrl(model_output, ad_list, share_list, nature_list, name_map):
    lists = [ad_list, share_list, nature_list]
    real_cat = []
    ranked_items = []
    clicks = []
    ptr = [0, 0, 0]
    for one_cat in model_output:
        if ptr[one_cat] < len(lists[one_cat]):
            cur_item = lists[one_cat][ptr[one_cat]]
            ptr[one_cat] += 1
            real_cat.append(one_cat)
        elif ptr[0] < len(lists[0]):
            cur_item = lists[0][ptr[0]]
            ptr[0] += 1
            real_cat.append(0)
        elif ptr[1] < len(lists[1]):
            cur_item = lists[1][ptr[1]]
            ptr[1] += 1
            real_cat.append(1)
        else:
            cur_item = lists[2][ptr[2]]
            ptr[2] += 1
            real_cat.append(2)
        if "LABEL" in name_map:
            clicks.append(cur_item[name_map['LABEL']])
        ranked_items.append(cur_item)

    return real_cat, ranked_items


def eval_policy(agent, data_files,
                pool_worker=None, pool_process_n=15,
                redo_per_job=4, result_to_file=None,
                stage=-1, sample_first_sess=-1, scope_num_set=(3, 5, 10, 20),
                sort_position=False):
    if not isinstance(data_files, list):
        data_files = [data_files]
    name_map = {'uid': 0, 'u_age': 1, 'u_gender': 2, 'u_buy': 3, 'iid': 4, 'cat1d': 5, 'i_gender': 6, 'i_buy': 7, 'i_BC': 8,
            'price':18, 'label': -1}
    for k in name_map.keys():
        name_map[k]=name_map[k]-1
    name_map['label'] = -1

    skip_item_cnt = 0
    to_handle_item_cnt, cur_handle_item_cnt = 0, 0
    eval_sess_cnt = 0

    whole_results = list()
    time_info = defaultdict(float)
    lp_st = time()


    for hdf5_item in data_files:
        _st = time()
        f = open(hdf5_item, "rb")
        df_raw = pickle.load(f)

        to_handle_item_cnt += len(df_raw)
        time_info["read"] += time() - _st
        print("EVAL file:{} raw shape: {}".format(hdf5_item, to_handle_item_cnt))

        _st = time()
        time_info["sort"] += time() - _st

        _st = time()
        batch_max_n = pool_process_n
        eval_sess_val = list([None for k in range(batch_max_n)])
        cur_eval_history = list([None for k in range(batch_max_n)])
        d_cur_p = 0

        for sess_idx, value in enumerate(df_raw):

            eval_sess_cnt += 1
            cur_handle_item_cnt += len(value[2])

            if sess_idx + 1 != len(df_raw) and d_cur_p < batch_max_n:
                u_feat = np.array(value[1])
                i_feat = np.array(value[2])
                u_feat = np.expand_dims(u_feat, axis=0).repeat(i_feat.shape[0], axis=0)
                dense1 = np.array(value[3])
                dense2 = np.array(value[4])
                label = np.expand_dims(np.array(value[5]), axis=1)
                #print(u_feat.shape,i_feat.shape,dense2.shape,dense1.shape,label.shape)
                np_value = np.concatenate([u_feat,i_feat,dense1,dense2,label],axis=1)
                cur_eval_history[d_cur_p] = np.zeros((np_value.shape[0],np_value.shape[1]+1))
                eval_sess_val[d_cur_p] = np_value
                d_cur_p += 1
                continue
            time_info["append_data"] += time() - _st
            final_pred = [[]for k in range(d_cur_p)]
            seq_len = [eval_sess_val[k].shape[0] for k in range(d_cur_p)]

            for eval_pos in range(0,20):
                # predict 20 category in one session
                partial_job = partial(construct_state_hrl, cur_evaluate_length=eval_pos,max_time_length=10)
                batch_data = pool_worker.map(partial_job, cur_eval_history[:d_cur_p])
                batch_state = np.concatenate([b[0] for b in batch_data], axis=0)
                batch_len = np.concatenate([b[1] for b in batch_data], axis=0)
                #print(batch_state.shape)
                pre_actions = agent.predict(data=batch_state, state_len=batch_len)
                pre_actions = pre_actions.astype(int).tolist()
                for k in range(d_cur_p):
                    if eval_pos<seq_len[k]:
                        final_pred[k].append(pre_actions[k])
                cnt=0
                for sess_item, _pre in zip(eval_sess_val[:d_cur_p], final_pred):
                    if eval_pos<seq_len[cnt]:
                        init_ads, init_shares, init_natures = get_initial_ranking_in_one_session(sess_item, name_map)
                        real_cat, items = rank_sess_with_cate_hrl(_pre,init_ads,init_shares,init_natures,name_map)
                        cur_eval_history[cnt][eval_pos:eval_pos+1,:-2] = items[-1][:-1]
                        cur_eval_history[cnt][eval_pos:eval_pos+1,-1] = items[-1][name_map['label']]
                        cur_eval_history[cnt][eval_pos:eval_pos + 1, -2] = items[-1][name_map['label']]*items[-1][name_map['price']]
                    cnt+=1


            for sess_item, _pre in zip(eval_sess_val[:d_cur_p], final_pred):
                init_ads, init_shares, init_natures = get_initial_ranking_in_one_session(sess_item, name_map)
                for scope_num in scope_num_set:
                    ret = cal_one_session_criterion(_pre, init_ads, init_shares, init_natures, scope_num, name_map)
                    whole_results.append([stage, scope_num, *ret])
            time_info["calc_metric"] += time() - _st

            # early stopping eval with sample
            if 0 < sample_first_sess < eval_sess_cnt:
                break

            d_cur_p = 0
            _st = time()

    if skip_item_cnt > 0:
        print("Skip [{}] item within session len < {}".format(skip_item_cnt, conf.STEP_LEN))
    print("Eval <{}> session with sample: {} used time: {}".format(
        eval_sess_cnt, cur_handle_item_cnt, time() - lp_st))
    print("time detail: {}".format(dict(time_info)))

    return _metric_format(whole_results, scope_num_set, stage, result_to_file)


def _metric_format(whole_results, scope_num_set, stage, result_to_file):
    df_results = pd.DataFrame(
        whole_results,
        columns=["stage", "scope", "ndcg", "revenue", "real_ad_cnt", "real_share_cnt", "real_nature_cnt", "model_ad_cnt",
                 "model_share_cnt","model_nature_cnt","alpha_ndcg","alpha_revenue"],
    )

    summary_list = list()

    _format = "{:^20}\t" * 8
    f_format = "{:^20}\t{:^20}" + "\t{:^20.6f}" * 6
    print(_format.format("stage", "topK", "NDCG", "REVENUE", "real RATIO", "predict RATIO", "alpha NDCG", "alpha REVENUE"))
    for scope_num in scope_num_set:
        set_df = df_results[df_results.scope == scope_num]
        ndcg_val = set_df.ndcg.mean()
        revenue_val = set_df.revenue.mean()
        alpha_ndcg_val = set_df.alpha_ndcg.mean()
        alpha_revenue_val = set_df.alpha_revenue.mean()
        # if set_df.real_share_cnt.sum() < 1e-5:
        #     real_ratio = (set_df.real_ad_cnt.sum() / (set_df.real_share_cnt.sum() + 1e-5) +
        #                  set_df.real_share_cnt.sum() / (set_df.real_nature_cnt.sum() + 1e-5) +
        #                  set_df.real_nature_cnt.sum() / (set_df.real_ad_cnt.sum() + 1e-5))/float(3)
        # else:
        #     real_ratio = (set_df.real_ad_cnt.sum() / (set_df.real_share_cnt.sum()) +
        #                  set_df.real_share_cnt.sum() / (set_df.real_nature_cnt.sum()) +
        #                  set_df.real_nature_cnt.sum() / (set_df.real_ad_cnt.sum()))/float(3)

        # if set_df.model_share_cnt.sum() < 1e-5:
        #     pre_ratio = (set_df.model_ad_cnt.sum() / (set_df.model_share_cnt.sum() + 1e-5)+
        #                 set_df.model_share_cnt.sum() / (set_df.model_nature_cnt.sum() + 1e-5)+
        #                 set_df.model_nature_cnt.sum() / (set_df.model_ad_cnt.sum() + 1e-5))/float(3)
        # else:
        #     pre_ratio = (set_df.model_ad_cnt.sum() / (set_df.model_share_cnt.sum())+
        #                 set_df.model_share_cnt.sum() / (set_df.model_nature_cnt.sum())+
        #                 set_df.model_nature_cnt.sum() / (set_df.model_ad_cnt.sum()))/float(3)
        real_ad_cnt = set_df.real_ad_cnt.sum()
        real_share_cnt = set_df.real_share_cnt.sum()
        real_nature_cnt = set_df.real_nature_cnt.sum()
        real_arr = np.array([real_ad_cnt,real_share_cnt,real_nature_cnt],dtype=np.float)
        real_percentage = real_arr/np.sum(real_arr)
        real_ratio = np.sum(real_percentage*np.log(real_percentage/0.333))

        model_ad_cnt = set_df.model_ad_cnt.sum()
        model_share_cnt = set_df.model_share_cnt.sum()
        model_nature_cnt = set_df.model_nature_cnt.sum()
        model_arr = np.array([model_ad_cnt,model_share_cnt,model_nature_cnt],dtype=np.float)
        model_percentage = model_arr/np.sum(model_arr)
        pre_ratio = np.sum(model_percentage*np.log(model_percentage/0.333))


        print(f_format.format(stage, scope_num, ndcg_val, revenue_val, real_ratio, pre_ratio, alpha_ndcg_val,alpha_revenue_val))
        summary_list.append([stage, scope_num, ndcg_val, revenue_val, real_ratio, pre_ratio, alpha_ndcg_val, alpha_revenue_val])

    if result_to_file:
        if not os.path.exists(os.path.dirname(result_to_file)):
            os.makedirs(os.path.dirname(result_to_file))

        print("save results to file: {}".format(result_to_file))
        df_results.to_hdf(result_to_file, conf.default_pd_ds, mode="w")

    return whole_results, summary_list
