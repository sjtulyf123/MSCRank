"""Utils for system, and some utils sync from rnn common."""
import os
import glob
import numpy as np
from collections import Counter, defaultdict

def rank_sess_with_cate(model_output, ad_list, share_list, nature_list, name_map, share_first=False):
    lists = [ad_list, share_list, nature_list]
    real_cat = []
    clicks = []
    bids = []
    ptr = [0, 0, 0]
    if share_first==False:
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
            clicks.append(cur_item[name_map['label']])
            bids.append(cur_item[name_map['price']])
            #bids.append(1.0)
    else:
        for one_cat in model_output:
            if ptr[one_cat] < len(lists[one_cat]):
                cur_item = lists[one_cat][ptr[one_cat]]
                ptr[one_cat] += 1
                real_cat.append(one_cat)
            elif ptr[1] < len(lists[1]):
                cur_item = lists[1][ptr[1]]
                ptr[1] += 1
                real_cat.append(1)
            elif ptr[2] < len(lists[2]):
                cur_item = lists[2][ptr[2]]
                ptr[2] += 1
                real_cat.append(2)
            else:
                cur_item = lists[0][ptr[0]]
                ptr[0] += 1
                real_cat.append(0)
            clicks.append(cur_item[name_map['label']])
            bids.append(cur_item[name_map['price']])
            #bids.append(1.0)

    return real_cat, clicks, bids

def cal_one_session_criterion(model_output, ad_list, share_list, nature_list,
                              scope_number, name_map, share_first=False):
    real_cat, clicks, bids = rank_sess_with_cate(model_output, ad_list, share_list, nature_list, name_map, share_first)

    # count category
    model_cat_cnt = Counter(model_output[:scope_number])
    model_ad_cnt = model_cat_cnt[0]
    model_share_cnt = model_cat_cnt[1]
    model_nature_cnt = model_cat_cnt[2]
    real_cat_cnt = Counter(real_cat[:scope_number])
    real_ad_cnt = real_cat_cnt[0]
    real_share_cnt = real_cat_cnt[1]
    real_nature_cnt = real_cat_cnt[2]

    # revenue and alpha revenue
    revenue = 0.0
    alpha_revenue = 0.0
    cat_acc = [0,0,0]
    for item in range(min(len(clicks), scope_number)):
        revenue += clicks[item] * bids[item]

        alpha_revenue += (clicks[item] * bids[item]) * pow(1-0.5,cat_acc[real_cat[item]])
        cat_acc[real_cat[item]]+=1

    # ndcg
    if len(clicks) < 2:
        ndcg = sum(clicks)
    else:
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
        ndcg = float(dcg) / ideal_dcg if ideal_dcg != 0 else 0.
    alpha_ndcg = cal_session_alpha_ndcg(clicks,real_cat,scope_number)
    return ndcg, revenue, real_ad_cnt, real_share_cnt, real_nature_cnt, model_ad_cnt, model_share_cnt, model_nature_cnt, alpha_ndcg, alpha_revenue


def get_initial_ranking_in_one_session(session, name_map):
    # generate initial ranking lists
    _ad_list = session[np.where(session[:,name_map['cat1d']]==0)]
    _share_list = session[np.where(session[:, name_map['cat1d']] == 1)]
    _nature_list = session[np.where(session[:, name_map['cat1d']] == 2)]
    assert len(_ad_list)+len(_share_list)+len(_nature_list) == len(session)

    return _ad_list, _share_list, _nature_list


def get_reranked_list(preds):
    return sorted(range(len(preds)), key=lambda k: preds[k], reverse=True)

def cal_session_alpha_ndcg(clicks, cats, scope_number,alpha=0.5):
    """
    cal alpha NDCG in one session
    """
    if len(clicks) < 2:
        alpha_ndcg = sum(clicks)
    else:
        final = list(range(len(clicks)))
        gold = get_reranked_list(clicks)
        ideal_alpha_dcg = 0
        alpha_dcg = 0
        # define scope for calculation
        scope_final = final[:scope_number]
        scope_gold = gold[:scope_number]
        final_cat_acc = [0,0,0]
        gold_cat_acc = [0,0,0]
        for _i, _f, _g in zip(range(1, scope_number + 1), scope_final, scope_gold):
            alpha_dcg += (clicks[_f]*pow(1-alpha,final_cat_acc[cats[_f]])) / (np.log2(_i + 1))
            ideal_alpha_dcg += (clicks[_g]*pow(1-alpha,final_cat_acc[cats[_g]])) / (np.log2(_i + 1))
            final_cat_acc[cats[_f]] += 1
            gold_cat_acc[cats[_g]] += 1
        alpha_ndcg = float(alpha_dcg) / ideal_alpha_dcg if ideal_alpha_dcg != 0 else 0.
    return alpha_ndcg
