import numpy as np
from rnn_version.utils import process_baseline_data, gen_initial_ranking, name_map
from collections import Counter
from common.sys_utils import get_reranked_list, cal_session_alpha_ndcg
import math



class base_algorithm(object):
    def __init__(self):
        pass

    def run(self):
        raise NotImplementedError

    def cal_criterion(self, cat, ad_list, share_list, nature_list, rerank_length, scope_number):
        cat = cat[:rerank_length]
        clicks, bids, ltvs = [], [], []
        lists = [ad_list, share_list, nature_list]
        ptr = [0, 0, 0]
        for one_cat in cat:
            cur_item = lists[one_cat][ptr[one_cat]]
            clicks.append(int(cur_item[name_map['label']]))
            bids.append(cur_item[name_map['price']])
            ptr[one_cat] += 1

        real_cat_cnt = Counter(cat[:scope_number])
        real_ad_cnt = real_cat_cnt[0]
        real_share_cnt = real_cat_cnt[1]
        real_nature_cnt = real_cat_cnt[2]

        # revenue and alpha revenue
        revenue = 0.0
        alpha_revenue = 0.0
        cat_acc = [0, 0, 0]
        for item in range(min(len(clicks), scope_number)):
            revenue += clicks[item] * bids[item]

            alpha_revenue += (clicks[item] * bids[item]) * pow(1 - 0.5,
                                                                                           cat_acc[cat[item]])
            cat_acc[cat[item]] += 1

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
        alpha_ndcg = cal_session_alpha_ndcg(clicks,cat,scope_number)
        return ndcg, revenue, real_ad_cnt, real_share_cnt,real_nature_cnt, alpha_ndcg, alpha_revenue

    def cal_evaluate_res(self, cri):
        # real_ratio = np.sum(cri[:, 2]) / np.sum(cri[:, 3]) if np.sum(cri[:, 3]) > 0 else np.inf
        # real_ratio += (np.sum(cri[:, 3]) / np.sum(cri[:, 4]) if np.sum(cri[:, 3]) > 0 else np.inf)
        # real_ratio += (np.sum(cri[:, 4]) / np.sum(cri[:, 2]) if np.sum(cri[:, 3]) > 0 else np.inf)
        # real_ratio = real_ratio / float(3)
        real_percentage = np.sum(cri[:,2:5],axis=0)/np.sum(cri[:,2:5],axis=(0,1))
        real_ratio = np.sum(real_percentage*np.log(real_percentage/0.333))
        print("Count of current pair: [{}, {}, {}]".format(np.sum(cri[:, 2]),np.sum(cri[:, 3]),np.sum(cri[:, 4])))
        return [np.mean(cri[:, 0]), np.mean(cri[:, 1]),
                real_ratio,
                np.mean(cri[:, 5]), np.mean(cri[:, 6])]

    def output_result(self, cri10, cri20):
        _format = "{:^20}\t" * 6
        f_format = "{:^20}\t" + "{:^20.6f}\t" * 5
        cri10, cri20 = np.array(cri10), np.array(cri20)
        res_10 = self.cal_evaluate_res(cri10)
        res_20 = self.cal_evaluate_res(cri20)
        print(_format.format("topK", "NDCG", "REVENUE", "real RATIO","alpha NDCG", "alpha REVENUE"))
        for res, top in zip([res_10, res_20], [10, 20]):
            print(f_format.format(top, res[0], res[1], res[2],res[3],res[4]))


class MMR(base_algorithm):
    def __init__(self, test_data,ad_lists,share_lists,nature_lists,pair=(0,1,2),top_k=30, lamb=0.5):
        base_algorithm.__init__(self)
        self.top_k = top_k
        self.lamb = lamb
        self.baseline_data = test_data
        self.ad_lists=ad_lists
        self.share_lists = share_lists
        self.nature_lists = nature_lists
        self.pair = pair

    def run(self):
        cri10, cri20 = [], []
        for session_idx in range(len(self.baseline_data)):
            data = self.baseline_data[session_idx]
            _ad_list = self.ad_lists[session_idx]
            _share_list = self.share_lists[session_idx]
            _nature_list = self.nature_lists[session_idx]
            valid_length = [len(_ad_list), len(_share_list), len(_nature_list)]
            rerank_length = min(self.top_k, len(data))
            res_cat = self.get_reranked_list_for_one_session(data, valid_length, rerank_length)
            tmp = self.cal_criterion(res_cat, _ad_list, _share_list, _nature_list, rerank_length, 10)
            cri10.append(tmp)
            tmp = self.cal_criterion(res_cat, _ad_list, _share_list, _nature_list, rerank_length, 20)
            cri20.append(tmp)
        self.output_result(cri10, cri20)

    def get_reranked_list_for_one_session(self, data, valid_length, rerank_length):
        if valid_length[0] == 0:
            return self.get_reranked_list_2cat(data, valid_length, rerank_length, (1, 2, 0))
        elif valid_length[1] == 0:
            return self.get_reranked_list_2cat(data, valid_length, rerank_length, (0, 2, 1))
        elif valid_length[2] == 0:
            return self.get_reranked_list_2cat(data, valid_length, rerank_length, (0, 1, 2))
        else:
            res_cat = []
            original_cat = data[:, name_map['cat1d']].astype(np.int)
            ad_idx = np.where(original_cat == self.pair[0])[0]
            share_idx = np.where(original_cat == self.pair[1])[0]
            nature_idx = np.where(original_cat == self.pair[2])[0]
            bid = data[:, name_map['price']].copy().astype(np.float)


            bid[np.isnan(bid)] = 0.0
            value = bid

            ptr = [0, 0, 0]

            # P(v|u) = value

            for i in range(rerank_length):
                cur_ad_sim1 = value[ad_idx[ptr[self.pair[0]]]]
                cur_share_sim1 = value[share_idx[ptr[self.pair[1]]]]
                cur_nature_sim1 = value[nature_idx[ptr[self.pair[2]]]]
                ad_sim2 = int(self.pair[0] in res_cat)
                share_sim2 = int(self.pair[1] in res_cat)
                nature_sim2 = int(self.pair[2] in res_cat)
                ad_score = self.lamb * cur_ad_sim1 - (1 - self.lamb) * ad_sim2
                share_score = self.lamb * cur_share_sim1 - (1 - self.lamb) * share_sim2
                nature_score = self.lamb * cur_nature_sim1 - (1-self.lamb) * nature_sim2
                chosen_cat = -1
                if ad_score >= share_score:
                    if ad_score>=nature_score:
                        chosen_cat = self.pair[0]
                    else:
                        chosen_cat = self.pair[2]
                else:
                    if share_score>=nature_score:
                        chosen_cat = self.pair[1]
                    else:
                        chosen_cat = self.pair[2]

                # cal real cat based on init list
                ptr[chosen_cat] += 1
                res_cat.append(chosen_cat)
                if ptr[chosen_cat] >= valid_length[chosen_cat]:
                    # full
                    res_cat += [self.pair[0]] * (valid_length[self.pair[0]] - ptr[self.pair[0]]) + [self.pair[1]] * (valid_length[self.pair[1]] - ptr[self.pair[1]]) + [self.pair[2]] * (
                            valid_length[self.pair[2]] - ptr[self.pair[2]])
                    break
            return res_cat

    def get_reranked_list_2cat(self, data, valid_length, rerank_length, pair):
        if valid_length[pair[0]] == 0:
            return [pair[1]] * valid_length[pair[1]]
        elif valid_length[pair[1]] == 0:
            return [pair[0]] * valid_length[pair[0]]
        else:
            res_cat = []
            original_cat = data[:, name_map['cat1d']].astype(np.int)
            ad_idx = np.where(original_cat == pair[0])[0]
            share_idx = np.where(original_cat == pair[1])[0]
            bid = data[:, name_map['price']].copy().astype(np.float)


            bid[np.isnan(bid)] = 0.0
            value = bid

            ptr = [0, 0, 0]

            # P(v|u) = value

            for i in range(rerank_length):
                cur_ad_sim1 = value[ad_idx[ptr[pair[0]]]]
                cur_share_sim1 = value[share_idx[ptr[pair[1]]]]
                ad_sim2 = int(pair[0] in res_cat)
                share_sim2 = int(pair[1] in res_cat)
                ad_score = self.lamb * cur_ad_sim1 - (1 - self.lamb) * ad_sim2
                share_score = self.lamb * cur_share_sim1 - (1 - self.lamb) * share_sim2
                chosen_cat = -1
                if ad_score >= share_score:
                    chosen_cat = pair[0]
                else:
                    chosen_cat = pair[1]

                # cal real cat based on init list
                ptr[chosen_cat] += 1
                res_cat.append(chosen_cat)
                if ptr[chosen_cat] >= valid_length[chosen_cat]:
                    # full
                    res_cat += [pair[0]] * (valid_length[pair[0]] - ptr[pair[0]]) + [pair[1]] * (valid_length[pair[1]] - ptr[pair[1]])
                    break
            return res_cat


class LinkedInDelta(base_algorithm):
    def __init__(self, test_data,ad_lists,share_lists,nature_lists, pair=(0,1,2), top_k=30, p=0.5):
        base_algorithm.__init__(self)
        self.top_k = top_k
        self.p = p
        self.detcons = True
        self.detrelaxed = False
        self.detconstsort=False
        self.baseline_data = test_data
        self.ad_lists=ad_lists
        self.share_lists = share_lists
        self.nature_lists = nature_lists
        self.pair = pair

    def run(self):
        cri10, cri20 = [], []
        for session_idx in range(len(self.baseline_data)):
            data = self.baseline_data[session_idx]
            _ad_list = self.ad_lists[session_idx]
            _share_list = self.share_lists[session_idx]
            _nature_list = self.nature_lists[session_idx]
            valid_length = [len(_ad_list), len(_share_list), len(_nature_list)]
            rerank_length = min(self.top_k, len(data))
            if self.detconstsort:
                res_cat=self.get_reranked_list_DetConstSort(data,valid_length,rerank_length)
            else:
                res_cat = self.get_reranked_list_for_one_session(data, valid_length, rerank_length)
            tmp = self.cal_criterion(res_cat, _ad_list, _share_list, _nature_list, rerank_length, 10)
            cri10.append(tmp)
            tmp = self.cal_criterion(res_cat, _ad_list, _share_list, _nature_list, rerank_length, 20)
            cri20.append(tmp)
        self.output_result(cri10, cri20)

    def get_reranked_list_for_one_session(self, data, valid_length, rerank_length):
        if valid_length[0] == 0:
            return self.get_reranked_list_2cat(data,valid_length,rerank_length,(1,2,0))
        elif valid_length[1] == 0:
            return self.get_reranked_list_2cat(data,valid_length,rerank_length,(0,2,1))
        elif valid_length[2] == 0:
            return self.get_reranked_list_2cat(data,valid_length,rerank_length,(0,1,2))
        else:
            res_cat = []
            original_cat = data[:, name_map['cat1d']].astype(np.int)
            ad_idx = np.where(original_cat == 0)[0]
            share_idx = np.where(original_cat == 1)[0]
            nature_idx = np.where(original_cat == 2)[0]
            bid = data[:, name_map['price']].copy().astype(np.float)

            bid[np.isnan(bid)] = 0.0
            value = bid

            ptr = [0, 0, 0]

            for i in range(rerank_length):
                k = i + 1
                below_min = []
                below_max = []
                for j in [0, 1, 2]:
                    if ptr[j] < math.floor(k * self.p):
                        below_min.append(j)
                    if ptr[j] >= math.floor(k * self.p) and ptr[j] < math.ceil(k * self.p):
                        below_max.append(j)
                if len(below_min) > 0:
                    if len(below_min) == 1:
                        chosen_cat = below_min[0]
                    else:
                        chosen_cat = np.argmax([value[ad_idx[ptr[0]]],value[share_idx[ptr[1]]],value[nature_idx[ptr[2]]]])
                        # if value[ad_idx[ptr[0]]] >= value[share_idx[ptr[1]]]:
                        #     chosen_cat = 0
                        # else:
                        #     chosen_cat = 1
                else:
                    if self.detcons:
                        #print(i,below_min,below_max,ptr)
                        chosen_cat = np.argmin([math.ceil(k * self.p) / self.p for j in below_max])
                    elif self.detrelaxed:
                        next_cat_set = below_max
                        if len(next_cat_set) == 1:
                            chosen_cat = below_max[0]
                        else:
                            chosen_cat = np.argmax(
                                [value[ad_idx[ptr[0]]], value[share_idx[ptr[1]]],value[nature_idx[ptr[2]]]])
                            # if value[ad_idx[ptr[0]]] >= value[share_idx[ptr[1]]]:
                            #     chosen_cat = 0
                            # else:
                            #     chosen_cat = 1
                ptr[chosen_cat] += 1
                res_cat.append(chosen_cat)
                if ptr[chosen_cat] >= valid_length[chosen_cat]:
                    # full
                    res_cat += [0] * (valid_length[0] - ptr[0]) + [1] * (valid_length[1] - ptr[1]) + [2] * (
                            valid_length[2] - ptr[2])
                    break
            return res_cat

    def get_reranked_list_DetConstSort(self, data, valid_length, rerank_length):
        if valid_length[0] == 0:
            return self.get_reranked_list_DetConstSort_2d(data,valid_length,rerank_length,(1,2,0))
        elif valid_length[1] == 0:
            return self.get_reranked_list_DetConstSort_2d(data,valid_length,rerank_length,(0,2,1))
        elif valid_length[2] == 0:
            return self.get_reranked_list_DetConstSort_2d(data,valid_length,rerank_length,(0,1,2))
        else:
            res_cat = []
            original_cat = data[:, name_map['cat1d']].astype(np.int)
            ad_idx = np.where(original_cat == 0)[0]
            share_idx = np.where(original_cat == 1)[0]
            nature_idx = np.where(original_cat == 2)[0]
            bid = data[:, name_map['price']].copy().astype(np.float)

            bid[np.isnan(bid)] = 0.0
            value = bid

            ptr = [0, 0, 0]

            mincounts = [0, 0, 0]
            maxindices = []
            ranked_score = []
            last_empty = 0
            k = 0
            while last_empty <= rerank_length:
                k += 1
                tmpmincounts = [math.floor(k * self.p) for j in [0, 1, 2]]
                changed_mins = []
                for j in [0, 1, 2]:
                    if mincounts[j] < tmpmincounts[j]:
                        changed_mins.append(j)
                if len(changed_mins) > 0:
                    if len(changed_mins) == 1:
                        ordchangedmins = [changed_mins[0]]
                    else:
                        ordchangedmins =  np.argsort([value[ad_idx[ptr[0]]],value[share_idx[ptr[1]]],value[nature_idx[ptr[2]]]])[::-1].tolist()

                    for one_cat in ordchangedmins:
                        res_cat.append(one_cat)
                        if one_cat == 0:
                            ranked_score.append(value[ad_idx[ptr[0]]])
                        elif one_cat==1:
                            ranked_score.append(value[share_idx[ptr[1]]])
                        else:
                            ranked_score.append(value[nature_idx[ptr[2]]])
                        maxindices.append(k)
                        start = last_empty
                        while start > 0 and maxindices[start - 1] >= start and ranked_score[start - 1] < ranked_score[
                            start]:
                            maxindices[start - 1], maxindices[start] = maxindices[start], maxindices[start - 1]
                            res_cat[start - 1], res_cat[start] = res_cat[start], res_cat[start - 1]
                            ranked_score[start - 1], ranked_score[start] = ranked_score[start], ranked_score[start - 1]
                            start -= 1
                        ptr[one_cat] += 1
                        last_empty += 1
                    mincounts = tmpmincounts
                if ptr[0] >= valid_length[0] or ptr[1] >= valid_length[1] or ptr[2] >= valid_length[2]:
                    res_cat += [0] * (valid_length[0] - ptr[0]) + [1] * (valid_length[1] - ptr[1]) + [2] * (
                            valid_length[2] - ptr[2])
                    break
            return res_cat

    def get_reranked_list_DetConstSort_2d(self, data, valid_length, rerank_length, pair):
        if valid_length[pair[0]] == 0:
            return [pair[1]] * valid_length[pair[1]]
        elif valid_length[pair[1]] == 0:
            return [pair[0]] * valid_length[pair[0]]
        else:
            res_cat = []
            original_cat = data[:, name_map['cat1d']].astype(np.int)
            ad_idx = np.where(original_cat == pair[0])[0]
            share_idx = np.where(original_cat == pair[1])[0]
            bid = data[:, name_map['price']].copy().astype(np.float)

            bid[np.isnan(bid)] = 0.0
            value = bid

            ptr = [0, 0, 0]

            mincounts = [0, 0, 0]
            maxindices = []
            ranked_score = []
            last_empty = 0
            k = 0
            while last_empty <= rerank_length:
                k += 1
                tmpmincounts = [0,0,0]
                tmpmincounts[pair[0]] = math.floor(k * self.p)
                tmpmincounts[pair[1]] = math.floor(k * self.p)
                changed_mins = []
                for j in [pair[0], pair[1]]:
                    if mincounts[j] < tmpmincounts[j]:
                        changed_mins.append(j)
                if len(changed_mins) > 0:
                    if len(changed_mins) == 1:
                        ordchangedmins = [changed_mins[0]]
                    else:
                        ordchangedmins = [pair[0], pair[1]] if value[ad_idx[ptr[pair[0]]]] >= value[share_idx[ptr[pair[1]]]] else [pair[1], pair[0]]

                    for one_cat in ordchangedmins:
                        res_cat.append(one_cat)
                        if one_cat == pair[0]:
                            ranked_score.append(value[ad_idx[ptr[pair[0]]]])
                        else:
                            ranked_score.append(value[share_idx[ptr[pair[1]]]])
                        maxindices.append(k)
                        start = last_empty
                        while start > 0 and maxindices[start - 1] >= start and ranked_score[start - 1] < ranked_score[
                            start]:
                            maxindices[start - 1], maxindices[start] = maxindices[start], maxindices[start - 1]
                            res_cat[start - 1], res_cat[start] = res_cat[start], res_cat[start - 1]
                            ranked_score[start - 1], ranked_score[start] = ranked_score[start], ranked_score[start - 1]
                            start -= 1
                        ptr[one_cat] += 1
                        last_empty += 1
                    mincounts = tmpmincounts
                if ptr[pair[0]] >= valid_length[pair[0]] or ptr[pair[1]] >= valid_length[pair[1]]:
                    res_cat += [pair[0]] * (valid_length[pair[0]] - ptr[pair[0]]) + [pair[1]] * (valid_length[pair[1]] - ptr[pair[1]])
                    break
            return res_cat

    def get_reranked_list_2cat(self, data, valid_length, rerank_length,pair):
        if valid_length[pair[0]] == 0:
            return [pair[1]] * valid_length[pair[1]]
        elif valid_length[pair[1]] == 0:
            return [pair[0]] * valid_length[pair[0]]
        else:
            res_cat = []
            original_cat = data[:, name_map['cat1d']].astype(np.int)
            ad_idx = np.where(original_cat == pair[0])[0]
            share_idx = np.where(original_cat == pair[1])[0]
            bid = data[:, name_map['price']].copy().astype(np.float)

            bid[np.isnan(bid)] = 0.0
            value = bid

            ptr = [0, 0, 0]

            for i in range(rerank_length):
                k = i + 1
                below_min = []
                below_max = []
                for j in [pair[0], pair[1]]:
                    if ptr[j] < math.floor(k * self.p):
                        below_min.append(j)
                    if ptr[j] >= math.floor(k * self.p) and ptr[j] < math.ceil(k * self.p):
                        below_max.append(j)
                if len(below_min) > 0:
                    if len(below_min) == 1:
                        chosen_cat = below_min[0]
                    else:
                        if value[ad_idx[ptr[pair[0]]]] >= value[share_idx[ptr[pair[1]]]]:
                            chosen_cat = pair[0]
                        else:
                            chosen_cat = pair[1]
                else:
                    if self.detcons:
                        chosen_cat = pair[0]#np.argmin([math.ceil(k * self.p) / self.p for j in below_max])
                    elif self.detrelaxed:
                        next_cat_set = below_max
                        if len(next_cat_set) == 1:
                            chosen_cat = below_max[0]
                        else:
                            if value[ad_idx[ptr[pair[0]]]] >= value[share_idx[ptr[pair[1]]]]:
                                chosen_cat = pair[0]
                            else:
                                chosen_cat = pair[1]
                ptr[chosen_cat] += 1
                res_cat.append(chosen_cat)
                if ptr[chosen_cat] >= valid_length[chosen_cat]:
                    # full
                    res_cat += [pair[0]] * (valid_length[pair[0]] - ptr[pair[0]]) + [pair[1]] * (valid_length[pair[1]] - ptr[pair[1]])
                    break
            return res_cat
