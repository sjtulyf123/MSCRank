import os
import pickle
import pickle as pkl
from datetime import date
import random
from collections import defaultdict, Counter
import numpy as np
import time



def process_data(raw_dir1, raw_dir2, store_dir, chosen_cat=[3,29,37]):
    assert len(chosen_cat)==3
    cur_time = time.time()
    fin1 = open(raw_dir1, 'r')
    records=fin1.readlines()
    fin2 = open(raw_dir2, 'r')
    records.extend(fin2.readlines())
    file_length = len(records)
    print(file_length)
    print('finish loading data')
    print(time.time() - cur_time)
    fin1.close()
    fin2.close()

    with_null = 0
    without_chosen_cat = 0
    no_click = 0

    cur_time = time.time()

    user_map = {}
    user_feat_map = [{},{},{}]
    item_feat_map = [{},{},{},{},{}]
    item_feat_map[1]={3:0,29:1,37:2}
    user_map_cnt=0
    user_feat_map_cnt = [0,0,0]
    item_feat_map_cnt = [0,0,0,0,0]

    train_save_cnt = 0
    test_save_cnt = 0

    cur_train_data = []
    cur_train_length = 0
    cur_test_data = []
    cur_test_length = 0
    length_to_change_file = file_length//10
    max_list_length = 0

    for i, v in enumerate(records):
        if v.find('null') != -1:
            with_null += 1
            continue
        piece_data = v.strip().split('|')
        item_feat = eval(piece_data[2])
        cur_cats = np.array(item_feat)[:, 1]
        valid_idx = np.where((cur_cats==chosen_cat[0])|(cur_cats==chosen_cat[1])|(cur_cats==chosen_cat[2]))[0].tolist()
        if len(valid_idx)<1:
            # dont have chosen cat in one list
            without_chosen_cat+=1
            continue
        max_list_length = max(len(valid_idx),max_list_length)
        user = eval(piece_data[0])
        user_feat = eval(piece_data[1])
        item_feat = np.array(item_feat)[valid_idx,:].tolist()
        dense_feat1 = np.array(eval(piece_data[3]))[valid_idx,:].tolist()
        dense_feat2 = np.array(eval(piece_data[4]))[valid_idx,:].tolist()
        labels = np.array(eval(piece_data[5]))[valid_idx].tolist()
        if sum(labels)<1:
            no_click+=1
            continue
        # remap id
        if user not in user_map.keys():
            user_map_cnt+=1
            user_map[user]=user_map_cnt
        user = user_map[user]
        for idx,feat in enumerate(user_feat):
            if feat not in user_feat_map[idx].keys():
                user_feat_map_cnt[idx]+=1
                user_feat_map[idx][feat]=user_feat_map_cnt[idx]
            user_feat[idx] = user_feat_map[idx][feat]
        for l in range(len(item_feat)):
            for idx,feat in enumerate(item_feat[l]):
                if feat not in item_feat_map[idx].keys():
                    item_feat_map_cnt[idx]+=1
                    item_feat_map[idx][feat]=item_feat_map_cnt[idx]
                item_feat[l][idx] = item_feat_map[idx][feat]

        # save to train or test
        if random.random()<0.2:
            cur_test_data.append([user,user_feat,item_feat,dense_feat1,dense_feat2,labels])
            cur_test_length+=1
            if cur_test_length>=length_to_change_file:
                test_save_cnt+=1
                with open(os.path.join(store_dir,"test_part{}.pkl".format(test_save_cnt)),"wb") as f:
                    pkl.dump(cur_test_data, f, protocol=4)
                    print("Save to test_part{}.pkl with length {}.".format(test_save_cnt,cur_test_length))
                cur_test_data = []
                cur_test_length = 0
        else:
            cur_train_data.append([user, user_feat, item_feat, dense_feat1, dense_feat2,labels])
            cur_train_length += 1
            if cur_train_length >= length_to_change_file:
                train_save_cnt += 1
                with open(os.path.join(store_dir, "train_part{}.pkl".format(train_save_cnt)), "wb") as f:
                    pkl.dump(cur_train_data, f, protocol=4)
                    print("Save to train_part{}.pkl with length {}.".format(train_save_cnt,cur_train_length))
                cur_train_data = []
                cur_train_length = 0
    if cur_train_length>0:
        train_save_cnt += 1
        with open(os.path.join(store_dir, "train_part{}.pkl".format(train_save_cnt)), "wb") as f:
            pkl.dump(cur_train_data, f, protocol=4)
            print("Save to train_part{}.pkl with length {}.".format(train_save_cnt, cur_train_length))
    if cur_test_length>0:
        test_save_cnt += 1
        with open(os.path.join(store_dir, "test_part{}.pkl".format(test_save_cnt)), "wb") as f:
            pkl.dump(cur_test_data, f, protocol=4)
            print("Save to test_part{}.pkl with length {}.".format(test_save_cnt, cur_test_length))
    #print(user_map, user_feat_map, item_feat_map)
    with open(os.path.join(store_dir, "map_dicts.pkl"), "wb") as f:
        pkl.dump([user_map,user_feat_map,item_feat_map], f, protocol=4)
    print(time.time() - cur_time)
    print("null: {}, invalid: {}, no click: {}".format(with_null,without_chosen_cat,no_click))
    print(max_list_length)


if __name__ == '__main__':
    # parameters
    random.seed(8888)
    np.random.seed(8888)
    data_dir = 'prm_data'
    save_dir = 'prm_data'
    max_hist_len = 50
    raw_dir1 = os.path.join(data_dir, 'raw_data/set1.train.txt.part1')
    raw_dir2 = os.path.join(data_dir, 'raw_data/set1.train.txt.part2')
    processed_dir = os.path.join(save_dir, 'clean_data')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    process_data(raw_dir1, raw_dir2, processed_dir)
