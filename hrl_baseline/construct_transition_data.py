import math
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

def construct_sarsd_hrl(sorted_sess, max_time_length=10,max_interaction_length=30):
    """sarsd with numpy."""
    uid = sorted_sess[0]
    u_feat = sorted_sess[1]
    i_feat = sorted_sess[2]
    dense1 = sorted_sess[3]
    dense2 = sorted_sess[4]
    label = sorted_sess[5]
    data_len = len(i_feat)
    reward = np.array(label)*(np.array(dense2)[:,2])
    reward = np.expand_dims(reward,axis=1)

    # only fetch ‘floor’ round number
    state_num = min(data_len,max_interaction_length)

    d_list = list([0 for _ in range(state_num - 1)]) + [1, ]
    user_feat = np.array(u_feat)
    user_feat = np.expand_dims(user_feat,axis=0).repeat(state_num,axis=0)
    s_feature = np.concatenate([user_feat,np.array(i_feat),np.array(dense1),np.array(dense2)],axis=1)
    all_state = np.concatenate([s_feature,reward],axis=1)
    all_action = np.array(i_feat)[:,1:2]
    all_reward = reward
    all_label = np.expand_dims(np.array(label),axis=1)

    state = np.zeros((state_num+1,2,max_time_length,all_state.shape[1]))
    action = np.zeros((state_num+1,all_action.shape[1]))
    reward = np.zeros((state_num+1,all_reward.shape[1]))
    all_state_len = np.zeros((state_num+1,2))  # positive and negative state len
    for step in range(1,state_num+1):
        history_interact = all_state[:step,:]
        history_label = all_label[:step,0]
        positive_state = history_interact[np.where(history_label ==1.0)]
        negative_state = history_interact[np.where(history_label ==0.0)]
        state[step,0,:min(positive_state.shape[0], max_time_length),:] = positive_state[max(0, positive_state.shape[0] - max_time_length):positive_state.shape[0], :]
        state[step, 1, :min(negative_state.shape[0], max_time_length), :] = negative_state[max(0, negative_state.shape[
            0] - max_time_length):negative_state.shape[0], :]
        all_state_len[step,0] = min(positive_state.shape[0], max_time_length)
        all_state_len[step,1] = min(negative_state.shape[0], max_time_length)
    for step in range(state_num):
        action[step, :] = all_action[step, :]
        reward[step, :] = all_reward[step, :]

    cur_state_len = all_state_len[:-1,:]
    next_state_len = all_state_len[1:,:]
    state_len = np.concatenate([cur_state_len,next_state_len],axis=1)

    return state[:-1], action[:-1], reward[:-1], state[1:], d_list, state_len
