import os
import random
import time
import glob
import pandas as pd
import numpy as np
import argparse
import tensorflow as tf
import pprint
import pickle


from rnn_version.model import TwoLevelRNN
from rnn_version.utils import get_batch, gen_data, name_map, padding_all_data, padding_test_data, \
    process_baseline_data, cal_baseline_criterion, save_padded_data, load_padded_data, print_baselines, \
    get_default_config

from common.sys_utils import cal_one_session_criterion
from multiprocessing import Pool


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--reg_lambda', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=8888)
    parser.add_argument('--label_decay', type=float, default=0.9)

    parser.add_argument('--embedding_dim', type=int, default=64, help='embedding dimension of input')
    parser.add_argument('--feature_size', type=int, default=3381)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--window_size', type=int, default=30,
                        help='slide window size when performing data augmentation')
    parser.add_argument('--keep_prob', type=float, default=0.8)
    parser.add_argument('--sample_pair_num', type=int, default=5,
                        help='sample how many pos and neg pairs for rank task')

    parser.add_argument('--cat_rnn_type', default='gru')
    parser.add_argument('--item_rnn_type', default='gru')
    parser.add_argument('--interaction_mode', default='one_to_one')
    parser.add_argument('--multi_task', action='store_true')
    parser.add_argument('--use_2level_category', action='store_true')
    parser.add_argument('--use_3level_category', action='store_true')
    parser.add_argument('--optimize_ratio', action='store_true')
    parser.add_argument('--use_l2_regular', action='store_true')
    parser.add_argument('--use_new_rnn', action='store_true')
    parser.add_argument('--use_initlist_attention', action='store_true')
    parser.add_argument('--use_js', action='store_true')
    parser.add_argument('--kl_weight', type=float, default=1.0)
    parser.add_argument('--ranker_weight', type=float, default=1.0)

    parser.add_argument('--evaluate_cri_number', type=int, default=20)
    parser.add_argument('--only_store_best', action='store_true')
    parser.add_argument('--revenue_decrease_bound', type=float, default=0.03)
    parser.add_argument('--file_path', default='prm_data')
    parser.add_argument('--result_path', default='exp_logs/test')
    parser.add_argument('--load_pb_path', default='exp_logs/test')
    parser.add_argument('--online_output', type=str, help="output for online")
    parser.add_argument('--ideal_ratio', type=float, default=1.0)

    parser.add_argument('--update_padded_data', action='store_true',
                        help='whether update padded data for training (need set True when raw data is updated)')
    parser.add_argument('--padded_file_path', type=str, default='padded_feature',
                        help='padded feature save path, under file_dir')
    parser.add_argument('--behavior_length',type=int,default=0,help='add length when use multi-behavior, set 0 when use length')

    parser.add_argument('--used_train_days', type=int, default=100)
    parser.add_argument('--data_list_id', type=str, default='cds_111124073')
    parser.add_argument('--use_mtp_config_file', action='store_true')

    parser.add_argument('--evaluate_times_per_epoch', type=int, default=5)
    parser.add_argument('--soft_ranker_tau', type=float, default=1.0)
    parser.add_argument('--gpu_num', type=str, default='0')
    parser.add_argument('--share_first', action='store_true')
    return parser.parse_args()


def evaluate_original_list(file_list, parallel_worker, evaluate_number=20):
    data = process_baseline_data(file_list,parallel_worker)
    res_ndcg, res_revenue, res_ratio,_,_ = cal_baseline_criterion(data, evaluate_number)
    return res_revenue, res_ratio


def cal_evaluate_res(cri):
    # real_ratio = np.sum(cri[:, 2]) / np.sum(cri[:, 3]) if np.sum(cri[:, 3]) > 0 else np.inf
    # real_ratio += (np.sum(cri[:, 3]) / np.sum(cri[:, 4]) if np.sum(cri[:, 4]) > 0 else np.inf)
    # real_ratio += (np.sum(cri[:, 4]) / np.sum(cri[:, 2]) if np.sum(cri[:, 2]) > 0 else np.inf)
    # model_ratio = np.sum(cri[:, 5]) / np.sum(cri[:, 6]) if np.sum(cri[:, 6]) > 0 else np.inf
    # model_ratio += (np.sum(cri[:, 6]) / np.sum(cri[:, 7]) if np.sum(cri[:, 7]) > 0 else np.inf)
    # model_ratio += (np.sum(cri[:, 7]) / np.sum(cri[:, 5]) if np.sum(cri[:, 5]) > 0 else np.inf)
    # real_ratio = real_ratio/float(3)
    # model_ratio = model_ratio/float(3)
    real_percentage = np.sum(cri[:,2:5],axis=0)/np.sum(cri[:,2:5],axis=(0,1))
    real_ratio = np.sum(real_percentage*np.log(real_percentage/0.333))
    model_percentage = np.sum(cri[:,5:8],axis=0)/np.sum(cri[:,5:8],axis=(0,1))
    model_ratio = np.sum(model_percentage*np.log(model_percentage/0.333))
    return [np.mean(cri[:, 0]), np.mean(cri[:, 1]),
            real_ratio,
            model_ratio,
            np.mean(cri[:, 8]),np.mean(cri[:, 9]),
            np.sum(cri[:, 2]),np.sum(cri[:, 3]),np.sum(cri[:, 4])]


def add_to_monitor(monitor, data_3, data_5, data_10, data_20):
    monitor['ndcg_3'].append(data_3[0])
    monitor['revenue_3'].append(data_3[1])
    monitor['real_ratio_3'].append(data_3[2])
    monitor['model_ratio_3'].append(data_3[3])
    monitor['alpha_ndcg_3'].append(data_3[4])
    monitor['alpha_revenue_3'].append(data_3[5])
    monitor['ad_cnt_3'].append(data_3[6])
    monitor['share_cnt_3'].append(data_3[7])
    monitor['nature_cnt_3'].append(data_3[8])
    monitor['ndcg_5'].append(data_5[0])
    monitor['revenue_5'].append(data_5[1])
    monitor['real_ratio_5'].append(data_5[2])
    monitor['model_ratio_5'].append(data_5[3])
    monitor['alpha_ndcg_5'].append(data_5[4])
    monitor['alpha_revenue_5'].append(data_5[5])
    monitor['ad_cnt_5'].append(data_5[6])
    monitor['share_cnt_5'].append(data_5[7])
    monitor['nature_cnt_5'].append(data_5[8])
    monitor['ndcg_10'].append(data_10[0])
    monitor['revenue_10'].append(data_10[1])
    monitor['real_ratio_10'].append(data_10[2])
    monitor['model_ratio_10'].append(data_10[3])
    monitor['alpha_ndcg_10'].append(data_10[4])
    monitor['alpha_revenue_10'].append(data_10[5])
    monitor['ad_cnt_10'].append(data_10[6])
    monitor['share_cnt_10'].append(data_10[7])
    monitor['nature_cnt_10'].append(data_10[8])
    monitor['ndcg_20'].append(data_20[0])
    monitor['revenue_20'].append(data_20[1])
    monitor['real_ratio_20'].append(data_20[2])
    monitor['model_ratio_20'].append(data_20[3])
    monitor['alpha_ndcg_20'].append(data_20[4])
    monitor['alpha_revenue_20'].append(data_20[5])
    monitor['ad_cnt_20'].append(data_20[6])
    monitor['share_cnt_20'].append(data_20[7])
    monitor['nature_cnt_20'].append(data_20[8])


def evaluate(params_dict, model, sess, data_source, seq_len,padded_init_list, ad_lists, share_lists, nature_lists):
    in_reg_lambda = params_dict['reg_lambda']
    batch_size = params_dict['batch_size']
    share_first = params_dict['share_first']
    assert share_first in [True,False]
    losses = []
    kl_losses = []
    ranker_losses = []
    model_output_cat = []

    session_num = len(data_source)
    batch_num = session_num // batch_size if session_num % batch_size == 0 else (session_num // batch_size + 1)

    cri3, cri5, cri10, cri20 = [], [], [], []

    for batch_no in range(batch_num):
        data_batch, sub_seq_len,batch_init_list = get_batch(data_source, seq_len,padded_init_list, batch_size=batch_size, batch_no=batch_no)
        sub_ad_lists, _,_ = get_batch(ad_lists, seq_len, padded_init_list,batch_size=batch_size, batch_no=batch_no)
        sub_share_lists, _,_ = get_batch(share_lists, seq_len,padded_init_list, batch_size=batch_size, batch_no=batch_no)
        sub_nature_lists, _,_ = get_batch(nature_lists, seq_len,padded_init_list, batch_size=batch_size, batch_no=batch_no)

        pred, label, loss, kl_loss, ranker_loss = model.eval(sess, data_batch, sub_seq_len,batch_init_list, in_reg_lambda,
                                                             keep_prob=1)
        for session_idx in range(len(sub_seq_len)):
            session_seq_len = sub_seq_len[session_idx]
            _pred = pred[session_idx]

            chosen_cat=_pred[:session_seq_len].astype(int).tolist()
            #print(chosen_cat)
            tmp = cal_one_session_criterion(chosen_cat, sub_ad_lists[session_idx], sub_share_lists[session_idx],
                                            sub_nature_lists[session_idx], 3, name_map,share_first)
            cri3.append(tmp)
            tmp = cal_one_session_criterion(chosen_cat, sub_ad_lists[session_idx], sub_share_lists[session_idx],
                                            sub_nature_lists[session_idx], 5, name_map,share_first)
            cri5.append(tmp)
            tmp = cal_one_session_criterion(chosen_cat, sub_ad_lists[session_idx], sub_share_lists[session_idx],
                                            sub_nature_lists[session_idx], 10, name_map,share_first)
            cri10.append(tmp)
            tmp = cal_one_session_criterion(chosen_cat, sub_ad_lists[session_idx], sub_share_lists[session_idx],
                                            sub_nature_lists[session_idx], 20, name_map,share_first)
            cri20.append(tmp)
            model_output_cat.append(chosen_cat)

        losses.append(loss)
        kl_losses.append(kl_loss)
        ranker_losses.append(ranker_loss)
    cri3, cri5, cri10, cri20 = np.array(cri3), np.array(cri5), np.array(cri10), np.array(cri20)
    final_loss = sum(losses) / len(losses)
    final_kl_loss = sum(kl_losses) / len(kl_losses)
    final_ranker_loss = sum(ranker_losses) / len(ranker_losses)
    res_3 = cal_evaluate_res(cri3)
    res_5 = cal_evaluate_res(cri5)
    res_10 = cal_evaluate_res(cri10)
    print(np.sum(cri10[:, 2]),np.sum(cri10[:, 3]),np.sum(cri10[:, 4]),np.sum(cri10[:, 5]),np.sum(cri10[:, 6]),np.sum(cri10[:, 7]))
    print(np.sum(cri20[:, 2]), np.sum(cri20[:, 3]), np.sum(cri20[:, 4]), np.sum(cri20[:, 5]), np.sum(cri20[:, 6]),
          np.sum(cri20[:, 7]))
    res_20 = cal_evaluate_res(cri20)
    return model_output_cat, final_loss, final_kl_loss, final_ranker_loss, res_3, res_5, res_10, res_20


def train(args):
    print(pprint.pformat(args, indent=0, width=1, ))
    _format = "{:^20}\t" * 8
    f_format = "{:^20}\t{:^20}" + "\t{:^20.6f}" * 6

    if args.use_mtp_config_file:
        params_dict=get_default_config()
    else:
        params_dict = vars(args)
    print("exp params:{}".format(params_dict))
    random.seed(params_dict['seed'])
    tf.set_random_seed(params_dict['seed'])
    np.random.seed(params_dict['seed'])

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    if not os.path.exists(os.path.join(args.result_path, 'models')):
        os.makedirs(os.path.join(args.result_path, 'models'))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    gpu_options = tf.GPUOptions(allow_growth=True)
    parallel_worker=None

    tf.reset_default_graph()

    # read feature size
    f = open(os.path.join(args.file_path,"clean_data","map_dicts.pkl"),"rb")
    map_dicts = pickle.load(f)
    feature_size = [len(map_dicts[1][0].keys())+1,len(map_dicts[1][1].keys())+1,len(map_dicts[1][2].keys())+1,
                    len(map_dicts[2][0].keys())+1,len(map_dicts[2][1].keys())+1,len(map_dicts[2][2].keys())+1,len(map_dicts[2][3].keys())+1,
                    len(map_dicts[2][4].keys())+1]
    print(feature_size)

    assert params_dict['behavior_length'] in [0, 12]

    model = TwoLevelRNN(name_map, feature_size, params_dict['embedding_dim'], params_dict['hidden_size'], params_dict['window_size'],
                        params_dict['cat_rnn_type'], params_dict['item_rnn_type'], params_dict['interaction_mode'], params_dict['multi_task'],
                        params_dict['use_2level_category'], params_dict['use_3level_category'], params_dict['optimize_ratio'], params_dict['use_l2_regular'],
                        params_dict['use_new_rnn'], params_dict['use_initlist_attention'],params_dict['use_js'],params_dict['ideal_ratio'],
                        params_dict['kl_weight'], params_dict['sample_pair_num'], params_dict['ranker_weight'], params_dict['behavior_length'],params_dict['soft_ranker_tau'])

    training_monitor = {
        'evaluate_step': [],
        'train_loss': [],
        'test_loss': [],
        'train_kl_loss': [],
        'test_kl_loss': [],
        'train_ranker_loss': [],
        'test_ranker_loss': [],
        'ndcg_3': [],
        'revenue_3': [],
        'real_ratio_3': [],
        'model_ratio_3': [],
        'alpha_ndcg_3':[],
        'alpha_revenue_3': [],
        'ad_cnt_3': [],
        'share_cnt_3': [],
        'nature_cnt_3': [],
        'ndcg_5': [],
        'revenue_5': [],
        'real_ratio_5': [],
        'model_ratio_5': [],
        'alpha_ndcg_5': [],
        'alpha_revenue_5': [],
        'ad_cnt_5': [],
        'share_cnt_5': [],
        'nature_cnt_5': [],
        'ndcg_10': [],
        'revenue_10': [],
        'real_ratio_10': [],
        'model_ratio_10': [],
        'alpha_ndcg_10': [],
        'alpha_revenue_10': [],
        'ad_cnt_10': [],
        'share_cnt_10': [],
        'nature_cnt_10': [],
        'ndcg_20': [],
        'revenue_20': [],
        'real_ratio_20': [],
        'model_ratio_20': [],
        'alpha_ndcg_20': [],
        'alpha_revenue_20': [],
        'ad_cnt_20': [],
        'share_cnt_20': [],
        'nature_cnt_20': [],
    }
    file_lists = glob.glob(
        os.path.join(args.file_path,"clean_data", "train_part*.pkl"))
    file_lists = sorted(file_lists, reverse=False)

    test_file_list = glob.glob(
        os.path.join(args.file_path,"clean_data", "test_part*.pkl"))
    test_file_list = sorted(test_file_list, reverse=False)

    # evaluate original baseline for saving best model
    #original_revenue, original_ratio = evaluate_original_list(test_file_list, parallel_worker,params_dict['evaluate_cri_number'])
    # d = print_baselines(test_file_list,parallel_worker)
    # print(d)
    original_revenue = 0.0339
    original_ratio = 0.373573
    min_gap = 100.0

    if params_dict['update_padded_data']:
        train_file_list = file_lists
        file_cnt=0
        for one_file in train_file_list:
            file_cnt+=1
            train_data, _, _, _, init_list_data = gen_data([one_file], parallel_worker,params_dict['window_size'], data_mode='train', shuffle=True)
            cur_time = time.time()
            padded_train_data, seq_len, padding_init_list_train_data = padding_all_data(train_data,init_list_data, params_dict['window_size'], behavior_length=params_dict['behavior_length'])
            print("padding time: {}".format(time.time() - cur_time))
            save_padded_data(padded_train_data, os.path.join(args.file_path, args.padded_file_path),
                             'feature_{}.pkl'.format(file_cnt))
            save_padded_data(np.array(seq_len), os.path.join(args.file_path, args.padded_file_path),
                             'seqlen_{}.pkl'.format(file_cnt))
            save_padded_data(padding_init_list_train_data, os.path.join(args.file_path, args.padded_file_path),
                             'init_list_{}.pkl'.format(file_cnt))

    if not os.path.exists(os.path.join(args.file_path, args.padded_file_path)):
        raise FileExistsError("Padded files are not exist. Need generate padded files first.")

    padded_file_lists = glob.glob(
        os.path.join(args.file_path, args.padded_file_path, "feature_*.pkl"))
    padded_file_lists = sorted(padded_file_lists, reverse=False)
    seqlen_file_lists = glob.glob(
        os.path.join(args.file_path, args.padded_file_path, "seqlen_*.pkl"))
    seqlen_file_lists = sorted(seqlen_file_lists, reverse=False)
    init_list_file_lists = glob.glob(
        os.path.join(args.file_path, args.padded_file_path, "init_list_*.pkl"))
    init_list_file_lists = sorted(init_list_file_lists, reverse=False)

    print("train files: {}".format(padded_file_lists))
    print("test files: {}".format(test_file_list))

    total_session_num = 0
    for l in seqlen_file_lists:
        lenn = load_padded_data(l)
        total_session_num += lenn.shape[0]

    test_data, ad_lists, share_lists, nature_lists, _ = gen_data(test_file_list, parallel_worker, params_dict['window_size'], data_mode='test')
    padded_test_data, test_seq_len,padded_test_init_list_data = padding_test_data(test_data, [ad_lists,share_lists,nature_lists], window_size=params_dict['window_size'], behavior_length=params_dict['behavior_length'])
    print("test mean sequence length: {}".format(np.mean(test_seq_len)))
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # eval the model before training process
        step = 0
        eval_step = 0

        print("train data size: {}".format(total_session_num))
        batch_size = params_dict['batch_size']
        epochs = params_dict['epochs']
        eval_iter_num = (total_session_num // params_dict['evaluate_times_per_epoch']) // batch_size
        eval_steps_per_save = 10
        # batch_num = data_size // batch_size
        beg_time = time.time()

        # evaluate initial criterion
        training_monitor['train_loss'].append(0.0)
        training_monitor['train_kl_loss'].append(0.0)
        training_monitor['train_ranker_loss'].append(0.0)
        training_monitor['evaluate_step'].append(eval_step)

        out_res, vali_loss, vali_kl_loss, vali_ranker_loss, res_3, res_5, res_10, res_20 = evaluate(
            params_dict, model, sess, padded_test_data, test_seq_len, padded_test_init_list_data, ad_lists, share_lists,
            nature_lists)
        print(_format.format("eval step", "topK", "NDCG", "REVENUE", "real RATIO", "predict RATIO","alpha NDCG","alpha REVENUE"))
        for res, top in zip([res_3, res_5, res_10, res_20], [3, 5, 10, 20]):
            print(f_format.format(eval_step, top, res[0], res[1], res[2], res[3],res[4],res[5]))
        print("test loss: {}".format(vali_loss))
        print("test kl loss: {}".format(vali_kl_loss))
        print("test ranker loss: {}".format(vali_ranker_loss))
        training_monitor['test_loss'].append(vali_loss)
        training_monitor['test_kl_loss'].append(vali_kl_loss)
        training_monitor['test_ranker_loss'].append(vali_ranker_loss)

        add_to_monitor(training_monitor, res_3, res_5, res_10, res_20)
        eval_step += 1

        file_chosen_num = total_session_num//len(seqlen_file_lists)

        for epoch in range(epochs):
            print("Start epoch {}/{}.".format(epoch, epochs))
            train_losses = []
            train_kl_losses = []
            train_ranker_losses = []


            for f_name, l_name, i_name in zip(padded_file_lists, seqlen_file_lists, init_list_file_lists):
                padded_train_data = load_padded_data(f_name)
                seq_len = load_padded_data(l_name)
                padded_train_init_lists=load_padded_data(i_name)
                state = np.random.get_state()
                np.random.shuffle(padded_train_data)
                np.random.set_state(state)
                np.random.shuffle(seq_len)
                np.random.set_state(state)
                np.random.shuffle(padded_train_init_lists)

                this_file_size = seq_len.shape[0]
                batch_num = this_file_size // batch_size if this_file_size % batch_size == 0 else (
                        this_file_size // batch_size + 1)

                for batch_no in range(batch_num):
                    data_batch, batch_seq_len, batch_init_list = get_batch(padded_train_data, seq_len,padded_train_init_lists, batch_size=batch_size,
                                                          batch_no=batch_no)

                    loss, kl_loss, ranker_loss = model.train(sess, data_batch, batch_seq_len, batch_init_list,params_dict['lr'], params_dict['reg_lambda'],
                                                             keep_prob=params_dict['keep_prob'])
                    train_losses.append(loss)
                    train_kl_losses.append(kl_loss)
                    train_ranker_losses.append(ranker_loss)
                    step += 1

                    if step % eval_iter_num == 0:
                        train_loss = sum(train_losses) / len(train_losses)
                        train_kl_loss = sum(train_kl_losses) / len(train_kl_losses)
                        train_ranker_loss = sum(train_ranker_losses) / len(train_ranker_losses)
                        training_monitor['train_loss'].append(train_loss)
                        training_monitor['train_kl_loss'].append(train_kl_loss)
                        training_monitor['train_ranker_loss'].append(train_ranker_loss)
                        training_monitor['evaluate_step'].append(eval_step)

                        print("Training time: {}".format(time.time() - beg_time))
                        eval_time = time.time()
                        out_res, vali_loss, vali_kl_loss, vali_ranker_loss, res_3, res_5, res_10, res_20 = evaluate(
                            params_dict, model, sess, padded_test_data, test_seq_len,padded_test_init_list_data, ad_lists, share_lists, nature_lists)
                        print(_format.format("eval step", "topK", "NDCG", "REVENUE", "real RATIO", "predict RATIO",
                                             "alpha NDCG", "alpha REVENUE"))
                        for res, top in zip([res_3, res_5, res_10, res_20], [3, 5, 10, 20]):
                            print(f_format.format(eval_step, top, res[0], res[1], res[2], res[3],res[4],res[5]))
                        print("train loss: {}, test loss: {}".format(train_loss, vali_loss))
                        print("train kl loss: {}, test kl loss: {}".format(train_kl_loss, vali_kl_loss))
                        print("train ranker loss: {}, test ranker loss: {}".format(train_ranker_loss, vali_ranker_loss))

                        print("Evaluating time: {}".format(time.time() - eval_time))
                        training_monitor['test_loss'].append(vali_loss)
                        training_monitor['test_kl_loss'].append(vali_kl_loss)
                        training_monitor['test_ranker_loss'].append(vali_ranker_loss)
                        add_to_monitor(training_monitor, res_3, res_5, res_10, res_20)

                        beg_time = time.time()

                        if params_dict['only_store_best'] == False and epoch%5 == 0:
                            if params_dict['multi_task']:
                                model.save_pb(sess,
                                              os.path.join(args.result_path, 'models', 'eval_step_{}.pb'.format(eval_step)),
                                              name_list=['pred', 'logits', 'ranker_score'])
                            else:
                                model.save_pb(sess,
                                              os.path.join(args.result_path, 'models', 'eval_step_{}.pb'.format(eval_step)),
                                              name_list=['pred', 'logits'])

                        cur_ratio = training_monitor['real_ratio_{}'.format(params_dict['evaluate_cri_number'])][-1]
                        cur_revenue = training_monitor['revenue_{}'.format(params_dict['evaluate_cri_number'])][-1]
                        model_ratio_record = training_monitor['revenue_{}'.format(params_dict['evaluate_cri_number'])][-11:-1]
                        if epoch >= 10 \
                                and (original_revenue - cur_revenue) / float(
                            original_revenue) < params_dict['revenue_decrease_bound'] \
                                and abs(cur_ratio - 0.0) < min_gap \
                                and np.max(model_ratio_record) - np.mean(model_ratio_record) < 0.1 \
                                and np.mean(model_ratio_record) - np.min(model_ratio_record) < 0.1:
                            print("Saving best model to int_rank_seq.pb")
                            min_gap = abs(cur_ratio - params_dict['ideal_ratio'])
                            model.save_pb(sess,
                                          os.path.join(args.result_path, 'model.pb'),
                                          name_list=['pred', 'logits'])
                        eval_step += 1
            sess.graph.finalize()
            if epoch%5==0:
                df = pd.DataFrame.from_dict(training_monitor)
                df.to_csv(os.path.join(args.result_path, 'result.csv'))
        df = pd.DataFrame.from_dict(training_monitor)
        df.to_csv(os.path.join(args.result_path, 'result.csv'))
        print("Training finish.")


if __name__ == '__main__':
    args = argparser()
    train(args)
