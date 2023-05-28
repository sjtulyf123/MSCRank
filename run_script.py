import os
import numpy as np

def run():
    cmd = "python run_rnn.py --file_path prm_data --result_path exp_logs/1031/1kl_lr1e-5 --batch_size 512 --lr 0.00001 \
     --epochs 35 --optimize_ratio --multi_task --kl_weight 1.0 --ranker_weight 1.0 --use_new_rnn\
     --padded_file_path padded_feature --update_padded_data --gpu_num 1 --evaluate_times_per_epoch 1"
    print(cmd)
    os.system(cmd)


run()
