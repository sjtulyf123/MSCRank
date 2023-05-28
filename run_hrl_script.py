import os

def run():
    cmd = " python run_hrl_baseline.py --file_path prm_data --target_dir exp_logs/hrl/batch512_lr5e-5_30kl --batch_size 512 --lr 0.00005 \
     --epochs 20 --kl_weight 30.0 --gpu_num 0 --seed 8887"
    print(cmd)
    os.system(cmd)


run()
