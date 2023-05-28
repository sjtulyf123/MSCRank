import numpy as np
from baselines.fairness_baselines import MMR, LinkedInDelta
import glob
import os
import argparse
from multiprocessing import Pool
from rnn_version.utils import process_baseline_data, gen_initial_ranking, name_map


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, default=30,
                        help='slide window size when performing data augmentation')

    parser.add_argument('--file_path', default='prm_data')
    parser.add_argument('--result_path', default='exp_logs/online_inference')
    parser.add_argument('--load_pb_path', default='exp_logs/online_inference')

    return parser.parse_args()

def run(args):
    file_lists = glob.glob(
        os.path.join(args.file_path, "clean_data", "test_part*.pkl"))
    file_lists = sorted(file_lists, reverse=False)
    print(file_lists)

    parallel_worker = Pool(2)
    baseline_data = process_baseline_data(file_lists, parallel_worker)
    ad_lists, share_lists, nature_lists = gen_initial_ranking(baseline_data)
    # print("MMR")
    # for pair in [(0,1,2)]:
    #     for p in np.arange(0.1,0.9,0.1):
    #         print(pair,p)
    #         mmr = MMR(baseline_data,ad_lists,share_lists,nature_lists,pair,30,p)
    #         mmr.run()
    #     for p in np.arange(0.1,0.9,0.1):
    #         print(pair,p)
    #         mmr = MMR(baseline_data,ad_lists,share_lists,nature_lists,pair,20,p)
    #         mmr.run()
    
    print("linkedin")
    for pair in [(0, 1, 2)]:
        for p in np.arange(0.05,0.501,0.025):
            print(pair,p)
            linkedin = LinkedInDelta(baseline_data,ad_lists,share_lists,nature_lists,pair,30,p)
            linkedin.run()
    
    parallel_worker.close()
    parallel_worker.join()

if __name__=="__main__":
    args = argparser()
    run(args)
