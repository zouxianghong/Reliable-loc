import os
import pickle

import numpy as np


def cmp_stats(ref_pkl, query_pkl, RRE_thresh=1.0, RTE_thresh=0.5, log_better=True, log_worse=False):
    # check input
    if not os.path.exists(ref_pkl) or not os.path.exists(query_pkl):
        return None
    with open(ref_pkl, 'rb') as handle:
        ref_aidxs, ref_RREs, ref_RTEs, ref_run_time = pickle.load(handle)
        if ref_aidxs is None or len(ref_aidxs) == 0:
            return None
    with open(query_pkl, 'rb') as handle:
        que_aidxs, que_RREs, que_RTEs, que_run_time = pickle.load(handle)
        if que_aidxs is None or len(que_aidxs) == 0:
            return None
    # format data
    ref_data = dict()
    for i in range(len(ref_aidxs)):
        if ref_aidxs[i] in que_aidxs:
            ref_data[ref_aidxs[i]] = {
                'RRE': ref_RREs[i],
                'RTE': ref_RTEs[i],
                'run_time': ref_run_time[i]
            }
    que_data = dict()
    for i in range(len(que_aidxs)):
        if que_aidxs[i] in ref_aidxs:
            que_data[que_aidxs[i]] = {
                'RRE': que_RREs[i],
                'RTE': que_RTEs[i],
                'run_time': que_run_time[i]
            }
    # compare
    keys = list(ref_data.keys())
    keys.sort()
    delta_RREs, delta_RTEs, delta_run_times = [], [], []
    for key in keys:
        ref, que = ref_data[key], que_data[key]
        delta_RREs.append(que['RRE'] - ref['RRE'])
        delta_RTEs.append(que['RTE'] - ref['RTE'])
        delta_run_times.append(que['run_time'] - ref['run_time'])
        is_worse = que['RRE'] - ref['RRE'] > RRE_thresh and que['RTE'] - ref['RTE'] > RTE_thresh
        is_better = ref['RRE'] - que['RRE'] > RRE_thresh and ref['RTE'] - que['RTE'] > RTE_thresh
        if is_better:
            if not log_better:
                continue
            print('a_idx: {}, delta RRE(deg): {:.3f}, delta RTE(m): {:.3f}, delta run time(ms): {:.2f}, state: better'.format(
                key, que['RRE'] - ref['RRE'], que['RTE'] - ref['RTE'], que['run_time'] - ref['run_time']))
        elif is_worse:
            if not log_worse:
                continue
            print('a_idx: {}, delta RRE(deg): {:.3f}, delta RTE(m): {:.3f}, delta run time(ms): {:.2f}, state: worse'.format(
                key, que['RRE'] - ref['RRE'], que['RTE'] - ref['RTE'], que['run_time'] - ref['run_time']))
        else:
            continue
    print('--------------------Summary--------------------')
    delta_RRE = np.mean(delta_RREs)
    delta_RTE = np.mean(delta_RTEs)
    delta_run_time = np.mean(delta_run_times)
    print('mean delta RRE(deg): {:.3f}, mean delta RTE(m): {:.3f}, mean delta run time(ms): {:.2f}'.format(
        delta_RRE, delta_RTE, delta_run_time))


if __name__ == '__main__':
    ref_pkl = '/home/ericxhzou/Code/ppt-net-plus/exp/pose_est/events/2023-08-06T22-15-27_deepl_nn_dist/2023-08-10T07-53-10_deepl_with_ransac_hankou/pose_est_res_deepl_infer_eval_individual_top_k/stat.pickle'
    query_pkl = '/home/ericxhzou/Code/ppt-net-plus/exp/pose_est/events/2023-08-06T22-15-27_deepl_nn_dist/2023-08-10T07-53-10_deepl_with_ransac_hankou/pose_est_res_deepl_infer_eval_group_top_k/stat.pickle'
    cmp_stats(ref_pkl, query_pkl, RRE_thresh=5.0, RTE_thresh=2.0, log_better=True, log_worse=False)
    print()
    print('********************************************************************************')
    cmp_stats(ref_pkl, query_pkl, RRE_thresh=5.0, RTE_thresh=2.0, log_better=False, log_worse=True)