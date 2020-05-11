import os
import argparse
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='compute robustness of the saliency mapping')
    parser.add_argument('--expdir', type=str, default='results/graph_neural_mapping', help='path to the experiment results')
    parser.add_argument('--saliency', type=str, default='saliency_female', help='path to the experiment results')
    parser.add_argument('--topk', type=int, default=20, help='top k items to compare robustness')
    opt = parser.parse_args()


    full_folds = pd.read_csv(os.path.join(opt.expdir, 'saliency_nii', f'{opt.saliency}.csv'))
    one_folds = [pd.read_csv(os.path.join(opt.expdir, 'saliency_nii_fold{}'.format(i), f'{opt.saliency}.csv')) for i in range(10)]
    five_folds = [pd.read_csv(os.path.join(opt.expdir, 'saliency_nii_fold{}'.format(i), f'{opt.saliency}.csv')) for i in ['01234', '56789']]

    full_one_match = count_matches(full_folds, one_folds, opt.topk)
    full_one_match_mean = np.mean(full_one_match)
    full_one_match_std = np.std(full_one_match)
    one_folds_robustness = 100*full_one_match_mean / opt.topk
    one_folds_robustness_std = 100*full_one_match_std / opt.topk


    full_five_match = count_matches(full_folds, five_folds, opt.topk)
    full_five_match_mean = np.mean(full_five_match)
    full_five_match_std = np.std(full_five_match)
    five_folds_robustness = 100*full_five_match_mean / opt.topk
    five_folds_robustness_std = 100*full_five_match_std / opt.topk

    print('===='*12)
    print('===='*12)
    print(f'one fold matches {full_one_match_mean} out of {opt.topk}. robustness: {one_folds_robustness}+{one_folds_robustness_std:.2f}%')
    print(f'five fold matches {full_five_match_mean} out of {opt.topk}. robustness: {five_folds_robustness}+{five_folds_robustness_std:.2f}%')
    print('===='*12)
    print('===='*12)


def count_matches(full_folds, partial_folds, topk):
    full_fold_rois = full_folds['roi'][:topk].to_list()
    partial_fold_rois = [fold['roi'][:topk].to_list() for fold in partial_folds]

    partial_fold_counts = []
    for rois in partial_fold_rois:
        count = 0
        for roi in rois:
            if roi in full_fold_rois:
                count += 1
        partial_fold_counts.append(count)
    return partial_fold_counts


if __name__=='__main__':
    main()
