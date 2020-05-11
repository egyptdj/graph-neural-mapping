import os
import argparse
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Summarize the performance of the model')
    parser.add_argument('--expdir', type=str, default='results/graph_neural_mapping', help='path containing the result.csv and the test_sequence.csv')
    parser.add_argument('--savedir', type=str, default='performance', help='path to save the performance record files within the expdir')
    parser.add_argument('--fold_idx', nargs='+', default=['0','1','2','3','4','5','6','7','8','9'], help='fold indices')

    opt = parser.parse_args()

    os.makedirs(os.path.join(opt.expdir, opt.savedir), exist_ok=True)

    accuracy = []
    precision = []
    recall = []
    for current_fold in opt.fold_idx:
        performance_table = pd.read_csv(os.path.join(opt.expdir, 'csv', str(current_fold), 'result.csv'))[['accuracy', 'precision', 'recall']]

        accuracy.append(performance_table.iloc[0]['accuracy'])
        precision.append(performance_table.iloc[0]['precision'])
        recall.append(performance_table.iloc[0]['recall'])

    accuracy = np.mean(accuracy)
    precision = np.mean(precision)
    recall = np.mean(recall)
    accuracy_std = np.std(accuracy)
    precision_std = np.std(precision)
    recall_std = np.std(recall)

    with open(os.path.join(opt.expdir, opt.savedir, 'result_summary.csv'), 'w') as f:
        f.write(','.join(['accuracy', 'precision', 'recall', 'accuracy_std', 'precision_std', 'recall_std']))
        f.write("\n")
        f.write(','.join([str(early_accuracy), str(early_precision), str(early_recall), str(early_accuracy_std), str(early_precision_std), str(early_recall_std)]))

if __name__ == '__main__':
    main()
