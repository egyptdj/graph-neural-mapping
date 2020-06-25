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
    accuracy_final = []
    precision_final = []
    recall_final = []
    for current_fold in opt.fold_idx:
        performance_table = pd.read_csv(os.path.join(opt.expdir, 'csv', str(current_fold), 'result.csv'))[['accuracy', 'precision', 'recall']]

        accuracy.append(performance_table.iloc[0]['accuracy'])
        precision.append(performance_table.iloc[0]['precision'])
        recall.append(performance_table.iloc[0]['recall'])
        accuracy_final.append(performance_table.iloc[1]['accuracy'])
        precision_final.append(performance_table.iloc[1]['precision'])
        recall_final.append(performance_table.iloc[1]['recall'])

    accuracy_mean = np.mean(accuracy)
    precision_mean = np.mean(precision)
    recall_mean = np.mean(recall)
    accuracy_std = np.std(accuracy)
    precision_std = np.std(precision)
    recall_std = np.std(recall)
    accuracy_mean_final = np.mean(accuracy_final)
    precision_mean_final = np.mean(precision_final)
    recall_mean_final = np.mean(recall_final)
    accuracy_std_final = np.std(accuracy_final)
    precision_std_final = np.std(precision_final)
    recall_std_final = np.std(recall_final)

    with open(os.path.join(opt.expdir, opt.savedir, 'result_summary.csv'), 'w') as f:
        f.write(','.join(['accuracy', 'precision', 'recall', 'accuracy_std', 'precision_std', 'recall_std']))
        f.write("\n")
        f.write(','.join([str(accuracy_mean), str(precision_mean), str(recall_mean), str(accuracy_std), str(precision_std), str(recall_std)]))
        f.write("\n")
        f.write(','.join([str(accuracy_mean_final), str(precision_mean_final), str(recall_mean_final), str(accuracy_std_final), str(precision_std_final), str(recall_std_final)]))

if __name__ == '__main__':
    main()
