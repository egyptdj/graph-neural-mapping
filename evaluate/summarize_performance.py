import os
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Summarize the performance of the model')
    parser.add_argument('--expdir', type=str, default='results/graph_neural_mapping', help='path containing the result.csv and the test_sequence.csv')
    parser.add_argument('--savedir', type=str, default='performance', help='path to save the performance record files within the expdir')
    parser.add_argument('--fold_idx', nargs='+', default=[0,1,2,3,4,5,6,7,8,9], help='fold indices')

    opt = parser.parse_args()

    os.makedirs(os.path.join(opt.expdir, opt.savedir), exist_ok=True)

    final = {'accuracy':[], 'precision':[], 'recall':[]}
    early = {'accuracy':[], 'precision':[], 'recall':[]}
    epoch = {'accuracy':[], 'precision':[], 'recall':[]}
    for current_fold in opt.fold_idx:
        performance_table = pd.read_csv(os.path.join(opt.expdir, 'csv', current_fold, 'result.csv'))[['accuracy', 'precision', 'recall']]
        performance_per_epochs = pd.read_csv(os.path.join(opt.expdir, 'csv', current_fold, 'test_sequence.csv'), header=None)[[1,2,3]]

        final['accuracy'].append(performance_table.iloc[0]['accuracy'])
        final['precision'].append(performance_table.iloc[0]['precision'])
        final['recall'].append(performance_table.iloc[0]['recall'])
        early['accuracy'].append(performance_table.iloc[1]['accuracy'])
        early['precision'].append(performance_table.iloc[1]['precision'])
        early['recall'].append(performance_table.iloc[1]['recall'])
        epoch['accuracy'].append(performance_per_epochs[0].to_list())
        epoch['precision'].append(performance_per_epochs[1].to_list())
        epoch['recall'].append(performance_per_epochs[2].to_list())

    final_accuracy = np.mean(final['accuracy'])
    final_precision = np.mean(final['precision'])
    final_recall = np.mean(final['recall'])
    early_accuracy = np.mean(early['accuracy'])
    early_precision = np.mean(early['precision'])
    early_recall = np.mean(early['recall'])
    epoch_accuracy = np.max(np.mean(np.asarray(epoch['accuracy'])), axis=1)
    epoch_precision = np.max(np.mean(np.asarray(epoch['precision'])), axis=1)
    epoch_recall = np.max(np.mean(np.asarray(epoch['recall'])), axis=1)

    with open(os.path.join(opt.expdir, opt.savedir, 'result_summary.csv'), 'w') as f:
        f.write(','.join(['method', 'accuracy', 'precision', 'recall']))
        f.write("\n")
        f.write(','.join(['early', str(early_accuracy), str(early_precision), str(early_recall)]))
        f.write("\n")
        f.write(','.join(['final', str(final_accuracy), str(final_precision), str(final_recall)]))
        f.write("\n")
        f.write(','.join(['epoch', str(epoch_accuracy), str(epoch_precision), str(epoch_recall)]))
        f.write("\n")

if __name__ == '__main__':
    main()
