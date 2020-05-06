import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    parser = argparse.ArgumentParser(description='Plot the saliency map in the nifti format')
    parser.add_argument('--expdir', type=str, default='results/graph_neural_mapping', help='path containing the saliency_female.npy and the saliency_male.npy')
    parser.add_argument('--savedir', type=str, default='saliency_heatmap', help='path to save the saliency heatmap images within the expdir')
    parser.add_argument('--seed', type=int, default=0, help='seed to select random subjects')
    parser.add_argument('--num_samples', type=int, default=10, help='number of sample subjects')
    parser.add_argument('--fold_idx', nargs='+', default=['0','1','2','3','4','5','6','7','8','9'], help='fold indices')

    opt = parser.parse_args()

    os.makedirs(os.path.join(opt.expdir, opt.savedir), exist_ok=True)
    np.random.seed(opt.seed)
    plt.style.use('ggplot')
    sns.set_style('ticks')
    sns.set_context('paper', font_scale=4)

    # plot gradient based saliency
    saliency0 = []
    saliency1 = []

    for current_fold in opt.fold_idx:
        saliency0.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'grad_saliency_female_early.npy')))
        saliency1.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'grad_saliency_male_early.npy')))

    num_subjects = len(saliency0)
    sample_subject_idx = np.random.choice(num_subjects, opt.num_samples)

    saliency0 = np.concatenate(saliency0, axis=0)
    saliency1 = np.concatenate(saliency1, axis=0)

    female_subjects = [np.mean(saliency0, axis=0)]
    for i in sample_subject_idx:
        female_subjects.append(saliency0[i])

    for saliency in female_subjects:
        saliency -= saliency.min()
        saliency /= saliency.max()

    ticks = np.arange(-1, 400, 50)
    ticks[0] = 0
    for i, saliency in enumerate(female_subjects):
        fig, ax = plt.subplots()
        if i==0:
            fig.set_size_inches((24, 21))
            ax = sns.heatmap(saliency, cmap='Greys', yticklabels=50, square=True)
        else:
            fig.set_size_inches((15, 15))
            ax = sns.heatmap(saliency, cmap='Greys', yticklabels=50, square=True, cbar=False)

        # ax.set_xlabel('Node index')
        # ax.set_ylabel('Node feature')
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks+1)

        fig.tight_layout()
        if i==0:
            fig.savefig(os.path.join(opt.expdir, opt.savedir, 'saliency_female_group.png'))
        else:
            fig.savefig(os.path.join(opt.expdir, opt.savedir, 'saliency_female_sample{}.png'.format(i)))
        plt.close()

    male_subjects = [np.mean(saliency1, axis=0)]
    for i in sample_subject_idx:
        male_subjects.append(saliency1[i])

    for saliency in male_subjects:
        saliency -= saliency.min()
        saliency /= saliency.max()

    for i, saliency in enumerate(male_subjects):
        fig, ax = plt.subplots()
        if i==0:
            fig.set_size_inches((24, 21))
            ax = sns.heatmap(saliency, cmap='Greys', yticklabels=50, square=True)
        else:
            fig.set_size_inches((15, 15))
            ax = sns.heatmap(saliency, cmap='Greys', yticklabels=50, square=True, cbar=False)

        # ax.set_xlabel('Node index')
        # ax.set_ylabel('Node feature')
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks+1)

        fig.tight_layout()
        if i==0:
            fig.savefig(os.path.join(opt.expdir, opt.savedir, 'saliency_male_group.png'))
        else:
            fig.savefig(os.path.join(opt.expdir, opt.savedir, 'saliency_male_sample{}.png'.format(i)))
        plt.close()


if __name__=='__main__':
    main()
