import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter
from sklearn import manifold
from sklearn.metrics import silhouette_score


def main():
    parser = argparse.ArgumentParser(description='Plot the t-SNE embedding the latent space')
    parser.add_argument('--expdir', type=str, default='results/graph_neural_mapping', help='path containing the latent_space_*.npy')
    parser.add_argument('--savedir', type=str, default='tsne', help='path to save the plotted tsne files within the expdir')
    parser.add_argument('--fold_idx', nargs='+', default=[0,1,2,3,4,5,6,7,8,9], help='fold indices')
    parser.add_argument('--perplexities', nargs='+', default=[5, 30, 50, 70, 100], help='tsne perplexities')
    parser.add_argument('--random_state', type=int, default=0, help='tsne random state')

    opt = parser.parse_args()

    os.makedirs(os.path.join(opt.expdir, opt.savedir), exist_ok=True)

    latent_space_initial = []
    latent_space_early = []
    latent_space_final = []
    labels = []

    for current_fold in opt.fold_idx:
        latent_space_initial.append(np.load(os.path.join(opt.expdir, 'latent', str(current_fold), 'latent_space_initial.npy')))
        latent_space_early.append(np.load(os.path.join(opt.expdir, 'latent', str(current_fold), 'latent_space_early.npy')))
        latent_space_final.append(np.load(os.path.join(opt.expdir, 'latent', str(current_fold), 'latent_space_final.npy')))
        labels.append(np.load(os.path.join(opt.expdir, 'latent', str(current_fold), 'labels.npy')))

    for perplexity in opt.perplexities:
        for idx, _ in enumerate(labels):
            plot_tsne(latent_space_initial[idx], labels[idx], perplexity, os.path.join(opt.expdir, opt.savedir), 'initial_latent_perp{}_fold{}'.format(perplexity, idx), opt.random_state)
            plot_tsne(latent_space_early[idx], labels[idx], perplexity, os.path.join(opt.expdir, opt.savedir), 'early_latent_perp{}_fold{}'.format(perplexity, idx), opt.random_state)
            plot_tsne(latent_space_final[idx], labels[idx], perplexity, os.path.join(opt.expdir, opt.savedir), 'final_latent_perp{}_fold{}'.format(perplexity, idx), opt.random_state)

    latent_space_initial = np.concatenate(latent_space_initial, 0)
    latent_space_early = np.concatenate(latent_space_early, 0)
    latent_space_final = np.concatenate(latent_space_final, 0)
    labels = np.concatenate(labels, 0).squeeze()

    for perplexity in opt.perplexities:
        plot_tsne(latent_space_initial, labels, perplexity, os.path.join(opt.expdir, opt.savedir), 'initial_latent_perp{}_full'.format(perplexity), opt.random_state)
        plot_tsne(latent_space_early, labels, perplexity, os.path.join(opt.expdir, opt.savedir), 'early_latent_perp{}_full'.format(perplexity), opt.random_state)
        plot_tsne(latent_space_final, labels, perplexity, os.path.join(opt.expdir, opt.savedir), 'final_latent_perp{}_full'.format(perplexity), opt.random_state)


def plot_tsne(latent, label, perplexity, savedir, savename, random_state=0):
    n_samples = label.shape[0]
    n_components = label.max()+1

    X, y = latent, label.squeeze()

    female = y == 0
    male = y == 1

    fig = plt.figure(figsize=(18, 18))
    tsne = manifold.TSNE(n_components=n_components, init='random',
                         random_state=random_state, perplexity=perplexity)
    Y = tsne.fit_transform(X)
    plt.scatter(Y[female, 0], Y[female, 1], c="r", s=1000, linewidth=2, edgecolors='black')
    plt.scatter(Y[male, 0], Y[male, 1], c="g", s=1000, linewidth=2, edgecolors='black')
    plt.axis('tight')
    plt.savefig(os.path.join(savedir, savename))
    plt.clf()
    plt.close()

if __name__=='__main__':
    main()
