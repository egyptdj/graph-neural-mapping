import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter
from sklearn import manifold
from sklearn.metrics import silhouette_score


def main():
    parser = argparse.ArgumentParser(description='Plot the t-SNE embedding the latent space')
    parser.add_argument('--expdir', type=str, default='results/graph_neural_mapping', help='path to the experiment results')
    parser.add_argument('--latentdir', type=str, default='latent', help='path containing the latent_space_*.npy')
    parser.add_argument('--savedir', type=str, default='tsne', help='path to save the plotted tsne files within the expdir')
    parser.add_argument('--fold_idx', nargs='+', default=['0','1','2','3','4','5','6','7','8','9'], help='fold indices')
    parser.add_argument('--perplexities', type=int, nargs='+', default=[50], help='tsne perplexities')
    parser.add_argument('--random_state', type=int, default=0, help='tsne random state')

    opt = parser.parse_args()

    os.makedirs(os.path.join(opt.expdir, opt.savedir), exist_ok=True)

    latent_space_initial = []
    latent_space = []
    labels = []

    for current_fold in opt.fold_idx:
        latent_space_initial.append(np.load(os.path.join(opt.expdir, opt.latentdir, str(current_fold), 'latent_space_initial.npy')))
        latent_space.append(np.load(os.path.join(opt.expdir, opt.latentdir, str(current_fold), 'latent_space.npy')))
        labels.append(np.load(os.path.join(opt.expdir, opt.latentdir, str(current_fold), 'labels.npy')))

    for perplexity in opt.perplexities:
        print('PLOTTING PER-FOLD LATENT SPACE PERPLEXITY: {}'.format(perplexity))
        for idx, _ in enumerate(labels):
            plot_tsne(latent_space_initial[idx], labels[idx], perplexity, os.path.join(opt.expdir, opt.savedir), 'initial_latent_perp{}_fold{}'.format(perplexity, idx), opt.random_state)
            plot_tsne(latent_space[idx], labels[idx], perplexity, os.path.join(opt.expdir, opt.savedir), 'latent_perp{}_fold{}'.format(perplexity, idx), opt.random_state)


def plot_tsne(latent, label, perplexity, savedir, savename, random_state=0):
    n_samples = label.shape[0]
    n_components = label.max()+1

    X, y = latent, label.squeeze()

    female = y == 0
    male = y == 1

    fig, ax = plt.subplots()
    fig.set_size_inches((8,8))
    ax.axis('off')
    tsne = manifold.TSNE(n_components=n_components, init='random',
                         random_state=random_state, perplexity=perplexity)
    Y = tsne.fit_transform(X)
    plt.scatter(Y[female, 0], Y[female, 1], c="r", s=1000, linewidth=3, edgecolors='black')
    plt.scatter(Y[male, 0], Y[male, 1], c="b", s=1000, linewidth=3, edgecolors='black')
    plt.axis('tight')
    plt.savefig(os.path.join(savedir, savename))
    plt.clf()
    plt.close()

if __name__=='__main__':
    main()
