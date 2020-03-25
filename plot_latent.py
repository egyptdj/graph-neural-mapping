import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter
from sklearn import manifold
from sklearn.metrics import silhouette_score


def main():
    parser = argparse.ArgumentParser(description='Plot the t-SNE embedding the latent space')
    parser.add_argument('--datadir', nargs='+', default=None, help='path containing the initial_latent_space.npy and the final_latent_space.npy')
    parser.add_argument('--savedir', type=str, default='./tsne', help='path to save the resulting tsne images')

    opt = parser.parse_args()

    os.makedirs(opt.savedir, exist_ok=True)

    perplexities = [5, 30, 50, 70, 100]

    initial_latent_space = []
    final_latent_space = []
    labels = []

    for dir in opt.datadir:
        initial_latent_space.append(np.load(os.path.join(dir, 'initial_latent_space.npy')))
        final_latent_space.append(np.load(os.path.join(dir, 'final_latent_space.npy')))
        labels.append(np.load(os.path.join(dir, 'labels.npy')))

    for perplexity in perplexities:
        for idx, _ in enumerate(labels):
            plot_tsne(initial_latent_space[idx], labels[idx], perplexity, opt.savedir, 'initial_latent_perp{}_fold{}'.format(perplexity, idx))
            plot_tsne(final_latent_space[idx], labels[idx], perplexity, opt.savedir, 'final_latent_perp{}_fold{}'.format(perplexity, idx))

    initial_latent_space = np.concatenate(initial_latent_space, 0)
    final_latent_space = np.concatenate(final_latent_space, 0)
    labels = np.concatenate(labels, 0).squeeze()

    for perplexity in perplexities:
        plot_tsne(initial_latent_space, labels, perplexity, opt.savedir, 'initial_latent_perp{}_full'.format(perplexity))
        plot_tsne(final_latent_space, labels, perplexity, opt.savedir, 'final_latent_perp{}_full'.format(perplexity))


def plot_tsne(latent, label, perplexity, savedir, savename):
    n_samples = label.shape[0]
    n_components = label.max()+1

    X, y = latent, label.squeeze()

    female = y == 0
    male = y == 1

    fig = plt.figure(figsize=(18, 18))
    tsne = manifold.TSNE(n_components=n_components, init='random',
                         random_state=0, perplexity=perplexity)
    Y = tsne.fit_transform(X)
    plt.scatter(Y[female, 0], Y[female, 1], c="r", s=400, linewidth=1, edgecolors='black')
    plt.scatter(Y[male, 0], Y[male, 1], c="g", s=400, linewidth=1, edgecolors='black')
    plt.axis('tight')
    plt.savefig(os.path.join(savedir, savename))
    plt.clf()
    plt.close()

if __name__=='__main__':
    main()
