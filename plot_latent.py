import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter
from sklearn import manifold
from sklearn.metrics import silhouette_score


def main():
    parser = argparse.ArgumentParser(description='Plot the t-SNE embedding the latent space')
    parser.add_argument('--datadir', nargs='+', default=None, help='path containing the latent_space_initial.npy, latent_space_early.npy, and the latent_space_final.npy')
    parser.add_argument('--savedir', type=str, default='./tsne', help='path to save the resulting tsne images')

    opt = parser.parse_args()

    os.makedirs(opt.savedir, exist_ok=True)

    perplexities = [5, 30, 50, 70, 100]

    latent_space_initial = []
    latent_space_early = []
    latent_space_final = []
    labels = []

    for dir in opt.datadir:
        latent_space_initial.append(np.load(os.path.join(dir, 'latent_space_initial.npy')))
        latent_space_early.append(np.load(os.path.join(dir, 'latent_space_early.npy')))
        latent_space_final.append(np.load(os.path.join(dir, 'latent_space_final.npy')))
        labels.append(np.load(os.path.join(dir, 'labels.npy')))

    for perplexity in perplexities:
        for idx, _ in enumerate(labels):
            plot_tsne(latent_space_initial[idx], labels[idx], perplexity, opt.savedir, 'initial_latent_perp{}_fold{}'.format(perplexity, idx))
            plot_tsne(latent_space_early[idx], labels[idx], perplexity, opt.savedir, 'early_latent_perp{}_fold{}'.format(perplexity, idx))
            plot_tsne(latent_space_final[idx], labels[idx], perplexity, opt.savedir, 'final_latent_perp{}_fold{}'.format(perplexity, idx))

    latent_space_initial = np.concatenate(latent_space_initial, 0)
    latent_space_early = np.concatenate(latent_space_early, 0)
    latent_space_final = np.concatenate(latent_space_final, 0)
    labels = np.concatenate(labels, 0).squeeze()

    for perplexity in perplexities:
        plot_tsne(latent_space_initial, labels, perplexity, opt.savedir, 'initial_latent_perp{}_full'.format(perplexity))
        plot_tsne(latent_space_early, labels, perplexity, opt.savedir, 'early_latent_perp{}_full'.format(perplexity))
        plot_tsne(latent_space_final, labels, perplexity, opt.savedir, 'final_latent_perp{}_full'.format(perplexity))


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
    plt.scatter(Y[female, 0], Y[female, 1], c="r", s=1000, linewidth=2, edgecolors='black')
    plt.scatter(Y[male, 0], Y[male, 1], c="g", s=1000, linewidth=2, edgecolors='black')
    plt.axis('tight')
    plt.savefig(os.path.join(savedir, savename))
    plt.clf()
    plt.close()

if __name__=='__main__':
    main()
