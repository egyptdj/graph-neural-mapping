import os
import argparse
import numpy as np
from sklearn.metrics import silhouette_score


def main():
    parser = argparse.ArgumentParser(description='Compute the silhouette score of the latent space')
    parser.add_argument('--datadir', nargs='+', default=None, help='paths containing the initial_latent_space.npy and the final_latent_space.npy')
    parser.add_argument('--savedir', type=str, default='./silhouette', help='path to save the resulting tsne images')

    opt = parser.parse_args()
    os.makedirs(opt.savedir, exist_ok=True)

    initial_latent_space = []
    final_latent_space = []
    labels = []

    for dir in opt.datadir:
        initial_latent_space.append(np.load(os.path.join(dir, 'initial_latent_space.npy')))
        final_latent_space.append(np.load(os.path.join(dir, 'final_latent_space.npy')))
        labels.append(np.load(os.path.join(dir, 'labels.npy')).squeeze())

    initial_silhouette = []
    final_silhouette = []

    for idx, _ in enumerate(labels):
        initial_silhouette.append(silhouette_score(initial_latent_space[idx], labels[idx], sample_size=initial_latent_space[idx].shape[0]))
        final_silhouette.append(silhouette_score(final_latent_space[idx], labels[idx], sample_size=final_latent_space[idx].shape[0]))

    with open(os.path.join(opt.savedir,'silhouette_score.csv'), 'w') as f:
        f.write("fold_idx,initial_score,final_score\n")
        for idx, (init, fin) in enumerate(zip(initial_silhouette, final_silhouette)):
            f.write("{},{},{}\n".format(idx, init, fin))
        f.write("mean,{},{}\n".format(np.mean(initial_silhouette), np.mean(final_silhouette)))


if __name__=='__main__':
    main()
