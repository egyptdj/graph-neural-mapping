import os
import argparse
import numpy as np
from sklearn.metrics import silhouette_score


def main():
    parser = argparse.ArgumentParser(description='Compute the silhouette score of the latent space')
    parser.add_argument('--datadir', nargs='+', default=None, help='paths containing the latent_space_initial.npy, latent_space_early.npy, and the latent_space_final.npy')
    parser.add_argument('--savedir', type=str, default='./silhouette', help='path to save the resulting tsne images')

    opt = parser.parse_args()
    os.makedirs(opt.savedir, exist_ok=True)

    latent_space_initial = []
    latent_space_early = []
    latent_space_final = []
    labels = []

    for dir in opt.datadir:
        latent_space_initial.append(np.load(os.path.join(dir, 'latent_space_initial.npy')))
        latent_space_early.append(np.load(os.path.join(dir, 'latent_space_early.npy')))
        latent_space_final.append(np.load(os.path.join(dir, 'latent_space_final.npy')))
        labels.append(np.load(os.path.join(dir, 'labels.npy')).squeeze())

    initial_silhouette = []
    early_silhouette = []
    final_silhouette = []

    for idx, _ in enumerate(labels):
        initial_silhouette.append(silhouette_score(latent_space_initial[idx], labels[idx], sample_size=latent_space_initial[idx].shape[0]))
        early_silhouette.append(silhouette_score(latent_space_early[idx], labels[idx], sample_size=latent_space_early[idx].shape[0]))
        final_silhouette.append(silhouette_score(latent_space_final[idx], labels[idx], sample_size=latent_space_final[idx].shape[0]))

    with open(os.path.join(opt.savedir,'silhouette_score.csv'), 'w') as f:
        f.write("fold_idx,initial_score,early_score,final_score\n")
        for idx, (init, early, fin) in enumerate(zip(initial_silhouette, early_silhouette, final_silhouette)):
            f.write("{},{},{},{}\n".format(idx, init, early, fin))
        f.write("mean,{},{},{}\n".format(np.mean(initial_silhouette), np.mean(early_silhouette), np.mean(final_silhouette)))


if __name__=='__main__':
    main()
