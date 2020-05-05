import os
import pickle
import argparse
import neurosynth
from wordcloud import WordCloud


def main():
    parser = argparse.ArgumentParser(description='Plot the saliency map in the nifti format')
    parser.add_argument('--expdir', type=str, default='results/graph_neural_mapping', help='path to the experiment results')
    parser.add_argument('--neurosynthdir', type=str, default='neurosynth', help='path to save and load the neurosynth database')
    parser.add_argument('--saliencydir', type=str, default='saliency_nii', help='path containing the saliency nifti files')
    parser.add_argument('--savedir', type=str, default='saliency_decoded', help='path to save the decoded files within the expdir')

    opt = parser.parse_args()

    assert os.path.exists(os.path.join(opt.expdir, opt.saliencydir))
    os.makedirs(os.path.join(opt.expdir, opt.savedir), exist_ok=True)

    if os.path.exists(os.path.normpath(opt.neurosynthdir)):
        with open(os.path.join(opt.neurosynthdir, 'neurosynth_decoder.pkl'), 'rb') as f:
            decoder = pickle.load(f)
    else:
        neurosynth.dataset.download(path=opt.neurosynthdir, unpack=True)
        dataset = neurosynth.Dataset(filename=opt.path.join(opt.neurosynthdir, 'database.txt'), feature_filename=opt.path.join(opt.neurosynthdir, 'features.txt'))
        decoder = neurosynth.decode.Decoder(dataset)
        with open(os.path.join(opt.neurosynthdir, 'neurosynth_decoder.pkl'), 'wb') as f:
            pickle.dump(decoder, f, protocol=4)

    # decode grad saliency
    grad_nii_list = [os.path.join(opt.expdir, opt.saliencydir, 'saliency_normalized_grad_female_early.nii'),
                os.path.join(opt.expdir, opt.saliencydir, 'saliency_normalized_grad_male_early.nii')]

    decoded_semantics = decoder.decode(grad_nii_list, save=os.path.join(opt.expdir, opt.savedir, 'grad_decoded_table.csv'))
    decoded_semantics_dict = decoded_semantics.to_dict()
    decoded_semantics_abs_dict = decoded_semantics.abs().to_dict()

    wordcloud = WordCloud(background_color='white', width=800, height=1600)
    for nii in grad_nii_list:
        fname = '_'.join(nii.split('/')[-1].split('.')[0].split('_')[2:])
        cloud = wordcloud.generate_from_frequencies(decoded_semantics_dict[nii])
        cloud.to_file(os.path.join(opt.expdir, opt.savedir, f'{fname}.png'))
        del cloud
        cloud_abs = wordcloud.generate_from_frequencies(decoded_semantics_abs_dict[nii])
        cloud_abs.to_file(os.path.join(opt.expdir, opt.savedir, f'{fname}_abs.png'))
        del cloud_abs

    del decoded_semantics

    # decode cam saliency
    cam_nii_list = [os.path.join(opt.expdir, opt.saliencydir, f'saliency_normalized_cam_female_early.nii'),
                os.path.join(opt.expdir, opt.saliencydir, f'saliency_normalized_cam_male_early.nii')]

    decoded_semantics = decoder.decode(cam_nii_list, save=os.path.join(opt.expdir, opt.savedir, f'cam_decoded_table.csv')
    decoded_semantics_dict = decoded_semantics.to_dict()
    decoded_semantics_abs_dict = decoded_semantics.abs().to_dict()

    for nii in cam_nii_list:
        fname = '_'.join(nii.split('/')[-1].split('.')[0].split('_')[2:])
        cloud = wordcloud.generate_from_frequencies(decoded_semantics_dict[nii])
        cloud.to_file(os.path.join(opt.expdir, opt.savedir, f'{fname}.png'))
        del cloud
        cloud_abs = wordcloud.generate_from_frequencies(decoded_semantics_abs_dict[nii])
        cloud_abs.to_file(os.path.join(opt.expdir, opt.savedir, f'{fname}_abs.png'))
        del cloud_abs



if __name__ == '__main__':
    main()
