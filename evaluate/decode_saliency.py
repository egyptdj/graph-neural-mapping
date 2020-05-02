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
    parser.add_argument('--abs', action='store_true', help='decode semantics from absolute saliency values')

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

    nii_list = [f'{opt.expdir}/{opt.saliencydir}/saliency_normalized_female_early.nii',
                f'{opt.expdir}/{opt.saliencydir}/saliency_normalized_male_early.nii',
                f'{opt.expdir}/{opt.saliencydir}/saliency_normalized_female_femalesubj_early.nii',
                f'{opt.expdir}/{opt.saliencydir}/saliency_normalized_male_malesubj_early.nii']

    decoded_semantics = decoder.decode(nii_list)
    if opt.abs: decoded_semantics = decoded_semantics.abs()
    decoded_semantics = decoded_semantics.to_dict()

    wordcloud = WordCloud(background_color='white', width=1600, height=800)
    for nii in nii_list:
        fname = '_'.join(nii.split('/')[-1].split('.')[0].split('_')[2:])
        cloud = wordcloud.generate_from_frequencies(decoded_semantics[nii])
        cloud.to_file(f'{opt.expdir}/{opt.savedir}/{fname}.png')


if __name__ == '__main__':
    main()
