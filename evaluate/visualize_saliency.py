# mricrogl script
import os
import argparse
parser = argparse.ArgumentParser(description='MRIcroGL visualization script')
parser.add_argument('--expdir', type=str, default='results/graph_neural_mapping', help='path to the experiment')
parser.add_argument('--saliency', type=str, default='saliency_normalized_grad_female_early_top20', help='path containing the result.csv and the test_sequence.csv')
parser.add_argument('--saliencydir', type=str, default='saliency_nii', help='path containing the result.csv and the test_sequence.csv')
opt = parser.parse_args()

import gl
gl.resetdefaults()
gl.loadimage('mni152')
gl.overlayload(os.path.join(opt.expdir, opt.saliencydir, f'{opt.saliency}.nii'))
gl.overlayminmax(1, 0.7, 1.0)
gl.overlaycolorname('4hot')
gl.colorbarposition(4)
