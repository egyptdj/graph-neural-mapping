import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import nilearn as nil


def main():
    parser = argparse.ArgumentParser(description='Plot the saliency map in the nifti format')
    parser.add_argument('--expdir', type=str, default='results/graph_neural_mapping', help='path containing the saliency_female.npy and the saliency_male.npy')
    parser.add_argument('--roidir', type=str, default='data/roi/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz', help='path containing the nifti ROI file')
    parser.add_argument('--roimetadir', type=str, default='data/roi/7_400.txt', help='path containing the metadata of the ROI file')
    parser.add_argument('--topk', type=int, default=20, help='top k rois to visualize')
    parser.add_argument('--savedir', type=str, default='saliency_nii', help='path to save the saliency nii files within the expdir')
    parser.add_argument('--fold_idx', nargs='+', default=['0','1','2','3','4','5','6','7','8','9'], help='fold indices')

    opt = parser.parse_args()

    os.makedirs(os.path.join(opt.expdir, opt.savedir), exist_ok=True)
    os.makedirs(os.path.join(opt.expdir, opt.savedir, 'network'), exist_ok=True)
    os.makedirs(os.path.join(opt.expdir, opt.savedir, 'description'), exist_ok=True)

    roiimg = nil.image.load_img(opt.roidir)
    roiimgarray = roiimg.get_fdata()
    roiimgaffine = roiimg.affine
    roimeta = pd.read_csv(opt.roimetadir, index_col=0, header=None, delimiter='\t')

    # plot proposed based saliency
    saliency0 = []
    saliency1 = []

    for current_fold in opt.fold_idx:
        saliency0.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'saliency_female.npy')))
        saliency1.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'saliency_male.npy')))

    saliency0 = np.mean(np.concatenate(saliency0, 0), axis=1)
    saliency1 = np.mean(np.concatenate(saliency1, 0), axis=1)

    saliency0subjects = []
    saliency1subjects = []

    for subjidx, (sal0, sal1) in enumerate(zip(saliency0, saliency1)):
        saliency0array = roiimgarray.copy()
        saliency1array = roiimgarray.copy()
        print("EXTRACTING SALIENCY SUBJECT: {}".format(subjidx))
        for i, (s0, s1) in enumerate(zip(sal0, sal1)):
            roi_voxel_idx = np.where(roiimgarray==i+1)
            for j in range(roi_voxel_idx[0].shape[0]):
                saliency0array[roi_voxel_idx[0][j], roi_voxel_idx[1][j], roi_voxel_idx[2][j]] = s0
                saliency1array[roi_voxel_idx[0][j], roi_voxel_idx[1][j], roi_voxel_idx[2][j]] = s1

        saliency0subjects.append(saliency0array)
        saliency1subjects.append(saliency1array)

    plot_nii(saliency0subjects, opt.topk, roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'female')
    plot_nii(saliency1subjects, opt.topk, roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'male')


def plot_nii(subject_list, topk, roiimgaffine, roiimgarray, roimeta, savepath, desc):
    saliency_array = np.mean(np.stack(subject_list), axis=0)

    saliency_array_normalized = saliency_array.copy()
    saliency_array_normalized -= saliency_array_normalized.min()
    saliency_array_normalized /= saliency_array_normalized.max()

    if topk:
        values = np.unique(saliency_array_normalized)
        topk_idx = np.argsort(values)[-topk]
        topk_value = values[topk_idx]
        saliency_array_normalized_topk = saliency_array_normalized.copy()
        saliency_array_normalized_topk[saliency_array_normalized_topk<topk_value]=0.0
        saliency_img_normalized_topk = nib.Nifti1Image(saliency_array_normalized_topk, roiimgaffine)

        nib.save(saliency_img_normalized_topk, os.path.join(savepath, 'saliency_{}_top{}.nii'.format(desc, topk)))

        saliency_values = np.unique(saliency_array_normalized_topk)

        network_dicts = {'Vis':{'LH': np.zeros_like(saliency_array), 'RH': np.zeros_like(saliency_array)}, 'SomMot':{'LH': np.zeros_like(saliency_array), 'RH': np.zeros_like(saliency_array)}, 'DorsAttn':{'LH': np.zeros_like(saliency_array), 'RH': np.zeros_like(saliency_array)}, 'SalVentAttn':{'LH': np.zeros_like(saliency_array), 'RH': np.zeros_like(saliency_array)}, 'Limbic':{'LH': np.zeros_like(saliency_array), 'RH': np.zeros_like(saliency_array)}, 'Cont':{'LH': np.zeros_like(saliency_array), 'RH': np.zeros_like(saliency_array)}, 'Default':{'LH': np.zeros_like(saliency_array), 'RH': np.zeros_like(saliency_array)}}
        for value in saliency_values:
            if value==0.0: continue
            roi_array = saliency_array_normalized_topk.copy()
            roi_array[saliency_array_normalized_topk!=value] = 0.0
            idx_tuple = np.nonzero(roi_array)
            roi_id = roiimgarray[idx_tuple[0][0], idx_tuple[1][0], idx_tuple[2][0]]
            roi_network = roimeta[1][roi_id]
            for key in network_dicts.keys():
                if key in roi_network:
                    if 'LH' in roi_network:
                        network_dicts[key]['LH'] += roi_array
                    elif 'RH' in roi_network:
                        network_dicts[key]['RH'] += roi_array
                    else:
                        print('ERROR IDENTIFYING HEMISPHERE INFORMATION')
        for key in network_dicts.keys():
            network_lh_img = nib.Nifti1Image(network_dicts[key]['LH'], roiimgaffine)
            nib.save(network_lh_img, os.path.join(savepath, 'network', 'saliency_{}_top{}_{}_lh'.format(desc, topk, key)))
            network_rh_img = nib.Nifti1Image(network_dicts[key]['RH'], roiimgaffine)
            nib.save(network_rh_img, os.path.join(savepath, 'network', 'saliency_{}_top{}_{}_rh'.format(desc, topk, key)))

    saliency_img_normalized = nib.Nifti1Image(saliency_array_normalized, roiimgaffine)
    nib.save(saliency_img_normalized, os.path.join(savepath, 'saliency_{}.nii'.format(desc)))


def write_csv(normalized_array, roiimgarray, roimeta, savepath, desc, threshold=None):
    if threshold:
        normalized_idx = (normalized_array>threshold).astype(np.uint8)
    else:
        normalized_idx = (roiimgarray!=0).astype(np.uint8)
    idx_tuple = np.nonzero(normalized_idx)
    rois = []
    values = []
    abs_values = []
    labels = []
    for i in range(len(idx_tuple[0])):
        roi = roiimgarray[idx_tuple[0][i],idx_tuple[1][i],idx_tuple[2][i]]
        value = normalized_array[idx_tuple[0][i],idx_tuple[1][i],idx_tuple[2][i]]
        if str(roi) not in rois:
            assert value not in values
            rois.append(str(roi))
            values.append(str(value))
            abs_values.append(str(abs(value)))
            labels.append(str(roimeta[1][roi]))
    zipped = list(zip(abs_values, rois, labels, values))
    zipped.sort(reverse=True)

    with open(os.path.join(savepath, 'description', 'saliency_{}.csv'.format(desc)), 'w') as f:
        f.write('abs_value,roi,label,value\n')
        for item in zipped:
            f.write(','.join(item))
            f.write('\n')


if __name__ == '__main__':
    main()
