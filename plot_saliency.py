import os
import argparse
import numpy as np
import nibabel as nib


def main():
    parser = argparse.ArgumentParser(description='Plot the saliency map in the nifti format')
    parser.add_argument('--datadir', nargs='+', default=None, help='path containing the saliency0.npy and the saliency1.npy')
    parser.add_argument('--roidir', type=str, default='data/roi/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz', help='path containing the used ROI file')
    parser.add_argument('--savedir', type=str, default='./saliency_nii', help='path to save the saliency nii files')

    opt = parser.parse_args()

    os.makedirs(opt.savedir, exist_ok=True)

    saliency0 = []
    saliency1 = []

    for dir in opt.datadir:
        saliency0.append(np.load(os.path.join(dir, 'saliency0.npy')))
        saliency1.append(np.load(os.path.join(dir, 'saliency1.npy')))
        # saliency0.append(np.load(os.path.join(dir, 'saliency_female.npy')))
        # saliency1.append(np.load(os.path.join(dir, 'saliency_male.npy')))

    saliency0 = np.diagonal(np.concatenate(saliency0, 0), axis1=1, axis2=2)
    saliency1 = np.diagonal(np.concatenate(saliency1, 0), axis1=1, axis2=2)

    roiimg = nib.load(opt.roidir)
    roiimgarray = roiimg.get_fdata()
    roiimgaffine = roiimg.affine

    saliency0subjects = []
    saliency1subjects = []
    for subjidx, (sal0, sal1) in enumerate(zip(saliency0, saliency1)):
        saliency0array = roiimgarray.copy()
        saliency1array = roiimgarray.copy()
        print("EXTRACTING SUBJECT: {}".format(subjidx))
        for i, (s0, s1) in enumerate(zip(sal0, sal1)):
            roi_voxel_idx = np.where(roiimgarray==i+1)
            for j in range(roi_voxel_idx[0].shape[0]):
                saliency0array[roi_voxel_idx[0][j], roi_voxel_idx[1][j], roi_voxel_idx[2][j]] = s0
                saliency1array[roi_voxel_idx[0][j], roi_voxel_idx[1][j], roi_voxel_idx[2][j]] = s1

        saliency0subjects.append(saliency0array)
        saliency1subjects.append(saliency1array)


    saliency0array = np.mean(np.stack(saliency0subjects), axis=0)
    saliency1array = np.mean(np.stack(saliency1subjects), axis=0)
    saliency0array_normalized = saliency0array.copy()
    saliency1array_normalized = saliency1array.copy()
    if saliency0array_normalized.max() > np.abs(saliency0array_normalized).max():
        saliency0array_normalized /= saliency0array_normalized.max()
    else:
        saliency0array_normalized /= np.abs(saliency0array_normalized).max()
    if saliency1array_normalized.max() > np.abs(saliency1array_normalized).max():
        saliency1array_normalized /= saliency1array_normalized.max()
    else:
        saliency1array_normalized /= np.abs(saliency1array_normalized).max()

    zero_idx = np.where(saliency0array_normalized==0)
    saliency0img = nib.Nifti1Image(saliency0array, roiimgaffine)
    saliency1img = nib.Nifti1Image(saliency1array, roiimgaffine)
    saliency0img_normalized = nib.Nifti1Image(saliency0array_normalized, roiimgaffine)
    saliency1img_normalized = nib.Nifti1Image(saliency1array_normalized, roiimgaffine)

    saliency0_normalized_idx = (saliency0array_normalized>0.7).astype(np.uint8)+(saliency0array_normalized<-0.7).astype(np.uint8)
    saliency0_idx_tuple = np.nonzero(saliency0_normalized_idx)
    saliency0_rois = []
    saliency0_values = []
    for i in range(len(saliency0_idx_tuple[0])):
        roi = roiimgarray[saliency0_idx_tuple[0][i],saliency0_idx_tuple[1][i],saliency0_idx_tuple[2][i]]
        value = saliency0array_normalized[saliency0_idx_tuple[0][i],saliency0_idx_tuple[1][i],saliency0_idx_tuple[2][i]]
        if str(roi) not in saliency0_rois:
            assert value not in saliency0_values
            saliency0_rois.append(str(roi))
            saliency0_values.append(str(value))

    saliency1_normalized_idx = (saliency1array_normalized>0.7).astype(np.uint8)+(saliency1array_normalized<-0.7).astype(np.uint8)
    saliency1_idx_tuple = np.nonzero(saliency1_normalized_idx)
    saliency1_rois = []
    saliency1_values = []
    for i in range(len(saliency1_idx_tuple[0])):
        roi = roiimgarray[saliency1_idx_tuple[0][i],saliency1_idx_tuple[1][i],saliency1_idx_tuple[2][i]]
        value = saliency1array_normalized[saliency1_idx_tuple[0][i],saliency1_idx_tuple[1][i],saliency1_idx_tuple[2][i]]
        if str(roi) not in saliency1_rois:
            assert value not in saliency1_values
            saliency1_rois.append(str(roi))
            saliency1_values.append(str(value))

    with open(os.path.join(opt.savedir, 'saliency_female.csv'), 'w') as f:
        f.write(','.join(saliency0_rois))
        f.write("\n")
        f.write(','.join(saliency0_values))

    with open(os.path.join(opt.savedir, 'saliency_male.csv'), 'w') as f:
        f.write(','.join(saliency1_rois))
        f.write("\n")
        f.write(','.join(saliency1_values))

    nib.save(saliency0img, os.path.join(opt.savedir, 'saliency_female.nii'))
    nib.save(saliency1img, os.path.join(opt.savedir, 'saliency_male.nii'))
    nib.save(saliency0img_normalized, os.path.join(opt.savedir, 'saliency_normalized_female.nii'))
    nib.save(saliency1img_normalized, os.path.join(opt.savedir, 'saliency_normalized_male.nii'))

if __name__ == '__main__':
    main()
