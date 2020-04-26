import os
import argparse
import numpy as np
import nibabel as nib


def main():
    parser = argparse.ArgumentParser(description='Plot the saliency map in the nifti format')
    parser.add_argument('--expdir', type=str, default='results/graph_neural_mapping', help='path containing the saliency_female.npy and the saliency_male.npy')
    parser.add_argument('--roidir', type=str, default='data/roi/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz', help='path containing the used ROI file')
    parser.add_argument('--savedir', type=str, default='saliency_nii', help='path to save the saliency nii files within the expdir')
    parser.add_argument('--fold_idx', nargs='+', default=[0,1,2,3,4,5,6,7,8,9], help='fold indices')

    opt = parser.parse_args()

    os.makedirs(os.path.join(opt.expdir, opt.savedir), exist_ok=True)

    saliency0 = []
    saliency1 = []
    saliency0_early = []
    saliency1_early = []

    for current_fold in opt.fold_idx:
        saliency0.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'saliency_female.npy')))
        saliency1.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'saliency_male.npy')))
        saliency0_early.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'saliency_female_early.npy')))
        saliency1_early.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'saliency_male_early.npy')))
    
    saliency0 = np.diagonal(np.concatenate(saliency0, 0), axis1=1, axis2=2)
    saliency1 = np.diagonal(np.concatenate(saliency1, 0), axis1=1, axis2=2)
    saliency0early = np.diagonal(np.concatenate(saliency0_early, 0), axis1=1, axis2=2)
    saliency1early = np.diagonal(np.concatenate(saliency1_early, 0), axis1=1, axis2=2)

    roiimg = nib.load(opt.roidir)
    roiimgarray = roiimg.get_fdata()
    roiimgaffine = roiimg.affine

    saliency0subjects = []
    saliency1subjects = []
    saliency0earlysubjects = []
    saliency1earlysubjects = []

    for subjidx, (sal0, sal1, sal0early, sal1early) in enumerate(zip(saliency0, saliency1, saliency0early, saliency1early)):
        saliency0array = roiimgarray.copy()
        saliency1array = roiimgarray.copy()
        saliency0earlyarray = roiimgarray.copy()
        saliency1earlyarray = roiimgarray.copy()
        print("EXTRACTING SUBJECT: {}".format(subjidx))
        for i, (s0, s1, s0e, s1e) in enumerate(zip(sal0, sal1, sal0early, sal1early)):
            roi_voxel_idx = np.where(roiimgarray==i+1)
            for j in range(roi_voxel_idx[0].shape[0]):
                saliency0array[roi_voxel_idx[0][j], roi_voxel_idx[1][j], roi_voxel_idx[2][j]] = s0
                saliency1array[roi_voxel_idx[0][j], roi_voxel_idx[1][j], roi_voxel_idx[2][j]] = s1
                saliency0earlyarray[roi_voxel_idx[0][j], roi_voxel_idx[1][j], roi_voxel_idx[2][j]] = s0e
                saliency1earlyarray[roi_voxel_idx[0][j], roi_voxel_idx[1][j], roi_voxel_idx[2][j]] = s1e

        saliency0subjects.append(saliency0array)
        saliency1subjects.append(saliency1array)
        saliency0earlysubjects.append(saliency0earlyarray)
        saliency1earlysubjects.append(saliency1earlyarray)


    saliency0array = np.mean(np.stack(saliency0subjects), axis=0)
    saliency1array = np.mean(np.stack(saliency1subjects), axis=0)
    saliency0earlyarray = np.mean(np.stack(saliency0earlysubjects), axis=0)
    saliency1earlyarray = np.mean(np.stack(saliency1earlysubjects), axis=0)
    saliency0array_normalized = saliency0array.copy()
    saliency1array_normalized = saliency1array.copy()
    saliency0earlyarray_normalized = saliency0earlyarray.copy()
    saliency1earlyarray_normalized = saliency1earlyarray.copy()
    
    saliency0array_normalized /= np.abs(saliency0array_normalized).max()
    saliency1array_normalized /= np.abs(saliency1array_normalized).max()
    saliency0earlyarray_normalized /= np.abs(saliency0earlyarray_normalized).max()
    saliency1earlyarray_normalized /= np.abs(saliency1earlyarray_normalized).max()

    zero_idx = np.where(saliency0array_normalized==0)
    saliency0img = nib.Nifti1Image(saliency0array, roiimgaffine)
    saliency1img = nib.Nifti1Image(saliency1array, roiimgaffine)
    saliency0earlyimg = nib.Nifti1Image(saliency0earlyarray, roiimgaffine)
    saliency1earlyimg = nib.Nifti1Image(saliency1earlyarray, roiimgaffine)
    saliency0img_normalized = nib.Nifti1Image(saliency0array_normalized, roiimgaffine)
    saliency1img_normalized = nib.Nifti1Image(saliency1array_normalized, roiimgaffine)
    saliency0earlyimg_normalized = nib.Nifti1Image(saliency0earlyarray_normalized, roiimgaffine)
    saliency1earlyimg_normalized = nib.Nifti1Image(saliency1earlyarray_normalized, roiimgaffine)

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

    saliency0early_normalized_idx = (saliency0earlyarray_normalized>0.7).astype(np.uint8)+(saliency0earlyarray_normalized<-0.7).astype(np.uint8)
    saliency0early_idx_tuple = np.nonzero(saliency0early_normalized_idx)
    saliency0early_rois = []
    saliency0early_values = []
    for i in range(len(saliency0early_idx_tuple[0])):
        roi = roiimgarray[saliency0early_idx_tuple[0][i],saliency0early_idx_tuple[1][i],saliency0early_idx_tuple[2][i]]
        value = saliency0earlyarray_normalized[saliency0early_idx_tuple[0][i],saliency0early_idx_tuple[1][i],saliency0early_idx_tuple[2][i]]
        if str(roi) not in saliency0early_rois:
            assert value not in saliency0early_values
            saliency0early_rois.append(str(roi))
            saliency0early_values.append(str(value))

    saliency1early_normalized_idx = (saliency1earlyarray_normalized>0.7).astype(np.uint8)+(saliency1earlyarray_normalized<-0.7).astype(np.uint8)
    saliency1early_idx_tuple = np.nonzero(saliency1early_normalized_idx)
    saliency1early_rois = []
    saliency1early_values = []
    for i in range(len(saliency1early_idx_tuple[0])):
        roi = roiimgarray[saliency1early_idx_tuple[0][i],saliency1early_idx_tuple[1][i],saliency1early_idx_tuple[2][i]]
        value = saliency1earlyarray_normalized[saliency1early_idx_tuple[0][i],saliency1early_idx_tuple[1][i],saliency1early_idx_tuple[2][i]]
        if str(roi) not in saliency1early_rois:
            assert value not in saliency1early_values
            saliency1early_rois.append(str(roi))
            saliency1early_values.append(str(value))

    with open(os.path.join(opt.expdir, opt.savedir, 'saliency_female_final.csv'), 'w') as f:
        f.write(','.join(saliency0_rois))
        f.write("\n")
        f.write(','.join(saliency0_values))
        f.write("\n")
        
    with open(os.path.join(opt.expdir, opt.savedir, 'saliency_male_final.csv'), 'w') as f:
        f.write(','.join(saliency1_rois))
        f.write("\n")
        f.write(','.join(saliency1_values))

    with open(os.path.join(opt.expdir, opt.savedir, 'saliency_female_early.csv'), 'w') as f:
        f.write(','.join(saliency0early_rois))
        f.write("\n")
        f.write(','.join(saliency0early_values))
        f.write("\n")
        
    with open(os.path.join(opt.expdir, opt.savedir, 'saliency_male_early.csv'), 'w') as f:
        f.write(','.join(saliency1early_rois))
        f.write("\n")
        f.write(','.join(saliency1early_values))


    nib.save(saliency0img, os.path.join(opt.expdir, opt.savedir, 'saliency_female_final.nii'))
    nib.save(saliency1img, os.path.join(opt.expdir, opt.savedir, 'saliency_male_final.nii'))
    nib.save(saliency0img_normalized, os.path.join(opt.expdir, opt.savedir, 'saliency_normalized_female_final.nii'))
    nib.save(saliency1img_normalized, os.path.join(opt.expdir, opt.savedir, 'saliency_normalized_male_final.nii'))

    nib.save(saliency0earlyimg, os.path.join(opt.expdir, opt.savedir, 'saliency_female_early.nii'))
    nib.save(saliency1earlyimg, os.path.join(opt.expdir, opt.savedir, 'saliency_male_early.nii'))
    nib.save(saliency0earlyimg_normalized, os.path.join(opt.expdir, opt.savedir, 'saliency_normalized_female_early.nii'))
    nib.save(saliency1earlyimg_normalized, os.path.join(opt.expdir, opt.savedir, 'saliency_normalized_male_early.nii'))


if __name__ == '__main__':
    main()
