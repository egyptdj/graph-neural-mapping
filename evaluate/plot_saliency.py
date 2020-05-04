import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib


def main():
    parser = argparse.ArgumentParser(description='Plot the saliency map in the nifti format')
    parser.add_argument('--expdir', type=str, default='results/graph_neural_mapping', help='path containing the saliency_female.npy and the saliency_male.npy')
    parser.add_argument('--roidir', type=str, default='data/roi/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz', help='path containing the used ROI file')
    parser.add_argument('--roimetadir', type=str, default='data/roi/7_400.txt', help='path containing the metadata of the ROI file')
    parser.add_argument('--threshold', type=float, default=0.0, help='threshold value of salient regions')
    parser.add_argument('--savedir', type=str, default='saliency_nii', help='path to save the saliency nii files within the expdir')
    parser.add_argument('--fold_idx', nargs='+', default=['0','1','2','3','4','5','6','7','8','9'], help='fold indices')

    opt = parser.parse_args()

    os.makedirs(os.path.join(opt.expdir, opt.savedir), exist_ok=True)

    roiimg = nib.load(opt.roidir)
    roiimgarray = roiimg.get_fdata()
    roiimgaffine = roiimg.affine
    roimeta = pd.read_csv(opt.roimetadir, index_col=0, header=None, delimiter='\t')

    # plot gradient based saliency
    saliency0 = []
    saliency1 = []
    saliency0_early = []
    saliency1_early = []
    labels = []

    for current_fold in opt.fold_idx:
        saliency0.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'grad_saliency_female.npy')))
        saliency1.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'grad_saliency_male.npy')))
        saliency0_early.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'grad_saliency_female_early.npy')))
        saliency1_early.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'grad_saliency_male_early.npy')))
        labels.append(np.load(os.path.join(opt.expdir, 'latent', str(current_fold), 'labels.npy')))

    saliency0_diag = np.diagonal(np.concatenate(saliency0, 0), axis1=1, axis2=2)
    saliency1_diag = np.diagonal(np.concatenate(saliency1, 0), axis1=1, axis2=2)
    saliency0early_diag = np.diagonal(np.concatenate(saliency0_early, 0), axis1=1, axis2=2)
    saliency1early_diag = np.diagonal(np.concatenate(saliency1_early, 0), axis1=1, axis2=2)

    saliency0_avg1 = np.mean(np.concatenate(saliency0, 0), axis=1)
    saliency1_avg1 = np.mean(np.concatenate(saliency1, 0), axis=1)
    saliency0early_avg1 = np.mean(np.concatenate(saliency0_early, 0), axis=1)
    saliency1early_avg1 = np.mean(np.concatenate(saliency1_early, 0), axis=1)

    saliency0_avg2 = np.mean(np.concatenate(saliency0, 0), axis=2)
    saliency1_avg2 = np.mean(np.concatenate(saliency1, 0), axis=2)
    saliency0early_avg2 = np.mean(np.concatenate(saliency0_early, 0), axis=2)
    saliency1early_avg2 = np.mean(np.concatenate(saliency1_early, 0), axis=2)

    labels = np.concatenate(labels, 0).squeeze()
    female_index = np.where(labels==0)
    male_index = np.where(labels==1)

    saliency0subjects = []
    saliency1subjects = []
    saliency0earlysubjects = []
    saliency1earlysubjects = []

    for subjidx, (sal0, sal1, sal0early, sal1early) in enumerate(zip(saliency0_diag, saliency1_diag, saliency0early_diag, saliency1early_diag)):
        saliency0array = roiimgarray.copy()
        saliency1array = roiimgarray.copy()
        saliency0earlyarray = roiimgarray.copy()
        saliency1earlyarray = roiimgarray.copy()
        print("EXTRACTING GRAD-SALIENCY SUBJECT: {} (LABEL:{})".format(subjidx, labels[subjidx]))
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

    normalized_array = plot_nii(saliency0subjects, roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_female_final')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_female_final')
    normalized_array = plot_nii(saliency1subjects, roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_male_final')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_male_final')
    normalized_array = plot_nii(saliency0earlysubjects, roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_female_early')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_female_early')
    normalized_array = plot_nii(saliency1earlysubjects, roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_male_early')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_male_early')

    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency0subjects) if labels[idx]==0], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_female_femalesubj_final')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_female_femalesubj_final')
    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency0subjects) if labels[idx]==1], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_female_malesubj_final')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_female_malesubj_final')
    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency1subjects) if labels[idx]==0], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_male_femalesubj_final')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_male_femalesubj_final')
    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency1subjects) if labels[idx]==1], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_male_malesubj_final')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_male_malesubj_final')
    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency0earlysubjects) if labels[idx]==0], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_female_femalesubj_early')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_female_femalesubj_early')
    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency0earlysubjects) if labels[idx]==1], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_female_malesubj_early')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_female_malesubj_early')
    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency1earlysubjects) if labels[idx]==0], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_male_femalesubj_early')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_male_femalesubj_early')
    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency1earlysubjects) if labels[idx]==1], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_male_malesubj_early')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_male_malesubj_early')


    # plot gradient based saliency with averaging axis 1
    saliency0subjects = []
    saliency1subjects = []
    saliency0earlysubjects = []
    saliency1earlysubjects = []

    for subjidx, (sal0, sal1, sal0early, sal1early) in enumerate(zip(saliency0_avg1, saliency1_avg1, saliency0early_avg1, saliency1early_avg1)):
        saliency0array = roiimgarray.copy()
        saliency1array = roiimgarray.copy()
        saliency0earlyarray = roiimgarray.copy()
        saliency1earlyarray = roiimgarray.copy()
        print("EXTRACTING GRAD-AVG1-SALIENCY SUBJECT: {} (LABEL:{})".format(subjidx, labels[subjidx]))
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

    normalized_array = plot_nii(saliency0subjects, roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_avg1_female_final')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_avg1_female_final')
    normalized_array = plot_nii(saliency1subjects, roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_avg1_male_final')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_avg1_male_final')
    normalized_array = plot_nii(saliency0earlysubjects, roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_avg1_female_early')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_avg1_female_early')
    normalized_array = plot_nii(saliency1earlysubjects, roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_avg1_male_early')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_avg1_male_early')

    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency0subjects) if labels[idx]==0], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_avg1_female_femalesubj_final')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_avg1_female_femalesubj_final')
    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency0subjects) if labels[idx]==1], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_avg1_female_malesubj_final')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_avg1_female_malesubj_final')
    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency1subjects) if labels[idx]==0], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_avg1_male_femalesubj_final')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_avg1_male_femalesubj_final')
    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency1subjects) if labels[idx]==1], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_avg1_male_malesubj_final')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_avg1_male_malesubj_final')
    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency0earlysubjects) if labels[idx]==0], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_avg1_female_femalesubj_early')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_avg1_female_femalesubj_early')
    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency0earlysubjects) if labels[idx]==1], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_avg1_female_malesubj_early')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_avg1_female_malesubj_early')
    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency1earlysubjects) if labels[idx]==0], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_avg1_male_femalesubj_early')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_avg1_male_femalesubj_early')
    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency1earlysubjects) if labels[idx]==1], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_avg1_male_malesubj_early')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_avg1_male_malesubj_early')

    # plot gradient based saliency with averaging axis 2
    saliency0subjects = []
    saliency1subjects = []
    saliency0earlysubjects = []
    saliency1earlysubjects = []

    for subjidx, (sal0, sal1, sal0early, sal1early) in enumerate(zip(saliency0_avg2, saliency1_avg2, saliency0early_avg2, saliency1early_avg2)):
        saliency0array = roiimgarray.copy()
        saliency1array = roiimgarray.copy()
        saliency0earlyarray = roiimgarray.copy()
        saliency1earlyarray = roiimgarray.copy()
        print("EXTRACTING GRAD-AVG2-SALIENCY SUBJECT: {} (LABEL:{})".format(subjidx, labels[subjidx]))
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

    normalized_array = plot_nii(saliency0subjects, roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_avg2_female_final')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_avg2_female_final')
    normalized_array = plot_nii(saliency1subjects, roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_avg2_male_final')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_avg2_male_final')
    normalized_array = plot_nii(saliency0earlysubjects, roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_avg2_female_early')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_avg2_female_early')
    normalized_array = plot_nii(saliency1earlysubjects, roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_avg2_male_early')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_avg2_male_early')

    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency0subjects) if labels[idx]==0], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_avg2_female_femalesubj_final')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_avg2_female_femalesubj_final')
    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency0subjects) if labels[idx]==1], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_avg2_female_malesubj_final')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_avg2_female_malesubj_final')
    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency1subjects) if labels[idx]==0], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_avg2_male_femalesubj_final')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_avg2_male_femalesubj_final')
    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency1subjects) if labels[idx]==1], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_avg2_male_malesubj_final')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_avg2_male_malesubj_final')
    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency0earlysubjects) if labels[idx]==0], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_avg2_female_femalesubj_early')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_avg2_female_femalesubj_early')
    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency0earlysubjects) if labels[idx]==1], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_avg2_female_malesubj_early')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_avg2_female_malesubj_early')
    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency1earlysubjects) if labels[idx]==0], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_avg2_male_femalesubj_early')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_avg2_male_femalesubj_early')
    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency1earlysubjects) if labels[idx]==1], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'grad_avg2_male_malesubj_early')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'grad_avg2_male_malesubj_early')

    # plot cam based saliency
    saliency0 = []
    saliency1 = []
    saliency0_early = []
    saliency1_early = []
    labels = []

    for current_fold in opt.fold_idx:
        saliency0.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'cam_saliency_female.npy')))
        saliency1.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'cam_saliency_male.npy')))
        saliency0_early.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'cam_saliency_female_early.npy')))
        saliency1_early.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'cam_saliency_male_early.npy')))
        labels.append(np.load(os.path.join(opt.expdir, 'latent', str(current_fold), 'labels.npy')))

    saliency0 = np.concatenate(saliency0, 0)
    saliency1 = np.concatenate(saliency1, 0)
    saliency0early = np.concatenate(saliency0_early, 0)
    saliency1early = np.concatenate(saliency1_early, 0)
    labels = np.concatenate(labels, 0).squeeze()
    female_index = np.where(labels==0)
    male_index = np.where(labels==1)

    saliency0subjects = []
    saliency1subjects = []
    saliency0earlysubjects = []
    saliency1earlysubjects = []

    for subjidx, (sal0, sal1, sal0early, sal1early) in enumerate(zip(saliency0, saliency1, saliency0early, saliency1early)):
        saliency0array = roiimgarray.copy()
        saliency1array = roiimgarray.copy()
        saliency0earlyarray = roiimgarray.copy()
        saliency1earlyarray = roiimgarray.copy()
        print("EXTRACTING CAM-SALIENCY SUBJECT: {} (LABEL:{})".format(subjidx, labels[subjidx]))
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

    normalized_array = plot_nii(saliency0subjects, roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'cam_female_final')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'cam_female_final')
    normalized_array = plot_nii(saliency1subjects, roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'cam_male_final')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'cam_male_final')
    normalized_array = plot_nii(saliency0earlysubjects, roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'cam_female_early')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'cam_female_early')
    normalized_array = plot_nii(saliency1earlysubjects, roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'cam_male_early')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'cam_male_early')

    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency0subjects) if labels[idx]==0], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'cam_female_femalesubj_final')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'cam_female_femalesubj_final')
    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency0subjects) if labels[idx]==1], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'cam_female_malesubj_final')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'cam_female_malesubj_final')
    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency1subjects) if labels[idx]==0], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'cam_male_femalesubj_final')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'cam_male_femalesubj_final')
    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency1subjects) if labels[idx]==1], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'cam_male_malesubj_final')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'cam_male_malesubj_final')
    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency0earlysubjects) if labels[idx]==0], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'cam_female_femalesubj_early')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'cam_female_femalesubj_early')
    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency0earlysubjects) if labels[idx]==1], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'cam_female_malesubj_early')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'cam_female_malesubj_early')
    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency1earlysubjects) if labels[idx]==0], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'cam_male_femalesubj_early')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'cam_male_femalesubj_early')
    normalized_array = plot_nii([subject for idx, subject in enumerate(saliency1earlysubjects) if labels[idx]==1], roiimgaffine, os.path.join(opt.expdir, opt.savedir), 'cam_male_malesubj_early')
    write_csv(normalized_array, roiimgarray, roimeta, opt.threshold, os.path.join(opt.expdir, opt.savedir), 'cam_male_malesubj_early')


def plot_nii(subject_list, roiimgaffine, savepath, desc):
    saliency_array = np.mean(np.stack(subject_list), axis=0)

    saliency_array_normalized = saliency_array.copy()
    saliency_array_normalized /= np.abs(saliency_array_normalized).max()

    zero_idx = np.where(saliency_array_normalized==0)
    saliency_img = nib.Nifti1Image(saliency_array, roiimgaffine)
    saliency_img_normalized = nib.Nifti1Image(saliency_array_normalized, roiimgaffine)

    nib.save(saliency_img, os.path.join(savepath, 'saliency_{}.nii'.format(desc)))
    nib.save(saliency_img_normalized, os.path.join(savepath, 'saliency_normalized_{}.nii'.format(desc)))
    del saliency_array
    del saliency_img
    del saliency_img_normalized
    return saliency_array_normalized

def write_csv(normalized_array, roiimgarray, roimeta, threshold, savepath, desc):
    assert threshold >=0.0 and threshold <= 1.0
    normalized_idx = (normalized_array>threshold).astype(np.uint8)+(normalized_array<-threshold).astype(np.uint8)
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

    with open(os.path.join(savepath, 'saliency_{}.csv'.format(desc)), 'w') as f:
        f.write('abs_value,roi,label,value\n')
        for item in zipped:
            f.write(','.join(item))
            f.write('\n')

if __name__ == '__main__':
    main()
