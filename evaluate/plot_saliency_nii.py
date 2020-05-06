import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import nilearn as nil


def main():
    parser = argparse.ArgumentParser(description='Plot the saliency map in the nifti format')
    parser.add_argument('--expdir', type=str, default='results/graph_neural_mapping', help='path containing the saliency_female.npy and the saliency_male.npy')
    parser.add_argument('--roidir', type=str, default='data/roi/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz', help='path containing the used ROI file')
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

    # plot gradient based saliency
    saliency0 = []
    saliency1 = []
    saliency0_early = []
    saliency1_early = []
    # labels = []

    for current_fold in opt.fold_idx:
        saliency0.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'grad_saliency_female.npy')))
        saliency1.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'grad_saliency_male.npy')))
        saliency0_early.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'grad_saliency_female_early.npy')))
        saliency1_early.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'grad_saliency_male_early.npy')))
        # labels.append(np.load(os.path.join(opt.expdir, 'latent', str(current_fold), 'labels.npy')))

    saliency0 = np.mean(np.concatenate(saliency0, 0), axis=1)
    saliency1 = np.mean(np.concatenate(saliency1, 0), axis=1)
    saliency0early = np.mean(np.concatenate(saliency0_early, 0), axis=1)
    saliency1early = np.mean(np.concatenate(saliency1_early, 0), axis=1)

    # labels = np.concatenate(labels, 0).squeeze()
    # female_index = np.where(labels==0)
    # male_index = np.where(labels==1)

    saliency0subjects = []
    saliency1subjects = []
    saliency0earlysubjects = []
    saliency1earlysubjects = []

    for subjidx, (sal0, sal1, sal0early, sal1early) in enumerate(zip(saliency0, saliency1, saliency0early, saliency1early)):
        saliency0array = roiimgarray.copy()
        saliency1array = roiimgarray.copy()
        saliency0earlyarray = roiimgarray.copy()
        saliency1earlyarray = roiimgarray.copy()
        print("EXTRACTING GRAD-SALIENCY SUBJECT: {}".format(subjidx))
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

    plot_nii(saliency0subjects, opt.topk, roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'grad_female_final')
    plot_nii(saliency1subjects, opt.topk, roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'grad_male_final')
    plot_nii(saliency0earlysubjects, opt.topk, roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'grad_female_early')
    plot_nii(saliency1earlysubjects, opt.topk, roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'grad_male_early')

    # plot_nii([subject for idx, subject in enumerate(saliency0subjects) if labels[idx]==0], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'grad_female_femalesubj_final')
    # plot_nii([subject for idx, subject in enumerate(saliency0subjects) if labels[idx]==1], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'grad_female_malesubj_final')
    # plot_nii([subject for idx, subject in enumerate(saliency1subjects) if labels[idx]==0], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'grad_male_femalesubj_final')
    # plot_nii([subject for idx, subject in enumerate(saliency1subjects) if labels[idx]==1], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'grad_male_malesubj_final')
    # plot_nii([subject for idx, subject in enumerate(saliency0earlysubjects) if labels[idx]==0], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'grad_female_femalesubj_early')
    # plot_nii([subject for idx, subject in enumerate(saliency0earlysubjects) if labels[idx]==1], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'grad_female_malesubj_early')
    # plot_nii([subject for idx, subject in enumerate(saliency1earlysubjects) if labels[idx]==0], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'grad_male_femalesubj_early')
    # plot_nii([subject for idx, subject in enumerate(saliency1earlysubjects) if labels[idx]==1], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'grad_male_malesubj_early')

    # plot cam based saliency
    saliency0 = []
    saliency1 = []
    saliency0_early = []
    saliency1_early = []
    # labels = []

    for current_fold in opt.fold_idx:
        saliency0.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'cam_saliency_female.npy')))
        saliency1.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'cam_saliency_male.npy')))
        saliency0_early.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'cam_saliency_female_early.npy')))
        saliency1_early.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'cam_saliency_male_early.npy')))
        # labels.append(np.load(os.path.join(opt.expdir, 'latent', str(current_fold), 'labels.npy')))

    saliency0 = np.concatenate(saliency0, 0)
    saliency1 = np.concatenate(saliency1, 0)
    saliency0early = np.concatenate(saliency0_early, 0)
    saliency1early = np.concatenate(saliency1_early, 0)
    # labels = np.concatenate(labels, 0).squeeze()
    # female_index = np.where(labels==0)
    # male_index = np.where(labels==1)

    saliency0subjects = []
    saliency1subjects = []
    saliency0earlysubjects = []
    saliency1earlysubjects = []

    for subjidx, (sal0, sal1, sal0early, sal1early) in enumerate(zip(saliency0, saliency1, saliency0early, saliency1early)):
        saliency0array = roiimgarray.copy()
        saliency1array = roiimgarray.copy()
        saliency0earlyarray = roiimgarray.copy()
        saliency1earlyarray = roiimgarray.copy()
        print("EXTRACTING CAM-SALIENCY SUBJECT: {}".format(subjidx))
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

    plot_nii(saliency0subjects, opt.topk, roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'cam_female_final')
    plot_nii(saliency1subjects, opt.topk, roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'cam_male_final')
    plot_nii(saliency0earlysubjects, opt.topk, roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'cam_female_early')
    plot_nii(saliency1earlysubjects, opt.topk, roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'cam_male_early')

    # plot_nii([subject for idx, subject in enumerate(saliency0subjects) if labels[idx]==0], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'cam_female_femalesubj_final')
    # plot_nii([subject for idx, subject in enumerate(saliency0subjects) if labels[idx]==1], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'cam_female_malesubj_final')
    # plot_nii([subject for idx, subject in enumerate(saliency1subjects) if labels[idx]==0], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'cam_male_femalesubj_final')
    # plot_nii([subject for idx, subject in enumerate(saliency1subjects) if labels[idx]==1], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'cam_male_malesubj_final')
    # plot_nii([subject for idx, subject in enumerate(saliency0earlysubjects) if labels[idx]==0], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'cam_female_femalesubj_early')
    # plot_nii([subject for idx, subject in enumerate(saliency0earlysubjects) if labels[idx]==1], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'cam_female_malesubj_early')
    # plot_nii([subject for idx, subject in enumerate(saliency1earlysubjects) if labels[idx]==0], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'cam_male_femalesubj_early')
    # plot_nii([subject for idx, subject in enumerate(saliency1earlysubjects) if labels[idx]==1], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'cam_male_malesubj_early')

    # # plot gradcam based saliency
    # saliency0 = []
    # saliency1 = []
    # saliency0_early = []
    # saliency1_early = []
    # labels = []
    #
    # for current_fold in opt.fold_idx:
    #     saliency0.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'gradcam_saliency_female.npy')))
    #     saliency1.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'gradcam_saliency_male.npy')))
    #     saliency0_early.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'gradcam_saliency_female_early.npy')))
    #     saliency1_early.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'gradcam_saliency_male_early.npy')))
    #     labels.append(np.load(os.path.join(opt.expdir, 'latent', str(current_fold), 'labels.npy')))
    #
    # saliency0 = np.concatenate(saliency0, 0)
    # saliency1 = np.concatenate(saliency1, 0)
    # saliency0early = np.concatenate(saliency0_early, 0)
    # saliency1early = np.concatenate(saliency1_early, 0)
    # labels = np.concatenate(labels, 0).squeeze()
    # female_index = np.where(labels==0)
    # male_index = np.where(labels==1)
    #
    # saliency0subjects = []
    # saliency1subjects = []
    # saliency0earlysubjects = []
    # saliency1earlysubjects = []
    #
    # for subjidx, (sal0, sal1, sal0early, sal1early) in enumerate(zip(saliency0, saliency1, saliency0early, saliency1early)):
    #     saliency0array = roiimgarray.copy()
    #     saliency1array = roiimgarray.copy()
    #     saliency0earlyarray = roiimgarray.copy()
    #     saliency1earlyarray = roiimgarray.copy()
    #     print("EXTRACTING GRADCAM-SALIENCY SUBJECT: {} (LABEL:{})".format(subjidx, labels[subjidx]))
    #     for i, (s0, s1, s0e, s1e) in enumerate(zip(sal0, sal1, sal0early, sal1early)):
    #         roi_voxel_idx = np.where(roiimgarray==i+1)
    #         for j in range(roi_voxel_idx[0].shape[0]):
    #             saliency0array[roi_voxel_idx[0][j], roi_voxel_idx[1][j], roi_voxel_idx[2][j]] = s0
    #             saliency1array[roi_voxel_idx[0][j], roi_voxel_idx[1][j], roi_voxel_idx[2][j]] = s1
    #             saliency0earlyarray[roi_voxel_idx[0][j], roi_voxel_idx[1][j], roi_voxel_idx[2][j]] = s0e
    #             saliency1earlyarray[roi_voxel_idx[0][j], roi_voxel_idx[1][j], roi_voxel_idx[2][j]] = s1e
    #
    #     saliency0subjects.append(saliency0array)
    #     saliency1subjects.append(saliency1array)
    #     saliency0earlysubjects.append(saliency0earlyarray)
    #     saliency1earlysubjects.append(saliency1earlyarray)
    #
    # plot_nii(saliency0subjects, roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gradcam_female_final')
    # plot_nii(saliency1subjects, roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gradcam_male_final')
    # plot_nii(saliency0earlysubjects, roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gradcam_female_early')
    # plot_nii(saliency1earlysubjects, roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gradcam_male_early')
    #
    # plot_nii([subject for idx, subject in enumerate(saliency0subjects) if labels[idx]==0], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gradcam_female_femalesubj_final')
    # plot_nii([subject for idx, subject in enumerate(saliency0subjects) if labels[idx]==1], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gradcam_female_malesubj_final')
    # plot_nii([subject for idx, subject in enumerate(saliency1subjects) if labels[idx]==0], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gradcam_male_femalesubj_final')
    # plot_nii([subject for idx, subject in enumerate(saliency1subjects) if labels[idx]==1], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gradcam_male_malesubj_final')
    # plot_nii([subject for idx, subject in enumerate(saliency0earlysubjects) if labels[idx]==0], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gradcam_female_femalesubj_early')
    # plot_nii([subject for idx, subject in enumerate(saliency0earlysubjects) if labels[idx]==1], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gradcam_female_malesubj_early')
    # plot_nii([subject for idx, subject in enumerate(saliency1earlysubjects) if labels[idx]==0], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gradcam_male_femalesubj_early')
    # plot_nii([subject for idx, subject in enumerate(saliency1earlysubjects) if labels[idx]==1], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gradcam_male_malesubj_early')

    # # save saliency for guided gradcam
    # saliency0_gradcam = saliency0
    # saliency1_gradcam = saliency1
    # saliency0_gradcam_early = saliency0_early
    # saliency1_gradcam_early = saliency1_early
    #
    # # plot guided_backprop based saliency
    # saliency0 = []
    # saliency1 = []
    # saliency0_early = []
    # saliency1_early = []
    # labels = []
    #
    # for current_fold in opt.fold_idx:
    #     saliency0.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'gradcam_saliency_female.npy')))
    #     saliency1.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'gradcam_saliency_male.npy')))
    #     saliency0_early.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'gradcam_saliency_female_early.npy')))
    #     saliency1_early.append(np.load(os.path.join(opt.expdir, 'saliency', str(current_fold), 'gradcam_saliency_male_early.npy')))
    #     labels.append(np.load(os.path.join(opt.expdir, 'latent', str(current_fold), 'labels.npy')))
    #
    # saliency0 = np.concatenate(saliency0, 0)
    # saliency1 = np.concatenate(saliency1, 0)
    # saliency0early = np.concatenate(saliency0_early, 0)
    # saliency1early = np.concatenate(saliency1_early, 0)
    # labels = np.concatenate(labels, 0).squeeze()
    # female_index = np.where(labels==0)
    # male_index = np.where(labels==1)
    #
    # saliency0subjects = []
    # saliency1subjects = []
    # saliency0earlysubjects = []
    # saliency1earlysubjects = []
    #
    # for subjidx, (sal0, sal1, sal0early, sal1early) in enumerate(zip(saliency0, saliency1, saliency0early, saliency1early)):
    #     saliency0array = roiimgarray.copy()
    #     saliency1array = roiimgarray.copy()
    #     saliency0earlyarray = roiimgarray.copy()
    #     saliency1earlyarray = roiimgarray.copy()
    #     print("EXTRACTING GUIDEDBACKPROP-SALIENCY SUBJECT: {} (LABEL:{})".format(subjidx, labels[subjidx]))
    #     for i, (s0, s1, s0e, s1e) in enumerate(zip(sal0, sal1, sal0early, sal1early)):
    #         roi_voxel_idx = np.where(roiimgarray==i+1)
    #         for j in range(roi_voxel_idx[0].shape[0]):
    #             saliency0array[roi_voxel_idx[0][j], roi_voxel_idx[1][j], roi_voxel_idx[2][j]] = s0
    #             saliency1array[roi_voxel_idx[0][j], roi_voxel_idx[1][j], roi_voxel_idx[2][j]] = s1
    #             saliency0earlyarray[roi_voxel_idx[0][j], roi_voxel_idx[1][j], roi_voxel_idx[2][j]] = s0e
    #             saliency1earlyarray[roi_voxel_idx[0][j], roi_voxel_idx[1][j], roi_voxel_idx[2][j]] = s1e
    #
    #     saliency0subjects.append(saliency0array)
    #     saliency1subjects.append(saliency1array)
    #     saliency0earlysubjects.append(saliency0earlyarray)
    #     saliency1earlysubjects.append(saliency1earlyarray)
    #
    # plot_nii(saliency0subjects, roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gbp_female_final')
    # plot_nii(saliency1subjects, roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gbp_male_final')
    # plot_nii(saliency0earlysubjects, roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gbp_female_early')
    # plot_nii(saliency1earlysubjects, roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gbp_male_early')
    #
    # plot_nii([subject for idx, subject in enumerate(saliency0subjects) if labels[idx]==0], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gbp_female_femalesubj_final')
    # plot_nii([subject for idx, subject in enumerate(saliency0subjects) if labels[idx]==1], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gbp_female_malesubj_final')
    # plot_nii([subject for idx, subject in enumerate(saliency1subjects) if labels[idx]==0], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gbp_male_femalesubj_final')
    # plot_nii([subject for idx, subject in enumerate(saliency1subjects) if labels[idx]==1], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gbp_male_malesubj_final')
    # plot_nii([subject for idx, subject in enumerate(saliency0earlysubjects) if labels[idx]==0], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gbp_female_femalesubj_early')
    # plot_nii([subject for idx, subject in enumerate(saliency0earlysubjects) if labels[idx]==1], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gbp_female_malesubj_early')
    # plot_nii([subject for idx, subject in enumerate(saliency1earlysubjects) if labels[idx]==0], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gbp_male_femalesubj_early')
    # plot_nii([subject for idx, subject in enumerate(saliency1earlysubjects) if labels[idx]==1], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gbp_male_malesubj_early')

    # # plot guided_gradcam based saliency
    # saliency0 = []
    # saliency1 = []
    # saliency0_early = []
    # saliency1_early = []
    # labels = []
    #
    # saliency0 *= saliency0_gradcam
    # saliency1 *= saliency1_gradcam
    # saliency0early *= saliency0_gradcam
    # saliency1early *= saliency0_gradcam
    #
    # saliency0subjects = []
    # saliency1subjects = []
    # saliency0earlysubjects = []
    # saliency1earlysubjects = []
    #
    # for subjidx, (sal0, sal1, sal0early, sal1early) in enumerate(zip(saliency0, saliency1, saliency0early, saliency1early)):
    #     saliency0array = roiimgarray.copy()
    #     saliency1array = roiimgarray.copy()
    #     saliency0earlyarray = roiimgarray.copy()
    #     saliency1earlyarray = roiimgarray.copy()
    #     print("EXTRACTING GUIDEDBACKPROP-SALIENCY SUBJECT: {} (LABEL:{})".format(subjidx, labels[subjidx]))
    #     for i, (s0, s1, s0e, s1e) in enumerate(zip(sal0, sal1, sal0early, sal1early)):
    #         roi_voxel_idx = np.where(roiimgarray==i+1)
    #         for j in range(roi_voxel_idx[0].shape[0]):
    #             saliency0array[roi_voxel_idx[0][j], roi_voxel_idx[1][j], roi_voxel_idx[2][j]] = s0
    #             saliency1array[roi_voxel_idx[0][j], roi_voxel_idx[1][j], roi_voxel_idx[2][j]] = s1
    #             saliency0earlyarray[roi_voxel_idx[0][j], roi_voxel_idx[1][j], roi_voxel_idx[2][j]] = s0e
    #             saliency1earlyarray[roi_voxel_idx[0][j], roi_voxel_idx[1][j], roi_voxel_idx[2][j]] = s1e
    #
    #     saliency0subjects.append(saliency0array)
    #     saliency1subjects.append(saliency1array)
    #     saliency0earlysubjects.append(saliency0earlyarray)
    #     saliency1earlysubjects.append(saliency1earlyarray)
    #
    # plot_nii(saliency0subjects, roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gbp_female_final')
    # plot_nii(saliency1subjects, roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gbp_male_final')
    # plot_nii(saliency0earlysubjects, roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gbp_female_early')
    # plot_nii(saliency1earlysubjects, roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gbp_male_early')
    #
    # plot_nii([subject for idx, subject in enumerate(saliency0subjects) if labels[idx]==0], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gbp_female_femalesubj_final')
    # plot_nii([subject for idx, subject in enumerate(saliency0subjects) if labels[idx]==1], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gbp_female_malesubj_final')
    # plot_nii([subject for idx, subject in enumerate(saliency1subjects) if labels[idx]==0], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gbp_male_femalesubj_final')
    # plot_nii([subject for idx, subject in enumerate(saliency1subjects) if labels[idx]==1], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gbp_male_malesubj_final')
    # plot_nii([subject for idx, subject in enumerate(saliency0earlysubjects) if labels[idx]==0], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gbp_female_femalesubj_early')
    # plot_nii([subject for idx, subject in enumerate(saliency0earlysubjects) if labels[idx]==1], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gbp_female_malesubj_early')
    # plot_nii([subject for idx, subject in enumerate(saliency1earlysubjects) if labels[idx]==0], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gbp_male_femalesubj_early')
    # plot_nii([subject for idx, subject in enumerate(saliency1earlysubjects) if labels[idx]==1], roiimgaffine, roiimgarray, roimeta, os.path.join(opt.expdir, opt.savedir), 'gbp_male_malesubj_early')



def plot_nii(subject_list, topk, roiimgaffine, roiimgarray, roimeta, savepath, desc):
    saliency_array = np.mean(np.stack(subject_list), axis=0)

    saliency_array_normalized = saliency_array.copy()
    saliency_array_normalized -= saliency_array_normalized.min()
    saliency_array_normalized /= saliency_array_normalized.max()
    saliency_array_normalized_posneg = saliency_array.copy()
    saliency_array_normalized_posneg /= np.abs(saliency_array_normalized_posneg).max()
    saliency_array_positive = (np.maximum(0, saliency_array) / saliency_array.max())
    saliency_array_negative = (np.maximum(0, -saliency_array) / -saliency_array.min())

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

        del saliency_array_normalized_topk
        del saliency_img_normalized_topk
        del network_dicts
        del network_lh_img
        del network_rh_img

    saliency_img = nib.Nifti1Image(saliency_array, roiimgaffine)
    saliency_img_normalized = nib.Nifti1Image(saliency_array_normalized, roiimgaffine)
    saliency_img_normalized_posneg = nib.Nifti1Image(saliency_array_normalized_posneg, roiimgaffine)
    saliency_img_positive = nib.Nifti1Image(saliency_array_positive, roiimgaffine)
    saliency_img_negative = nib.Nifti1Image(saliency_array_negative, roiimgaffine)
    nib.save(saliency_img, os.path.join(savepath, 'saliency_{}.nii'.format(desc)))
    nib.save(saliency_img_normalized, os.path.join(savepath, 'saliency_normalized_{}.nii'.format(desc)))
    nib.save(saliency_img_normalized_posneg, os.path.join(savepath, 'saliency_normalized_posneg_{}.nii'.format(desc)))
    nib.save(saliency_img_positive, os.path.join(savepath, 'saliency_positive_{}.nii'.format(desc)))
    nib.save(saliency_img_negative, os.path.join(savepath, 'saliency_negative_{}.nii'.format(desc)))

    write_csv(saliency_array_normalized, roiimgarray, roimeta, savepath, 'normalized_{}'.format(desc))
    write_csv(saliency_array_normalized_posneg, roiimgarray, roimeta, savepath, 'normalized_posneg_{}'.format(desc))
    write_csv(saliency_array_positive, roiimgarray, roimeta, savepath, 'positive_{}'.format(desc))
    write_csv(saliency_array_negative, roiimgarray, roimeta, savepath, 'negative_{}'.format(desc))

    del saliency_array
    del saliency_array_normalized
    del saliency_array_normalized_posneg
    del saliency_array_positive
    del saliency_array_negative
    del saliency_img
    del saliency_img_normalized
    del saliency_img_normalized_posneg
    del saliency_img_positive
    del saliency_img_negative


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
