# mricrogl>=1.2 python script
# run example:
# >> MRIcroGL evaluate/visualize_saliency.py

import os
import gl
import argparse

min = 0.75
max = 1.0
colorname = '4hot'

def main():
    EXP_DIR='results/graph_neural_mapping'
    NII_DIR='saliency_nii'
    SAVE_DIR='saliency_img'

    os.makedirs(os.path.join(EXP_DIR, SAVE_DIR), exist_ok=True)

    gl.resetdefaults()
    gl.loadimage('mni152')
    # gl.shadername('Shell')
    gl.overlayloadsmooth(True)
    gl.opacity(0, 50)
    gl.colorbarposition(0)

    visualize_axial(os.path.join(EXP_DIR, NII_DIR), os.path.join(EXP_DIR, SAVE_DIR))
    visualize_sagittal(os.path.join(EXP_DIR, NII_DIR), os.path.join(EXP_DIR, SAVE_DIR))
    visualize_colorbar(os.path.join(EXP_DIR, SAVE_DIR))


def visualize_axial(niidir, savedir, min=0.75, max=1.0):
    gl.viewaxial(1)
    for method in ['grad', 'cam']:
        for network in ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']:
            for gender in ['female', 'male']:
                gl.overlayload(os.path.join(niidir, 'network', 'saliency_{}_{}_top20_{}_lh'.format(method, gender, network)))
                gl.overlayload(os.path.join(niidir, 'network', 'saliency_{}_{}_top20_{}_rh'.format(method, gender, network)))
                gl.minmax(1, min, max)
                gl.minmax(2, min, max)
                gl.colorname(1, colorname)
                gl.colorname(2, colorname)

                gl.savebmp(os.path.join(savedir, '{}_{}_{}_axial.png'.format(method, network, gender)))
                gl.overlaycloseall()

def visualize_sagittal(niidir, savedir, min=0.75, max=1.0):
    for hemisphere in ['lh', 'rh']:
        if hemisphere =='lh': gl.clipazimuthelevation(0.49, 90, 0)
        elif hemisphere =='rh': gl.clipazimuthelevation(0.49, 270, 0)
        for network in ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']:
            for gender in ['female', 'male']:
                gl.overlayload(os.path.join(niidir, 'network', 'saliency_{}_top20_{}_{}'.format(method, gender, network, hemisphere)))
                gl.minmax(1, min, max)
                gl.colorname(1, colorname)

                gl.viewsagittal(1)
                gl.savebmp(os.path.join(savedir, '{}_{}_sagittal_{}_lt.png'.format(network, gender, hemisphere)))
                gl.viewsagittal(0)
                gl.savebmp(os.path.join(savedir, '{}_{}_sagittal_{}_rt.png'.format(network, gender, hemisphere)))
                gl.overlaycloseall()

def visualize_colorbar(savedir, min=0.75, max=1.0):
    gl.resetdefaults()
    gl.minmax(0, min, max)
    gl.colorname(0, colorname)
    gl.opacity(0, 0)
    gl.colorbarposition(1)
    gl.savebmp(os.path.join(savedir, 'colorbar.png'))


if __name__=='__main__':
    main()
