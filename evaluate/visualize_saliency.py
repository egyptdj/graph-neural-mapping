# mricrogl script
import gl
gl.resetdefaults()
gl.loadimage('mni152')
gl.overlayload('spmMotor')
# gl.overlayloadcluster('/media/bispl/dbx/Dropbox/Academic/gin_sparsity30_lr0005/saliency/gradcam_saliency_female_early.nii', 0.5, 0.0, False)
gl.overlayloadcluster('spmMotor', 10.5, 0.0, False)
# gl.overlayloadsmooth(True)
gl.overlayminmax(1, 0.3, 1.0)
gl.overlaycolorname('4hot')
gl.colorbarposition(4)
gl.overlayhidezeros(True)
