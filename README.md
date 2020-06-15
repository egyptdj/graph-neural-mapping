# Graph Neural Mapping

## Notice
PyTorch implementation of the paper [Understanding Graph Isomorphism Network for rs-fMRI Functional Connectivity Analysis](https://www.frontiersin.org/articles/10.3389/fnins.2020.00630) by Byung-Hoon Kim and Jong Chul Ye.

The model code is based on the official implementation of the
- Graph Isomorphism Network: [paper](https://arxiv.org/abs/1810.00826), [github](https://github.com/weihua916/powerful-gnns)
- Deep Graph Infomax: [paper](https://arxiv.org/abs/1809.10341), [github](https://github.com/PetarV-/DGI)

## Resources
Dataset:
- The Human Connectome Project: [paper](https://www.sciencedirect.com/science/article/pii/S1053811913005351), [web](https://www.humanconnectome.org/)

Processing:
- FSL: [paper](https://www.sciencedirect.com/science/article/pii/S1053811911010603), [web](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki)
- GRETNA: [paper](https://www.frontiersin.org/articles/10.3389/fnhum.2015.00386/full), [web](https://www.nitrc.org/projects/gretna/)

Atlas:
- Schaefer et al.: [paper](https://academic.oup.com/cercor/article/28/9/3095/3978804), [github](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal)

Visualization:
- MRIcroGL: [github](https://github.com/rordenlab/MRIcroGL12), [web](https://www.nitrc.org/plugins/mwiki/index.php/mricrogl:MainPage)

## Requirements
Python3 with following packages
- `pytorch >= 1.4.0`
- `scikit-learn >= 0.21.3`
- `nilearn >= 0.5.2`
- `nibabel >= 2.5.0`
- `tqdm`
