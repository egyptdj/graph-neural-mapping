# Graph Neural Mapping

PyTorch implementation of the paper Understanding Graph Isomorphism Network for rs-fMRI Functional Connectivity Analysis by Byung-Hoon Kim and Jong Chul Ye.

## Notice
The code is based on the official implementation of the
- Graph Isomorphism Network: [paper](https://arxiv.org/abs/1810.00826), [github](https://github.com/weihua916/powerful-gnns)
- Deep Graph Infomax: [paper](https://arxiv.org/abs/1809.10341), [github](https://github.com/PetarV-/DGI)

The data used for the experiments are available on the google-drive. [Download data](https://drive.google.com/file/d/1WDFUyyd6jA56r_9_jCajj-7mhHBZSJC-/view?usp=sharing)\
Download the data and unzip to create the 'data/' directory in the repository path.\
Processing of the data are based on the
- The Human Connectome Project: [paper](https://www.sciencedirect.com/science/article/pii/S1053811913005351), [web](https://www.humanconnectome.org/)
- FSL: [paper](https://www.sciencedirect.com/science/article/pii/S1053811911010603), [web](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki)
- GRETNA: [paper](https://www.frontiersin.org/articles/10.3389/fnhum.2015.00386/full), [web](https://www.nitrc.org/projects/gretna/)

*Redistribution of the provided data is not allowed.*

## Requirements
Python3 with following packages
- `pytorch >= 1.1.0`
- `scikit-learn >= 0.21.3`
- `nibabel >= 2.5.0`
