# FusionCRF
This repository is the implementation of "A Fusion Model for Road Detection based on Deep Learning and Fully Connected CRF".

### 1. Statement
Recently, the author had found some errors and confusing results in the experimental part of the original paper, which had bothered me for while. To make up for these mistakes, the author thus decided to re-implement the algorithm thoroughly and open the source code. Therefore, I provide this repository here, and the paper is also revised with the new experimental results. If the readers are interested in this work, please also find the revised paper in this repository.


### 2. Setup

#### 1) Building
```bash
cd utils
sh compile_op.sh
```

#### 2) Dependency
This repository is partially dependent on [PyTorch](https://pytorch.org/get-started/locally/) for model building, on [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#quick-start) and [Torch Points 3D](https://torch-points3d.readthedocs.io/en/latest/) in preparing datasets, and [pydensecrf](https://github.com/lucasb-eyer/pydensecrf) for fusing.

### 3. Running
#### 1) Dataset
Please see the 'dataset' dir for data preparing. Note that, you should use our [KITTI tookit](https://github.com/yangfei1223/KITTI) to prepare the point cloud data.

#### 2) Height upsampling
To construct height potential for the CRF model, you can use the [MRFUpsampling](https://github.com/yangfei1223/MRFUpsampling), which we also provide in our repository, to upsample the sparse height image to a dense one.

#### 3) Evaluation
Please see 'trainval.py' for training the FCN8s model on images and the PointNet++ model on point clouds. (Following our original paper setting, we use plain versions of deep models as same as their papers descript.)

Please see 'configure.py' for experimental settings.

Please see 'fusion.py' for fusing the image prediction and point cloud prediction.

By the way, you can also use the 'utils/statistic.py', which use the same metrics as KITTI, to evaluate the results, and use 'utils/seg_utils' for visualizations.

### 4. Acknowledgement
In processing the point cloud data, part of the codes refers to the [KPConv](https://github.com/HuguesTHOMAS/KPConv) and [RandLA-Net](https://github.com/QingyongHu/RandLA-Net).













