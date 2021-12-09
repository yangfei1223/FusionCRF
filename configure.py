# -*- coding:utf-8 -*-
'''
@Time : 2021/12/3 下午9:37
@Author: yangfei
@File : configure.py
'''
import torch.nn as nn
from torch_geometric.transforms import Compose, NormalizeScale, FixedPoints, RandomRotate
from torch_points3d.core.data_transform import *
from torchvision import transforms as T
import torchvision.transforms.functional as F
import dataset, model

def get_class_weights(dataset):
    # pre-calculate the number of points in each category
    num_per_class = []
    if dataset is 'S3DIS':
        num_per_class = np.array([3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                                  650464, 791496, 88727, 1284130, 229758, 2272837], dtype=np.int32)
    elif dataset is 'Semantic3D':
        num_per_class = np.array([5181602, 5012952, 6830086, 1311528, 10476365, 946982, 334860, 269353],
                                 dtype=np.int32)
    elif dataset is 'NPM3D':
        num_per_class = np.array([65075320, 33014819, 656096, 61715, 296523, 4052947, 172132, 4212295, 10599237],
                                 dtype=np.int32)
    elif dataset is 'SemanticKITTI':
        num_per_class = np.array([55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858,
                                  240942562, 17294618, 170599734, 6369672, 230413074, 101130274, 476491114,
                                  9833174, 129609852, 4506626, 1168181], dtype=np.int32)
    elif dataset is 'KITTIRoad2D':
        num_per_class = np.array([75751737, 23702601], dtype=np.int32)
    elif dataset is 'KITTIRoad3D':
        num_per_class = np.array([3512311, 1791793], dtype=np.int32)
    else:
        raise ValueError('Unsupported dataset!')
    weight = num_per_class / float(sum(num_per_class))
    ce_label_weight = 1 / (weight + 0.02)
    return torch.from_numpy(ce_label_weight.astype(np.float32))


class Config():
    def __init__(self,
                 device_id=0,
                 mode='train',
                 batch_size=8,
                 epochs=100):
        self.device = 'cuda'
        self.device_id = device_id
        self.mode = mode
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = 1e-2
        self.gamma = 0.1**0.02
        self.momentum = 0.95
        self.weight_decay = 1e-4
        self.num_classes = 2
        self.ignore_index = 255
        self.metric = 'Mean IoU'
        self.root = '/media/yangfei/Repository/KITTI/data_road'


class KITTIRoad2DConfig(Config):
    def __init__(self,
                 device_id=0,
                 mode='train',
                 batch_size=8,
                 epochs=100):
        super(KITTIRoad2DConfig, self).__init__(device_id, mode, batch_size, epochs)
        self.model_name = 'FCN8s_pretrained'
        self.dataset_name = 'KITTIRoad2DDataset'
        self.class_weights = get_class_weights('KITTIRoad2D')
        self.precompute_multiscale = False
        self.conv_type = 'dense'
        self.prefix = '{}_on_{}'.format(self.model_name, self.dataset_name)
        self.model_path = 'RUNS/checkpoints/{}.ckpt'.format(self.prefix)
        self.save_path = 'RUNS/results/test/{}/sparse'.format(self.dataset_name)

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=np.array([0.33053341, 0.34080286, 0.32288151]), std=np.array([0.27192578, 0.26952331, 0.27069592]))
        ])
        self.transform2 = None

        self.dataset_fn = partial(getattr(dataset, self.dataset_name),
                                  root=self.root,
                                  train_transform=self.transform,
                                  test_transform=self.transform)

        self.model_fn = partial(getattr(model, self.model_name),
                                in_channels=3,
                                n_classes=self.num_classes,
                                class_weights=self.class_weights,
                                ignore_index=self.ignore_index)


class KITTIRoad3DConfig(Config):
    def __init__(self,
                 device_id=0,
                 mode='train',
                 batch_size=8,
                 epochs=100):
        super(KITTIRoad3DConfig, self).__init__(device_id, mode, batch_size, epochs)
        self.model_name = 'PointNet2'
        self.dataset_name = 'KITTIRoad3DDataset'
        self.class_weights = get_class_weights('KITTIRoad3D')
        self.down_mlp = [[32, 32, 64], [64, 64, 128], [128, 128, 256], [256, 256, 512]]
        self.up_mlp = [[256, 256], [256, 256], [256, 128], [128, 128, 128]]
        self.kernel_size = [32, 32, 32, 32, 32]
        self.dilation_rate = [1, 1, 1, 1, 1]
        self.grid_size = [0.08, 0.16, 0.32, 0.64, 1.28]
        self.sample_ratio = [0.25, 0.25, 0.25, 0.25, 0.5]
        self.search_method = 'knn'
        self.sample_method = 'fps'
        self.precompute_multiscale = True
        self.conv_type = 'sparse'
        self.prefix = '{}_on_{}'.format(self.model_name, self.dataset_name)
        self.model_path = 'RUNS/checkpoints/{}.ckpt'.format(self.prefix)
        self.save_path = 'RUNS/results/test/{}'.format(self.dataset_name)

        self.train_transform = Compose([
            RandomScaleAnisotropic(scales=[0.8, 1.2]),
            RandomSymmetry(axis=[False, True, False]),      # random symmetric y-axis
            # RandomRotate(degrees=180, axis=-1),
            RandomNoise(sigma=0.001),
            DropFeature(drop_proba=0.2, feature_name='rgb'),
            AddFeatsByKeys(list_add_to_x=[True, True, False, True, False],
                           feat_names=['pos', 'rgb', 'intensity', 'norm', 'curv'],
                           delete_feats=[False, True, True, True, True])
        ])
        self.test_transform = Compose([
            AddFeatsByKeys(list_add_to_x=[True, True, False, True, False],
                           feat_names=['pos', 'rgb', 'intensity', 'norm', 'curv'],
                           delete_feats=[False, True, True, True, True])
        ])
        self.dataset_fn = partial(getattr(dataset, self.dataset_name),
                                  root=self.root,
                                  kernel_size=self.kernel_size,
                                  dilation_rate=self.dilation_rate,
                                  grid_size=self.grid_size,
                                  sample_ratio=self.sample_ratio,
                                  search_method=self.search_method,
                                  sample_method=self.sample_method,
                                  train_transform=self.train_transform,
                                  test_transform=self.test_transform)

        self.model_fn = partial(getattr(model, self.model_name),
                                in_channels=9,
                                down_mlp=self.down_mlp,
                                up_mlp=self.up_mlp,
                                n_classes=self.num_classes,
                                class_weights=self.class_weights,
                                ignore_index=self.ignore_index)
