# -*- coding:utf-8 -*-
'''
@Time : 2021/12/2 下午8:36
@Author: yangfei
@File : kitti_dataset_3d.py
'''

import os
from os.path import join, dirname, abspath
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torchvision import transforms as T
from dataset import *
from utils import Plot

BASE_DIR = dirname(abspath(__file__))

from torchvision.datasets import VisionDataset

class KITTIRoad3D():
    def __init__(self,
                 root='/media/yangfei/Repository/KITTI/data_road',
                 split='train',
                 transform=None):

        assert split in ['train', 'val', 'trainval', 'test']
        self.root = root
        self.split = split
        self.transform = transform

        self.filelist = list(np.loadtxt(join(BASE_DIR, '{}.txt'.format(split)), dtype=np.str))

    def _load_pc_kitti(self, filename):
        # cloud = np.loadtxt(filename)
        cloud = pd.read_csv(filename, header=None, delim_whitespace=True).values    # pandas for fast loading
        return cloud

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        if self.split != 'test':
            filename = os.path.join(self.root, 'training/velodyne_image', self.filelist[idx] + '.txt')
            cloud = self._load_pc_kitti(filename)
            uv = torch.from_numpy(cloud[:, :2]).to(torch.long)
            pos = torch.from_numpy(cloud[:, 2:5]).to(torch.float)
            intensity = torch.from_numpy(cloud[:, 5]).to(torch.float)
            normal = torch.from_numpy(cloud[:, 6:9]).to(torch.float)
            curv = torch.from_numpy(cloud[:, 9]).to(torch.float)
            rgb = torch.from_numpy(cloud[:, 10:13]).to(torch.float)
            label = torch.from_numpy(cloud[:, -1]).to(torch.long)
            data = Data(pos=pos, uv=uv, intensity=intensity, norm=normal, curv=curv, rgb=rgb, y=label)
        else:
            filename = os.path.join(self.root, 'testing/velodyne_image', self.filelist[idx] + '.txt')
            cloud = self._load_pc_kitti(filename)
            uv = torch.from_numpy(cloud[:, :2]).to(torch.long)
            pos = torch.from_numpy(cloud[:, 2:5]).to(torch.float)
            intensity = torch.from_numpy(cloud[:, 5]).to(torch.float)
            normal = torch.from_numpy(cloud[:, 6:9]).to(torch.float)
            curv = torch.from_numpy(cloud[:, 9]).to(torch.float)
            rgb = torch.from_numpy(cloud[:, 10:13]).to(torch.float)
            data = Data(pos=pos, uv=uv, intensity=intensity, norm=normal, curv=curv, rgb=rgb, y=None)
        data = data if self.transform is None else self.transform(data)
        return data

class KITTIRoad3DDataset(BaseDataset):
    def __init__(self,
                 root,
                 kernel_size=None,
                 dilation_rate=None,
                 grid_size=None,
                 sample_ratio=None,
                 search_method=None,
                 sample_method=None,
                 train_transform=None,
                 test_transform=None):
        super(KITTIRoad3DDataset, self).__init__(kernel_size,
                                                 dilation_rate,
                                                 grid_size,
                                                 sample_ratio,
                                                 search_method,
                                                 sample_method)

        self.train_set = KITTIRoad3D(root, split='train', transform=train_transform)
        self.val_set = KITTIRoad3D(root, split='val', transform=test_transform)
        self.test_set = KITTIRoad3D(root, split='test', transform=test_transform)



if __name__ == '__main__':
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=np.array([0.33053341, 0.34080286, 0.32288151]), std=np.array([0.27192578, 0.26952331, 0.27069592]))
    ])
    # dataset = KITTIRoad2D(transforms=transforms)
    # # im, lb = dataset[0]
    # dataset = KITTIRoad3D()
    # data = dataset[0]

    dataset = KITTIRoad3DDataset(root='/media/yangfei/Repository/KITTI/data_road',
                                 kernel_size=[32, 32, 32, 32, 32],
                                 grid_size=[0.08, 0.16, 0.32, 0.64, 1.28],
                                 sample_ratio=[0.25, 0.25, 0.25, 0.25, 0.5],
                                 search_method='knn',
                                 sample_method='fps')

    dataset.create_dataloader(batch_size=8, precompute_multiscale=True, conv_type='sparse')

    for data in dataset.train_loader:
        print(data)
        batch = 0
        Plot.draw_pc(data.multiscale[0].pos[data.multiscale[0].batch == batch].numpy())
        Plot.draw_pc(data.multiscale[1].pos[data.multiscale[1].batch == batch].numpy())
        Plot.draw_pc(data.multiscale[2].pos[data.multiscale[2].batch == batch].numpy())
        Plot.draw_pc(data.multiscale[3].pos[data.multiscale[3].batch == batch].numpy())
        Plot.draw_pc(data.multiscale[4].pos[data.multiscale[4].batch == batch].numpy())
    print(111)


