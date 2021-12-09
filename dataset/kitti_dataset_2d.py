# -*- coding:utf-8 -*-
'''
@Time : 2021/11/25 下午8:34
@Author: yangfei
@File : kitti_dataset_2d.py
'''

import os
from os.path import join, dirname, abspath
import cv2
from torchvision import transforms as T
from dataset import *

BASE_DIR = dirname(abspath(__file__))

class KITTIRoad2D():
    def __init__(self,
                 root='/media/yangfei/Repository/KITTI/data_road',
                 split='train',
                 transform=T.Compose([
                     T.ToTensor(),
                     T.Normalize(mean=np.array([0.33053341, 0.34080286, 0.32288151]), std=np.array([0.27192578, 0.26952331, 0.27069592])),
                 ]),
                 transform2=None,
                 ):
        assert split in ['train', 'val', 'trainval', 'test']
        self.root = root
        self.split = split
        self.transform = transform
        self.transform2 = transform2
        self.filelist = list(np.loadtxt(join(BASE_DIR, '{}.txt'.format(split)), dtype=np.str))

    # load rgb
    def _load_image(self, filename):
        im = cv2.imread(filename)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im[-288:, :1216, :]
    # load label
    def _load_label(self, filename):
        lb = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        return lb[-288:, :1216]

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        if self.split != 'test':
            filename = os.path.join(self.root, 'training/image_2', self.filelist[idx] + '.png')
            im = self._load_image(filename)
            if self.transform:
                im = self.transform(im)
            filename = os.path.join(self.root, 'training/gt_2', self.filelist[idx] + '.png')
            lb = self._load_label(filename)
            lb = torch.tensor(lb, dtype=torch.long)
            if self.transform2:
                im, lb = self.transform2(im, lb)
            data = Data(x=im, y=lb)
            return data
        else:
            filename = os.path.join(self.root, 'testing/image_2', self.filelist[idx] + '.png')
            im = self._load_image(filename)
            if self.transform:
                im = self.transform(im)
            if self.transform2:
                im = self.transform2(im)
            data = Data(x=im, y=None)
            return data


class KITTIRoad2DDataset(BaseDataset):
    def __init__(self,
                 root,
                 train_transform=None,
                 test_transform=None
                 ):
        super(KITTIRoad2DDataset, self).__init__(kernel_size=[32, 32, 32, 32, 32],      # shadow parameter, no used
                                                 dilation_rate=None,
                                                 grid_size=[0.08, 0.16, 0.32, 0.64, 1.28],
                                                 sample_ratio=[0.25, 0.25, 0.25, 0.25, 0.5])

        self.train_set = KITTIRoad2D(root, split='train', transform=train_transform)
        self.val_set = KITTIRoad2D(root, split='val', transform=test_transform)
        self.test_set = KITTIRoad2D(root, split='test', transform=test_transform)

if __name__ == '__main__':
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=np.array([0.33053341, 0.34080286, 0.32288151]), std=np.array([0.27192578, 0.26952331, 0.27069592]))
    ])
    dataset = KITTIRoad2DDataset(root='/media/yangfei/Repository/KITTI/data_road',
                                 train_transform=transform,
                                 test_transform=transform)
    dataset.create_dataloader(batch_size=4, precompute_multiscale=False, conv_type='dense')
    for data in dataset.train_loader:
        print(data.x.shape)
        print(data.y.shape)

    print(111)


