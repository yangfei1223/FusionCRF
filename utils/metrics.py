# -*- coding:utf-8 -*-
'''
@Time : 2020/9/13 下午4:26
@Author: yangfei
@File : metrics.py
'''

# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
import torch
import torch.nn.functional as F


def iou_from_confusions(confusions):
    TP = np.diagonal(confusions, axis1=-2, axis2=-1)
    TPFN = np.sum(confusions, axis=-1)
    TPFP = np.sum(confusions, axis=-2)

    IoU = TP / (TPFP + TPFN - TP + 1e-6)

    mask = TPFN < 1e-3
    counts = np.sum(1 - mask, axis=-1, keepdims=True)
    mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

    IoU += mask * mIoU

    return IoU


class runningScore(object):
    def __init__(self, n_classes, ignore_index=-1):
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class) & (label_true != self.ignore_index)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        if label_trues.ndim == 1:
            self.confusion_matrix += self._fast_hist(label_trues, label_preds, self.n_classes)
        else:
            for lt, lp in zip(label_trues, label_preds):
                self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                'Overall Acc': acc,
                'Mean Acc': acc_cls,
                'FreqW Acc': fwavacc,
                'Mean IoU': mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class runningScoreShapeNet(object):
    def __init__(self):
        self.obj_classes = {
            'Airplane': 0,
            'Bag': 1,
            'Cap': 2,
            'Car': 3,
            'Chair': 4,
            'Earphone': 5,
            'Guitar': 6,
            'Knife': 7,
            'Lamp': 8,
            'Laptop': 9,
            'Motorbike': 10,
            'Mug': 11,
            'Pistol': 12,
            'Rocket': 13,
            'Skateboard': 14,
            'Table': 15,
        }

        self.seg_classes = {
            "Airplane": [0, 1, 2, 3],
            "Bag": [4, 5],
            "Cap": [6, 7],
            "Car": [8, 9, 10, 11],
            "Chair": [12, 13, 14, 15],
            "Earphone": [16, 17, 18],
            "Guitar": [19, 20, 21],
            "Knife": [22, 23],
            "Lamp": [24, 25, 26, 27],
            "Laptop": [28, 29],
            "Motorbike": [30, 31, 32, 33, 34, 35],
            "Mug": [36, 37],
            "Pistol": [38, 39, 40],
            "Rocket": [41, 42, 43],
            "Skateboard": [44, 45, 46],
            "Table": [47, 48, 49],
        }
        self.category_IoU = np.zeros(16)
        self.category_num = np.zeros(16)

    def update(self, label_trues, label_preds, category):
        name = [k for (k, v) in self.obj_classes.items() if v == category][0]
        label = self.seg_classes[name]
        iu_part = 0.
        for l in label:
            locations_trues = (label_trues == l)
            locations_preds = (label_preds == l)
            i_locations = np.logical_and(locations_trues, locations_preds)
            u_locations = np.logical_or(locations_trues, locations_preds)
            i = np.sum(i_locations) + np.finfo(np.float32).eps
            u = np.sum(u_locations) + np.finfo(np.float32).eps
            iu_part += i / u
        iu = iu_part / (len(label))
        self.category_IoU[category] += iu
        self.category_num[category] += 1
        return iu

    def get_scores(self):
        pIoU = self.category_IoU.sum() / self.category_num.sum()
        per_class_pIoU = self.category_IoU / self.category_num
        mpIoU = per_class_pIoU.mean()
        cls_pIoU = {}
        for (k, v) in self.obj_classes.items():
            cls_pIoU[k] = per_class_pIoU[v]

        return pIoU, mpIoU, cls_pIoU

    def reset(self):
        self.category_IoU = np.zeros(16)
        self.category_num = np.zeros(16)


class runingScoreNormal(object):
    def __init__(self):
        self.total_err = 0
        self.counts = 0

    def update(self, label_trues, label_preds, batch_idx):
        cos_dist = 1. - F.cosine_similarity(label_trues, label_preds, dim=-1).abs()
        batches = torch.unique(batch_idx)
        for batch in batches:
            mask = batch_idx == batch
            err = cos_dist[mask].mean()
            self.total_err += err
        self.counts += len(batches)

    def get_scores(self):
        return self.total_err / self.counts

    def reset(self):
        self.total_err = 0
        self.counts = 0


if __name__ == '__main__':
    score = runningScoreShapeNet()
    print(score)
