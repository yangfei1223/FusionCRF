# -*- coding:utf-8 -*-
'''
@Time : 2021/12/1 下午8:40
@Author: yangfei
@File : trainval.py
'''
import os, sys, time, pickle, argparse
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq
from utils import runningScore
import configure

logging.getLogger().setLevel(logging.INFO)


class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        torch.cuda.set_device(cfg.device_id)

        self.dataset = cfg.dataset_fn()
        self.dataset.create_dataloader(batch_size=cfg.batch_size, precompute_multiscale=cfg.precompute_multiscale, conv_type=cfg.conv_type)
        self.model = cfg.model_fn()

        if 'FCN' in cfg.model_name:
            vgg_params = list(map(id, self.model.features.parameters()))
            base_params = filter(lambda p: id(p) not in vgg_params, self.model.parameters())
            self.optimizer = torch.optim.SGD([{'params': base_params},
                                              {'params': self.model.features.parameters(), 'lr': self.cfg.lr / 100}],
                                             lr=cfg.lr,
                                             momentum=cfg.momentum,
                                             weight_decay=cfg.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                             lr=cfg.lr,
                                             momentum=cfg.momentum,
                                             weight_decay=cfg.weight_decay)



        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=cfg.gamma)

        self.metrics = runningScore(cfg.num_classes, ignore_index=cfg.ignore_index)


    def train_one_epoch(self, epoch):
        self.model.train()
        self.metrics.reset()
        with Ctq(self.dataset.train_loader) as tq_loader:
            for i, data in enumerate(tq_loader):
                tq_loader.set_description('Train epoch[{}/{}]'.format(epoch, self.cfg.epochs))
                self.optimizer.zero_grad()
                self.model.forward(data.to(self.device))
                self.model.compute_loss()
                self.model.backward()
                self.optimizer.step()
                tq_loader.set_postfix(loss=self.model.get_loss())
                self.metrics.update(self.model.get_label(), self.model.get_pred())
        score_dict, _ = self.metrics.get_scores()
        logging.info('Training OA: {:.2f} %, mIoU: {:.2f} %'.format(score_dict['Overall Acc'] * 100, score_dict['Mean IoU'] * 100))

    def val_one_epoch(self, epoch):
        self.model.eval()
        self.metrics.reset()
        with Ctq(self.dataset.val_loader) as tq_loader:
            for i, data in enumerate(tq_loader):
                tq_loader.set_description('Val epoch[{}/{}]'.format(epoch, self.cfg.epochs))
                with torch.no_grad():
                    self.model.forward(data.to(self.device))
                    self.model.compute_loss()
                tq_loader.set_postfix(loss=self.model.get_loss())
                self.metrics.update(self.model.get_label(), self.model.get_pred())
        score_dict, _ = self.metrics.get_scores()
        logging.info('Test OA: {:.2f} %, mIoU: {:.2f} %'.format(score_dict['Overall Acc'] * 100, score_dict['Mean IoU'] * 100))


    def trainval(self):
        best_metric = 0
        self.model.to(self.device)
        for epoch in range(self.cfg.epochs):
            logging.info('Training epoch: {}, learning rate: {}'.format(epoch, self.scheduler.get_last_lr()[0]))
            # training
            self.train_one_epoch(epoch)
            # validation
            self.val_one_epoch(epoch)
            # save model
            score_dict, _ = self.metrics.get_scores()
            current_metric = score_dict[self.cfg.metric]
            if best_metric <= current_metric:
                best_metric = current_metric
                self.model.save(self.cfg.model_path)     # save model
                logging.info('Save {} succeed, best {}: {:.2f} % !'.format(self.cfg.model_path, self.cfg.metric, best_metric * 100))

            self.scheduler.step()
        logging.info('Training finished, best {}: {:.2f} %'.format(self.cfg.metric, best_metric * 100))

    def test_2d(self):
        logging.info('Test {} on {} ...'.format(self.cfg.model_name, self.cfg.dataset_name))
        os.makedirs(self.cfg.save_path) if not os.path.exists(self.cfg.save_path) else None

        self.model.load(self.cfg.model_path)
        self.model.to(self.device)
        self.model.eval()
        filelist = list(np.loadtxt('dataset/test.txt', dtype=np.str))
        with Ctq(self.dataset.test_loader) as tq_loader:
            for i, data in enumerate(tq_loader):
                tq_loader.set_description('Testing')
                # load src image
                filename = os.path.join(self.cfg.root, 'testing/image_2', filelist[i] + '.png')
                src_im = cv2.imread(filename)
                height, width = src_im.shape[0], src_im.shape[1]
                with torch.no_grad():
                    self.model.forward(data.to(self.device))
                prob = np.zeros((height, width), dtype=np.uint8)
                prob[-288:, :1216] = self.model.get_prob()*255
                filename = os.path.join(self.cfg.save_path, filelist[i] + '.png')
                cv2.imwrite(filename, prob)

    def test_3d(self):
        logging.info('Test {} on {} ...'.format(self.cfg.model_name, self.cfg.dataset_name))
        os.makedirs(self.cfg.save_path) if not os.path.exists(self.cfg.save_path) else None

        self.model.load(self.cfg.model_path)
        self.model.to(self.device)
        self.model.eval()
        filelist = list(np.loadtxt('dataset/test.txt', dtype=np.str))
        with Ctq(self.dataset.test_loader) as tq_loader:
            for i, data in enumerate(tq_loader):
                tq_loader.set_description('Testing')
                # load src image
                filename = os.path.join(self.cfg.root, 'testing/image_2', filelist[i] + '.png')
                src_im = cv2.imread(filename)
                height, width = src_im.shape[0], src_im.shape[1]
                with torch.no_grad():
                    self.model.forward(data.to(self.device))
                u = data.uv.cpu().numpy()[:, 0]
                v = data.uv.cpu().numpy()[:, 1]
                prob = np.zeros((height, width), dtype=np.uint8)
                prob[u, v] = self.model.get_prob() * 255
                filename = os.path.join(self.cfg.save_path, filelist[i] + '.png')
                cv2.imwrite(filename, prob)

    def test(self):
        if '2D' in self.cfg.dataset_name:
            self.test_2d()
        elif '3D' in self.cfg.dataset_name:
            self.test_3d()

    def __call__(self, *args, **kwargs):
        self.trainval() if self.cfg.mode.lower() == 'train' else self.test()

    def __repr__(self):
        return 'Trainer {} on {}, mode={}, batch_size={}'.format(self.cfg.model_name,
                                                                 self.cfg.dataset_name,
                                                                 self.cfg.mode,
                                                                 self.cfg.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='KITTIRoad3D', help='choose dataset')
    parser.add_argument('--device_id', type=int, default=0, help='choice device [0 or 1]')
    parser.add_argument('--mode', type=str, default='test', help='train or test')
    parser.add_argument('--batch_size', type=int, default=1, help='set batch size')
    parser.add_argument('--epochs', type=int, default=100, help='set epochs')
    FLAGS = parser.parse_args()

    cfg = getattr(configure, FLAGS.dataset + 'Config')(device_id=FLAGS.device_id,
                                                       mode=FLAGS.mode,
                                                       batch_size=FLAGS.batch_size,
                                                       epochs=FLAGS.epochs)

    trainer = Trainer(cfg)
    logging.info(trainer)
    logging.info(trainer.model)
    trainer()