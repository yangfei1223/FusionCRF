# -*- coding:utf-8 -*-
'''
@Time : 2021/12/7 下午9:50
@Author: yangfei
@File : fusion.py
'''
import os, sys, time
import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian

##################################
### Read images and annotation ###
##################################
rgb_path = '/media/yangfei/Repository/KITTI/data_road/testing/image_2/'
height_path = '/media/yangfei/Repository/KITTI/data_road/testing/dense/z/'
img_prob_path = '/home/yangfei/myPaper/FusionCRF/RUNS/results/test/KITTIRoad2DDataset'
lidar_prob_path = '/home/yangfei/myPaper/FusionCRF/RUNS/results/test/KITTIRoad3DDataset/dense'
output_path = '/home/yangfei/myPaper/FusionCRF/RUNS/results/test/DenseCRFFusion'
os.mkdir(output_path) if not os.path.exists(output_path) else None

filelist=os.listdir(img_prob_path)
filelist.sort()
for name in filelist:
    print(name)
    rgb = cv2.imread(os.path.join(rgb_path, name))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)      # rgb image
    height = cv2.imread(os.path.join(height_path, name), cv2.IMREAD_UNCHANGED)    # height image
    img_pred = cv2.imread(os.path.join(img_prob_path, name), cv2.IMREAD_UNCHANGED)      # img pred
    lidar_pred = cv2.imread(os.path.join(lidar_prob_path, name), cv2.IMREAD_UNCHANGED)  # lidar pred

    H, W, NLABELS = rgb.shape[0], rgb.shape[1], 2

    probs = (img_pred / 255.) * 0.5 + (lidar_pred / 255.) * 0.5
    probs = np.tile(probs[np.newaxis, ...], (2, 1, 1))
    probs[0, ...] = 1 - probs[0, ...]

    ###########################
    ### Setup the CRF model ###
    ###########################

    print("Using 2D specialized functions")

    # Example using the DenseCRF2D code
    d = dcrf.DenseCRF2D(W, H, NLABELS)

    # get unary potentials (neg log probability)
    # U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
    U = unary_from_softmax(probs)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(9, 3), compat=30, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(9, 3), srgb=(30, 10, 10), rgbim=rgb,
                           compat=100,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    # height pairwise energy
    height_energy = create_pairwise_bilateral(sdims=(9, 3), schan=(5, ), img=height[..., np.newaxis], chdim=2)
    d.addPairwiseEnergy(height_energy, compat=60)

    # img_en = height_energy.reshape((-1, H, W))
    # cv2.imshow('x', np.uint8(img_en[0]))
    # cv2.waitKey(0)
    # cv2.imshow('y', np.uint8(img_en[1]))
    # cv2.waitKey(0)
    # cv2.imshow('c', np.uint8(img_en[2]))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    ####################################
    ### Do inference and compute MAP ###
    ####################################

    start = time.process_time()
    # Run five inference steps.
    Q = d.inference(5)
    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)
    res = (MAP.reshape((H, W)) * 255).astype(np.uint8)
    # cv2.imshow('res', res)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(output_path, name), res)
    end = time.process_time()
    print('Inference time: {:.3f} s.'.format(end-start))

