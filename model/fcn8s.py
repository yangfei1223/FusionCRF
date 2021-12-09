# -*- coding:utf-8 -*-
'''
@Time : 2021/11/25 下午8:34
@Author: yangfei
@File : fcn8s.py
'''
from .common import *
from torchvision.models.vgg import vgg16_bn

vgg = vgg16_bn(pretrained=True)

class FCN8s_pretrained(Base):
    def __init__(self,
                 in_channels,
                 n_classes,
                 class_weights=None,
                 ignore_index=255):
        super(FCN8s_pretrained, self).__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index

        self.features = vgg.features

        self.classifier = nn.Sequential(
            CONV_2D(512, 1024, 1, activation=nn.ReLU(inplace=True)),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(1024, n_classes, (1, 1))
        )

        self.score_pool4 = nn.Conv2d(512, n_classes, (1, 1))
        self.score_pool3 = nn.Conv2d(256, n_classes, (1, 1))

        self.upsample2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample8x = nn.UpsamplingBilinear2d(scale_factor=8)
        self.upsample32x = nn.UpsamplingBilinear2d(scale_factor=32)

    def forward(self, data):
        x = data.x

        x = self.features[:24](x)      # conv3
        score3 = self.score_pool3(x)
        x = self.features[24:34](x)    # conv4
        score4 = self.score_pool4(x)
        x = self.features[34:](x)

        x = self.classifier(x)

        x = self.upsample2x(x)                          # upsample 2x
        x = self.upsample2x(x + score4)                 # upsample 2x
        self.y_hat = self.upsample8x(x + score3)        # upsample 8x

        self.y = data.y

        return self.y_hat

    def compute_loss(self):
        class_weights = self.class_weights.to(self.y_hat.device) if self.class_weights is not None else None
        self.loss = F.cross_entropy(self.y_hat, self.y, weight=class_weights, ignore_index=self.ignore_index)
        return self.loss


class FCN8s(Base):
    '''
    VGG16 feature extractor
    '''
    def __init__(self,
                 in_channels,
                 n_classes,
                 class_weights=None,
                 ignore_index=255):
        super(FCN8s, self).__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index

        # vgg-16 encoders
        self.conv1 = nn.Sequential(
            CONV_2D(in_channels, 64, 3, padding=1, activation=nn.ReLU(inplace=True)),
            CONV_2D(64, 64, 3, padding=1, activation=nn.ReLU(inplace=True)),
        )
        self.conv2 = nn.Sequential(
            CONV_2D(64, 128, 3, padding=1, activation=nn.ReLU(inplace=True)),
            CONV_2D(128, 128, 3, padding=1, activation=nn.ReLU(inplace=True)),
        )
        self.conv3 = nn.Sequential(
            CONV_2D(128, 256, 3, padding=1, activation=nn.ReLU(inplace=True)),
            CONV_2D(256, 256, 3, padding=1, activation=nn.ReLU(inplace=True)),
            CONV_2D(256, 256, 3, padding=1, activation=nn.ReLU(inplace=True)),
        )
        self.conv4 = nn.Sequential(
            CONV_2D(256, 512, 3, padding=1, activation=nn.ReLU(inplace=True)),
            CONV_2D(512, 512, 3, padding=1, activation=nn.ReLU(inplace=True)),
            CONV_2D(512, 512, 3, padding=1, activation=nn.ReLU(inplace=True)),
        )
        self.conv5 = nn.Sequential(
            CONV_2D(512, 512, 3, padding=1, activation=nn.ReLU(inplace=True)),
            CONV_2D(512, 512, 3, padding=1, activation=nn.ReLU(inplace=True)),
            CONV_2D(512, 512, 3, padding=1, activation=nn.ReLU(inplace=True)),
        )

        self.classifier = nn.Sequential(
            CONV_2D(512, 1024, 1, activation=nn.ReLU(inplace=True)),
            nn.Conv2d(1024, n_classes, (1, 1)),
        )

        self.score_pool4 = nn.Conv2d(512, n_classes, (1, 1))
        self.score_pool3 = nn.Conv2d(256, n_classes, (1, 1))

    def forward(self, data):
        x = data.x
        x = F.max_pool2d(self.conv1(x), 2, 2)
        x = F.max_pool2d(self.conv2(x), 2, 2)
        x = F.max_pool2d(self.conv3(x), 2, 2)
        score3 = self.score_pool3(x)
        x = F.max_pool2d(self.conv4(x), 2, 2)
        score4 = self.score_pool4(x)
        x = F.max_pool2d(self.conv5(x), 2, 2)
        score = self.classifier(x)

        score = F.interpolate(score, scale_factor=2, mode='bilinear', align_corners=False)
        score = F.interpolate(score + score4, scale_factor=2, mode='bilinear', align_corners=False)
        self.y_hat = F.interpolate(score + score3, scale_factor=8, mode='bilinear', align_corners=False)
        self.y = data.y

        return self.y_hat

    def compute_loss(self):
        class_weights = self.class_weights.to(self.y.device) if self.class_weights is not None else None
        self.loss = F.cross_entropy(self.y_hat, self.y, weight=class_weights, ignore_index=self.ignore_index)
        return self.loss

