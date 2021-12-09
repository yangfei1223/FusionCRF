# -*- coding:utf-8 -*-
'''
@Time : 2021/11/25 下午8:34
@Author: yangfei
@File : pointnet2.py
'''
from .common import *
# from torch_geometric.nn import PointConv

class SetAbstractionModule(nn.Module):
    def __init__(self, in_channels, layers1, layers2=None, use_xyz=True):
        super(SetAbstractionModule, self).__init__()
        self.use_xyz = use_xyz
        self.in_channels = in_channels + 3 if self.use_xyz else in_channels

        # self.local_nn = nn.ModuleList()
        self.local_nn = nn.Sequential()
        self.global_nn = None

        for i in range(len(layers1)):
            in_dim = self.in_channels if i == 0 else layers1[i-1]
            out_dim = layers1[i]
            self.local_nn.add_module('MLP {}'.format(i), MLP(in_dim, out_dim, activation=nn.ReLU(inplace=True)))

        if layers2 is not None:
            self.global_nn = nn.Sequential()
            for i in range(len(layers2)):
                in_dim =  layers1[-1]  if i == 0 else layers2[i - 1]
                out_dim = layers2[i]
                self.local_nn.add_module('MLP {}'.format(i), MLP(in_dim, out_dim, activation=nn.ReLU(inplace=True)))


    def forward(self, x, pos_in, pos_out, neighbor_idx):
        """
        :param x: [N_in, F_in]
        :param pos_in: [N_in, D]
        :param pos_out: [N_out, D]
        :param neighbor_idx: [N_out, K]
        :return: [N_out, F_out]
        """
        x = gather_neighbors(x, neighbor_idx)         # [N_out, K, F_in]

        # aggregate local coordinates
        if self.use_xyz:
            neighors = gather_neighbors(pos_in, neighbor_idx)
            neighors = neighors - pos_out.unsqueeze(1)      # [N_out, K, D]
            x = torch.cat([x, neighors], dim=-1)

        x = self.local_nn(x)     # [N_out, K, F_out]
        x = x.max(dim=1)[0]

        if self.global_nn is not None:
            x = self.global_nn(x)

        return x


class FeaturePropagationModule(nn.Module):
    def __init__(self, in_channels, layers):
        super(FeaturePropagationModule, self).__init__()

        self.mlp = nn.Sequential()
        for i in range(len(layers)):
            in_dim = in_channels if i == 0 else layers[i-1]
            out_dim = layers[i]
            self.mlp.add_module('MLP {}'.format(i), MLP(in_dim, out_dim, activation=nn.ReLU(inplace=True)))

    def forward(self, x, x_, pos_in, pos_out, neighbor_idx):
        """
        :param x: [N_in, F_in]
        :param x_: [N_out, F_in]    # high resolution feature from the encoder
        :param pos_in: [N_in, D]
        :param pos_out: [N_out, D]
        :param neighbor_idx: [N_out, K]
        :return:
        """
        # three interpolate
        x = knn_interpolate(x, pos_in, pos_out, neighbor_idx, k=3)
        if x_ is not None:
            x = torch.cat([x, x_], dim=-1)
        x = self.mlp(x)

        return x


class PointNet2(Base):
    def __init__(self,
                 in_channels=6,
                 down_mlp=None,
                 up_mlp=None,
                 n_classes=2,
                 class_weights=None,
                 ignore_index=255):
        super(PointNet2, self).__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index

        in_dims = in_channels
        self.encoder = nn.ModuleList()
        for i, mlp in enumerate(down_mlp):
            self.encoder.add_module('SA-{}'.format(i), SetAbstractionModule(in_dims, mlp))
            in_dims = mlp[-1]

        self.decoder = nn.ModuleList()
        for i, mlp in enumerate(up_mlp):
            if i != len(up_mlp) - 1:
                in_dims += down_mlp[-i - 2][-1]
            self.decoder.add_module('FP-{}'.format(i), FeaturePropagationModule(in_dims, mlp))
            in_dims = mlp[-1]


        self.classifier = nn.Sequential(
            MLP(in_dims, 128, activation=nn.ReLU(inplace=True)),
            nn.Dropout(p=0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, data):
        x, ms = data.x, data.multiscale

        # encoder
        encoder_stack = []
        for i, sa in enumerate(self.encoder):
            x = sa(x, ms[i].pos, ms[i+1].pos, ms[i].sub_idx)
            encoder_stack.append(x)

        # decoder
        for j, fp in enumerate(self.decoder):
            x_ = encoder_stack[-j-2] if j != len(self.decoder) - 1 else None
            x = fp(x, x_, ms[-j-1].pos, ms[-j-2].pos, ms[-j-2].up_idx)

        self.y_hat = self.classifier(x)
        self.y = data.y

        return self.y_hat

    def compute_loss(self):
        class_weights = self.class_weights.to(self.y_hat.device) if self.class_weights is not None else None
        self.loss = F.cross_entropy(self.y_hat, self.y, weight=class_weights, ignore_index=self.ignore_index)
        return self.loss








