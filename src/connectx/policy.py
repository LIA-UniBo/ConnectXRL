from typing import Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F


class NonLocalBlock(nn.Module):
    """
    Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick.
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf
    """

    def __init__(self,
                 in_channels: int,
                 inter_channels: Optional[int] = None,
                 mode: str = 'embedded',
                 dimension: int = 3,
                 bn_layer: bool = False):
        """
        :param in_channels: original channel size (1024 in the paper)
        :param inter_channels: channel size inside the block if not specified reduced to half (512 in the paper)
        :param mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
        :param dimension: can be 1 (temporal), 2 (spatial), 3 (spatio-temporal)
        :param bn_layer: whether to add batch norm
        """
        super(NonLocalBlock, self).__init__()

        assert dimension in [1, 2, 3]

        kernel_size_dimension = {1: (1,),
                                 2: (1, 1),
                                 3: (1, 1, 1)}

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # The channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # Assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=kernel_size_dimension[dimension])

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.w_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=kernel_size_dimension[dimension]),
                bn(self.in_channels)
            )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local
            # block is identity mapping
            nn.init.constant_(self.w_z[1].weight, 0)
            nn.init.constant_(self.w_z[1].bias, 0)
        else:
            self.w_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=kernel_size_dimension[dimension])

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing
            # architecture
            nn.init.constant_(self.w_z.weight, 0)
            nn.init.constant_(self.w_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=kernel_size_dimension[dimension])
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=kernel_size_dimension[dimension])

        if self.mode == "concatenate":
            self.w_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1,
                          kernel_size=kernel_size_dimension[dimension]),
                nn.ReLU()
            )

    def forward(self, x):
        """
        :param x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        global f, f_dic_c
        batch_size = x.size(0)

        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.w_f(concat)
            f = f.view(f.shape[0], f.shape[2], f.shape[3])

        if self.mode == "gaussian" or self.mode == "embedded":
            f_dic_c = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            # number of position in x
            n = f.shape[-1]
            f_dic_c = f / n

        y = torch.matmul(f_dic_c, g_x)

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        w_y = self.w_z(y)
        # residual connection
        z = w_y + x

        return z


class CNNPolicy(nn.Module):
    """
    Agent's policy based on CNN.
    """

    def __init__(self,
                 action_space_size: int,
                 input_shape: Tuple,
                 non_local: bool = False):
        """

        :param action_space_size: number of possible actions
        :param input_shape: Expected input image shape (channels, height, width)
        """
        super(CNNPolicy, self).__init__()
        self.action_space_size = action_space_size
        self.input_shape = input_shape
        self.non_local = non_local
        if self.non_local:
            self.non_local_block = NonLocalBlock(input_shape[0], dimension=2)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU()
        )

        with torch.no_grad():
            self.feature_size = self.feature_extractor\
                (torch.zeros(1, *(self.input_shape[0], self.input_shape[1], self.input_shape[2]))).view(1, -1).size(1)

        self.fc_head = nn.Sequential(
            nn.Linear(self.feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: input image
        :return: logits for each action
        """
        # If only 3 dims the batch is created adding one
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if self.non_local:
            x = self.non_local_block(x)
        # Extract features
        x = self.feature_extractor(x)
        # Flatten and pass them to fc heads
        return self.fc_head(x.view(x.size(0), -1))
