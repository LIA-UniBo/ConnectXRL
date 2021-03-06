from typing import Tuple, Optional, List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.connectx.constraints import ConstraintType, Constraints
from src.connectx.environment import convert_state_to_image


class Policy(nn.Module):
    """
    Common policy interface. Extends the nn.Module adding a predict function which encapsulates the logic of the game
    and can be used at testing time.
    """
    def __init__(self, based_on_image):
        """

        :param based_on_image: if True the policy uses images as input otherwise the board
        """
        super(Policy, self).__init__()
        self.based_on_image = based_on_image

    def forward(self,
                image: torch.Tensor,
                board: Optional[torch.Tensor] = None) -> torch.Tensor:
        """

        :param image: input image (3 channels, 2D image)
        :param board: input board (1 channel, 2D board)
        :return: the board or the image depending on what kind of input the policy expects
        """

        if self.based_on_image or board is None:
            return image
        else:
            return board

    def predict(self,
                observation: dict,
                configuration: dict) -> int:
        """
        Logic used to give choose actions at prediction time. The input is similar to the interface required by the
        Kaggle environment.

        :param observation: turn's data (board as a list, mark as 1 or 2)
        :param configuration: environment's data (num of columns, num of rows, optional constraint type)
        :return: the action
        """

        c_type = configuration['c_type']
        constraints = Constraints(c_type) if c_type else None
        action = None
        # Get the board
        board = np.array(observation['board']).reshape((configuration['rows'], configuration['columns'], 1))
        # True if the policy represent the first player (1 in the board)
        first_player = observation['mark'] == 1

        # If LOGIC_PURE detect a critical situation the correct action is performed
        if constraints and c_type is ConstraintType.LOGIC_PURE:
            action = constraints.select_constrained_action(board.squeeze(),
                                                           first_player=first_player)

        # If there are no constraints ot the LOGIC_PURE constraint is not supported (multiple options) exploit the
        # learnt policy
        if constraints is None or \
                action is None or \
                (action.sum().item() != 1 and c_type is ConstraintType.LOGIC_PURE):

            # Transform the board to an image and get the action from the network
            with torch.no_grad():
                action = self.forward(torch.from_numpy(convert_state_to_image(board)),
                                      torch.from_numpy(board).squeeze().unsqueeze(0)).squeeze()

            # Safe policy estimation on the action values
            if constraints and c_type in [ConstraintType.SPE, ConstraintType.CDQN]:
                # Compute action masks
                constraints = constraints.select_constrained_action(board.squeeze(),
                                                                    first_player=first_player)
                # Set invalid actions to -inf
                action[constraints == 0] = -np.inf

        # Return the best action
        return action.max(0)[1].item()


class FeedForward(Policy):
    """
    Agent's policy based on feed forward neural network.
    """

    def __init__(self,
                 layers_shapes: List[int]):
        """
        :param layers_shapes: Feed forward's layers shapes including the input shape (board columns * board rows) and
        the output shape (actions)
        """

        super(FeedForward, self).__init__(False)
        self.action_space_size = layers_shapes[-1]
        self.input_shape = layers_shapes[0]

        if len(layers_shapes) < 2:
            raise ValueError('Layers must be at least two, input and output')

        self.fc = nn.Sequential(
            *[e for sub in [
                [nn.Linear(pl, nl), nn.ReLU()] for pl, nl in zip(layers_shapes[:-1], layers_shapes[1:])
            ] for e in sub]
        )

    def forward(self,
                image: torch.Tensor,
                board: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Normalize the input and then use the network to get the action logits

        :param image: input image (3 channels, 2D image)
        :param board: input board (1 channel, 2D board) containing 1 or 2 depending on the player
        :return: logits for each action
        """

        # Get the board and normalize the values
        x = super().forward(image, board) / 2.0

        # The content is a batch (batch_size, height, width)
        if len(x.shape) == 3:
            x = x.view(x.shape[0], x.shape[1] * x.shape[2])

        # Pass batch to fc layers
        return torch.tanh(self.fc(x))


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


class CNNPolicy(Policy):
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
        super(CNNPolicy, self).__init__(True)
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
            self.feature_size = self.feature_extractor(
                torch.zeros(1, *(self.input_shape[0], self.input_shape[1], self.input_shape[2]))
            ).view(1, -1).size(1)

        self.fc_head = nn.Sequential(
            nn.Linear(self.feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space_size),
        )

    def forward(self,
                image: torch.Tensor,
                board: Optional[torch.Tensor] = None) -> torch.Tensor:
        """

        :param image: input image (3 channels, 2D image)
        :param board: input board (1 channel, 2D board)
        :return: logits for each action
        """

        x = super().forward(image, board)

        # If only 3 dims the batch is created adding one
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if self.non_local:
            x = self.non_local_block(x)
        # Extract features
        x = self.feature_extractor(x)
        # Flatten and pass them to fc heads
        return torch.tanh(self.fc_head(x.view(x.size(0), -1)))
