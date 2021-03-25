from random import choice
from typing import Tuple
from torch.nn import functional as F

import torch
from torch import nn

from src.connectx.environment import convert_state_to_image


def dqn_agent(observation: dict,
              configuration: dict) -> int:
    """
    Agent trained using DQN and trained on the images of the game.

    :param observation: turn's data (board status, step number, ...)
    :param configuration: environment's data (steps, board, timeouts, ...) and weights file path
    :return: the column where the stone is inserted
    """

    model = CNNPolicy(configuration.columns,
                      (3, configuration.rows, configuration.columns))

    model.load_state_dict(torch.load(configuration.weights_path))
    model.eval()

    col = model(torch.from_numpy(convert_state_to_image(observation.board)))

    # Check if selected column is valid
    is_valid = (observation.board[int(col)] == 0)

    # If not valid, select random move
    if is_valid:
        return int(col)
    else:
        return choice([col for col in range(configuration.columns) if observation.board[int(col)] == 0])


class NonLocalBlock(nn.Module):
    """
    spatial non-local block: https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, input_channels):
        """
        Spatial non-local attention over raws and columns
        :param input_channels: number of channels of input image
        :return: image with channels doubled
        """
        super().__init__()
        self.teta = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=(1, 1))
        self.fi = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=(1, 1))
        self.gi = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=(1, 1))
        self.out_1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=(1, 1))
        self.out_2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=(1, 1))
        self.flatten = nn.Flatten(start_dim=2)

    def forward(self, x):
        x_1 = self.flatten(self.teta(x))
        x_1 = x_1.view(x_1.shape[0], x_1.shape[2], x_1.shape[1])
        x_2 = self.flatten(self.fi(x))
        x_3 = self.flatten(self.gi(x))
        x_3 = x_3.view(x_3.shape[0], x_3.shape[2], x_3.shape[1])
        x_1_2 = torch.matmul(x_1, x_2)
        x_1_2_1 = F.softmax(x_1_2, dim=2)
        x_1_2_2 = F.softmax(x_1_2, dim=1)

        x_1_2_3_1 = self.out_1(
            torch.transpose(torch.matmul(x_1_2_1, x_3), dim0=1, dim1=2).view(x.shape[0], -1, x.shape[2], x.shape[3]))
        x_1_2_3_2 = self.out_2(
            torch.transpose(torch.matmul(x_1_2_2, x_3), dim0=1, dim1=2).view(x.shape[0], -1, x.shape[2], x.shape[3]))

        x = torch.cat([x, x], dim=1)
        x_aggl = torch.cat([x_1_2_3_1, x_1_2_3_2], dim=1)

        return F.relu(x + x_aggl)


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
        self.input_shape = input_shape
        self.action_space_size = action_space_size
        self.non_local = non_local
        if self.non_local:
            self.non_local_block = NonLocalBlock(input_shape[0])
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_shape[0] + input_shape[0] * self.non_local, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU()
        )

        with torch.no_grad():
            self.feature_size = self.feature_extractor\
                (torch.zeros(1, *(self.input_shape[0] + self.input_shape[0] * self.non_local,
                                  self.input_shape[1], self.input_shape[2]))).view(1, -1).size(1)

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
