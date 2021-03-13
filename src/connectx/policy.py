from random import choice
from typing import Tuple

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


class CNNPolicy(nn.Module):
    """
    Agent's policy based on CNN.
    """

    def __init__(self,
                 action_space_size: int,
                 input_shape: Tuple):
        """

        :param action_space_size: number of possible actions
        :param input_shape: Expected input image shape (channels, height, width)
        """
        super(CNNPolicy, self).__init__()
        self.input_shape = input_shape
        self.action_space_size = action_space_size

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(),
        )

        self.feature_size = self.feature_extractor(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

        self.fc_head = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_space_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: input image
        :return: logits for each action
        """
        # If only 3 dims the batch is created adding one
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        # Extract features
        x = self.feature_extractor(x)
        # Flatten and pass them to fc heads
        return self.fc_head(x.view(x.size(0), -1))
