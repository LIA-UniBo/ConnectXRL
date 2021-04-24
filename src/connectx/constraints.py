from enum import Enum
from typing import Optional, Callable

import torch
from scipy.signal import convolve2d
import numpy as np


class ConstraintType(Enum):
    """
    Different ways to apply the constraints
    """

    # The logic strategy decouples the logical and the learnt actions. The agent encapsulates a logical process to
    # solve particular situations.
    LOGIC_PURE = 1

    # The logic process is active only during training, in order to help the agent learning a better policy. At
    # inference time the agent will only use the learnt policy.
    LOGIC_TRAIN = 2

    # The logic is used to produce vectors used to regularize the loss function (Semantic Based Regularization)
    SBR = 3

    # The logic is used to create the safe action sets as described in https://arxiv.org/pdf/2003.09398.pdf (Constrained
    # Deep Q-Networks)
    CDQN = 4


def check_vertically(state, vertical_image, target):
    for i in range(vertical_image.shape[0]):
        for j in range(vertical_image.shape[1]):
            if vertical_image[i][j] == target:
                if i >= 2:
                    if state[i - 2][j] == 0:
                        return torch.tensor([j]).unsqueeze(dim=1)
    return None


# 1111111 -> 011111110 -> 2333332
def check_horizontally(state, horizontal_image, target):
    last_column_index = state.shape[1] - 1
    for i in range(horizontal_image.shape[0]):
        for j in range(horizontal_image.shape[1]):
            if horizontal_image[i][j] == target:
                if j >= 2:
                    if i < state.shape[0] - 1:
                        if state[i][j - 2] == 0 and state[i + 1][j - 2] != 0:
                            return torch.tensor([j - 2]).unsqueeze(dim=1)
                    else:
                        if state[i][j - 2] == 0:
                            return torch.tensor([j - 2]).unsqueeze(dim=1)
                if j <= last_column_index - 2:
                    if i < state.shape[0] - 1:
                        if state[i][j + 2] == 0 and state[i + 1][j + 2] != 0:
                            return torch.tensor([j + 2]).unsqueeze(dim=1)
                    else:
                        if state[i][j + 2] == 0:
                            return torch.tensor([j + 2]).unsqueeze(dim=1)
    return None


def check_first_diagonal(state, diagonal_image, target):
    for i in range(diagonal_image.shape[0]):
        for j in range(diagonal_image.shape[1]):
            if diagonal_image[i][j] == target:
                if i - 2 >= 0 and j - 2 >= 0:
                    if state[i - 1][j - 1] == 0 and state[i][j - 1] != 0:
                        return torch.tensor([j - 1]).unsqueeze(dim=1)
                if i + 3 < state.shape[0] and j + 3 < state.shape[1]:
                    if i + 4 >= state.shape[0]:
                        if state[i + 3][j + 3] == 0:
                            return torch.tensor([j + 3]).unsqueeze(dim=1)
                        else:
                            if state[i + 1][j + 3] != 0 and state[i + 3][j + 3] == 0:
                                return torch.tensor([j + 3]).unsqueeze(dim=1)

    return None


def check_second_diagonal(state, diagonal_image, target):
    for i in range(diagonal_image.shape[0]):
        for j in range(diagonal_image.shape[1]):
            if diagonal_image[i][j] == target:
                if i - 2 >= 0 and j + 3 < state.shape[1]:
                    if state[i - 1][j + 3] == 0 and state[i][j + 3] != 0:
                        return torch.tensor([j + 3]).unsqueeze(dim=1)
                if i + 3 < state.shape[0] and state.shape[1] > j - 3 >= 0:
                    if i + 4 >= state.shape[0]:
                        if state[i + 3][j - 3] == 0:
                            return torch.tensor([j - 3]).unsqueeze(dim=1)
                        else:
                            if state[i + 1][j - 3] != 0 and state[i + 3][j - 3] == 0:
                                return torch.tensor([j - 3]).unsqueeze(dim=1)
    return None


def check_logic(check: Callable,
                state: np.array,
                image: np.array) -> Optional[int]:
    """
    First whether the player is about to win otherwise repeat the process for the opponent in order to find a
    constrained action to perform.

    :param check: function used to check
    :param state: the board as a bidimensional array
    :param image: the board transformed by convolutions used by the check function
    :return: None if no critical situations are spotted otherwise the action the player must take to win in this
    round or to prevent the opponent from winning in the following round.
    """
    constrained_action_win = check(state, image, target=3)
    # Priority to win if possible
    if constrained_action_win is None:
        return check(state, image, target=-3)
    else:
        return constrained_action_win


class Constraints(object):
    """
    Class which encapsulates the constraint logics.
    """

    def __init__(self, type: ConstraintType):
        """

        :param type: the constraint type represented
        """

        # TODO: Add info on the player (1 or 2?)
        # Define kernels used in the convolutions to detect the critical situations
        self.type = type

        self.horizontal_kernel = np.array([[1, 1, 1]])
        self.vertical_kernel = np.transpose(self.horizontal_kernel)
        self.diag1_kernel = np.eye(3, dtype=np.uint8)
        self.diag2_kernel = np.fliplr(self.diag1_kernel)

    def select_constrained_action(self, state: np.array) -> torch.Tensor:
        """
        Checks every possible critical situations to constraint the action selection.

        :param state: the board as a bidimensional array
        :return: a tensor of 1 and 0 representing respectively actions which may lead to secure or insecure situations.
        """

        state = np.copy(state)

        # If the board is empty the player is the first one and should start in the middle of the board
        if not state.any():
            constrained_action = torch.tensor([3]).unsqueeze(dim=1)
        else:
            state[state == 2] = -1
            constrained_action = self.check_win_loss_horizontal(state)
            if constrained_action is None:
                constrained_action = self.check_win_loss_vertical(state)
            if constrained_action is None:
                constrained_action = self.check_win_loss_first_diagonal(state)
            if constrained_action is None:
                constrained_action = self.check_win_loss_second_diagonal(state)

        # TODO: workaround
        if constrained_action is None:
            constrained_action = torch.zeros(state.shape[1])
        else:
            constrained_action = torch.tensor([1 if i == constrained_action else 0 for i in range(state.shape[1])])
        return constrained_action

    def check_win_loss_horizontal(self, state: np.array) -> Optional[int]:
        """
        Chack the board rows.

        :param state: the board as a bidimensional array
        :return: None if no critical situations are spotted otherwise the action the player must take to win in this
        round or to prevent the opponent from winning in the following round.
        """
        # Example of padding plus convolution
        # 011111110 -> 2333332
        state_horizontal = np.pad(state, [(0, 0), (1, 1)], mode='constant')
        horizontal_image = convolve2d(state_horizontal, self.horizontal_kernel, mode="valid")

        return check_logic(check_horizontally, state, horizontal_image)

    def check_win_loss_vertical(self, state: np.array) -> Optional[int]:
        """
        Check the board columns.

        :param state: the board as a bidimensional array
        :return: None if no critical situations are spotted otherwise the action the player must take to win in this
        round or to prevent the opponent from winning in the following round.
        """
        # Example of padding plus convolution
        # 0     0
        # 0     0
        # 0  -> 1
        # 1     2
        # 1     3
        # 1     2
        state_vertical = np.pad(state, [(1, 1), (0, 0)], mode='constant')
        vertical_image = convolve2d(state_vertical, self.vertical_kernel, mode="valid")
        return check_logic(check_vertically, state, vertical_image)

    def check_win_loss_first_diagonal(self, state: np.array) -> Optional[int]:
        """
        Check the board \\ diagonals.

        :param state: the board as a bidimensional array
        :return: None if no critical situations are spotted otherwise the action the player must take to win in this
        round or to prevent the opponent from winning in the following round.
        """

        state_diagonal = np.pad(state, [(1, 1), (1, 1)], mode='constant')
        diagonal_image = convolve2d(state_diagonal,
                                    self.diag1_kernel,
                                    mode="valid")[1:(state.shape[1] - 1), 1:(state.shape[1] - 1)]
        return check_logic(check_first_diagonal, state, diagonal_image)

    def check_win_loss_second_diagonal(self, state: np.array) -> Optional[int]:
        """
        Check the board / diagonals.

        :param state: the board as a bidimensional array
        :return: None if no critical situations are spotted otherwise the action the player must take to win in this
        round or to prevent the opponent from winning in the following round.
        """

        state_diagonal = np.pad(state, [(1, 1), (1, 1)], mode='constant')
        diagonal_image = convolve2d(state_diagonal,
                                    self.diag2_kernel,
                                    mode="valid")[1:(state.shape[1] - 1), 1:(state.shape[1] - 1)]
        return check_logic(check_second_diagonal, state, diagonal_image)
