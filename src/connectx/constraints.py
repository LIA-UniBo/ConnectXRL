from enum import Enum
from typing import Optional

import torch
from scipy.signal import convolve2d
import numpy as np


class ConstraintType(Enum):
    """
    Different ways to apply the constraints
    """

    # A logic strategy which decouples the logical and the learnt actions. The agent encapsulates a logical process to
    # solve particular situations, during training and testing. It only supports single action decisions, namely the
    # logic is not able to decide or help deciding between multiple possible actions (the network is used in that case).
    LOGIC_PURE = 1

    # The logic process is active only during training, in order to help the agent learning a better policy. At
    # inference time the agent will only use the learnt policy. It only supports single action decisions, namely the
    # logic is not able to decide or help deciding between multiple possible actions (the network is used in that case).
    LOGIC_TRAIN = 2

    # The logic is used to produce action masks used to regularize the loss function (Semantic Based Regularization)
    SBR = 3

    # The logic is used to create safe action sets which are used to restrict the actions (action masking) at execution
    # time (Safe Policy Extraction). This approach can lead to non-optimal policies under the given set of constraints.
    SPE = 4

    # The logic is used to create safe action sets as described in https://arxiv.org/pdf/2003.09398.pdf (Constrained
    # Deep Q-Networks). In this scenario the sets are used to constrain the Q-update by allowing the agent to perform
    # only the limited portion of legal actions.
    CDQN = 5


def check_vertically(state: np.array,
                     vertical_image: np.array,
                     target: int) -> Optional[int]:
    """
    Check possible wins or losts along columns of the board.

    :param state: original board
    :param vertical_image: convolved board's image using a vertical kernel on a padded image (the original height is
    maintained)
    :param target: positive number if it is checking a victory, alternatively is negative
    :return: critical position if found else None
    """
    for i in range(vertical_image.shape[0]):
        for j in range(vertical_image.shape[1]):
            # If there is a critical situation
            if vertical_image[i][j] == target:
                # If the column is not full
                if i >= 2:
                    # Check if there is space above the critical situation
                    if state[i - 2][j] == 0:
                        return j
    return None


def check_horizontally(state: np.array,
                       horizontal_image: np.array,
                       target: int) -> Optional[int]:
    """
    Check possible wins or losts along rows of the board.

    :param state: original board
    :param horizontal_image: convolved board's image using a horizontal kernel on a padded image (the original width is
    maintained)
    :param target: positive number if it is checking a victory, alternatively is negative
    :return: critical position if found else None
    """
    last_column_index = state.shape[1] - 1
    # Rows are counted starting from above the board
    first_row_index = state.shape[0] - 1
    for i in range(horizontal_image.shape[0]):
        for j in range(horizontal_image.shape[1]):
            # If there is a critical situation
            if horizontal_image[i][j] == target:
                # If you are not in the left border check on the left
                # e.g.: Avoid checking on the left in |1|1|1|0|0|0|0|
                if j >= 2:
                    # Check if there is row below (you are not in the first row)
                    if i < first_row_index:
                        # If the cell on the left is free and the cell below it is not free
                        # e.g.:
                        # |0|1|1|1|0|0|0|
                        # |1|2|1|2|0|0|0|
                        if state[i][j - 2] == 0 and state[i + 1][j - 2] != 0:
                            return j - 2
                    else:
                        # If the cell on the left is free
                        # e.g.: |0|1|1|1|0|0|0|
                        if state[i][j - 2] == 0:
                            return j - 2
                # If you are not in the right border check on the right
                # e.g.: Avoid checking on the right in |0|0|0|0|1|1|1|
                if j <= last_column_index - 2:
                    # Check if there is row below (you are not in the first row)
                    if i < first_row_index:
                        # If the cell on the right is free and the cell below it is not free
                        if state[i][j + 2] == 0 and state[i + 1][j + 2] != 0:
                            return j + 2
                    else:
                        # If the cell on the right is free
                        if state[i][j + 2] == 0:
                            return j + 2

    # Check each row of the original board
    pattern1 = np.array([1, 0, 1, 1]) * (target / abs(target))
    pattern2 = np.array([1, 1, 0, 1]) * (target / abs(target))

    for i, row in enumerate(state):
        # Check if there is a 1011 or 1101 pattern in the board
        for j in range(row.shape[0] - 3):
            if np.all(row[j:j + 4] == pattern1) or np.all(row[j:j + 4] == pattern2):
                # Find the empty index
                empty_index = (j + 2) if state[i][j + 2] == 0 else (j + 1)
                # Check if it is the first row
                if i == first_row_index:
                    return empty_index
                # Else check if the row below has the cell not empty
                elif state[i + 1][empty_index] != 0:
                    return empty_index

    return None


class Constraints(object):
    """
    Class which encapsulates the constraint logics.
    """

    def __init__(self, c_type: ConstraintType, first_player: bool = True):
        """

        :param c_type: the constraint type represented
        :param first_player: if True the current player is number 1 the opponent will be number 2 and vice versa
        """
        # Define kernels used in the convolutions to detect the critical situations
        self.c_type = c_type

        self.player_number = 1 if first_player else 2

        self.horizontal_kernel = np.array([[1, 1, 1]])
        self.vertical_kernel = np.transpose(self.horizontal_kernel)

        # diag1 = \
        self.diag1_kernels = []
        # diag2 = /
        self.diag2_kernels = []

        for k in range(4):
            # Do not change the order of the kernels, they must have a diagonal structure
            kernel = [[1 if i == j and j != k else 0 for j in range(4)] for i in range(4)]
            self.diag1_kernels.append(np.vstack(kernel))
            self.diag2_kernels.insert(0, np.fliplr(self.diag1_kernels[-1]))

    def select_constrained_action(self, state: np.array) -> torch.Tensor:
        """
        Checks every possible critical situations to constraint the action selection.

        :param state: the board as a bi-dimensional array
        :return: a tensor of 1 and 0 representing respectively actions which may lead to secure or insecure situations,
        called action mask.
        """

        # Copy state to avoid modifying original one
        original_state = np.copy(state)
        state = np.copy(state)
        state[original_state == self.player_number] = 1
        state[original_state == (2 if self.player_number == 1 else 1)] = -1

        constrained_action = None

        # If the board is empty the player is the first one and should start in the middle of the board
        if not state.any():
            constrained_action = 3

        # Check first if it is possible to win (target = 3) then if there is a possibility to lose (target = -3)
        for target in [3, -3]:
            # If action hasn't been decided yet
            if constrained_action is None:
                constrained_action = self.check_win_loss_horizontal(state, target)
                if constrained_action is None:
                    constrained_action = self.check_win_loss_vertical(state, target)
                if constrained_action is None:
                    constrained_action = self.check_win_loss_diagonals(state, target)

        # Check invalid actions (full columns where placing a stone would lead to an invalid movement)
        invalid = torch.tensor([0 if state[0][j] != 0 else 1 for j in range(state.shape[1])])
        if constrained_action is not None and invalid[constrained_action] == 0:
            raise RuntimeError('Invalid action corresponds to a constrained action!')

        # No critical situations, the mask allow every possible action, otherwise some are masked
        if constrained_action is None:
            constrained_action = torch.ones(state.shape[1])
        else:
            constrained_action = torch.tensor([1 if i == constrained_action else 0 for i in range(state.shape[1])])

        # Invalid actions lead to multiple choice situations, which is not supported by LOGIC_* constraints
        return constrained_action * invalid if self.c_type not in (ConstraintType.LOGIC_PURE,
                                                                   ConstraintType.LOGIC_TRAIN) else constrained_action

    def check_win_loss_horizontal(self, state: np.array, target: int) -> Optional[int]:
        """
        Chack the board rows.

        :param state: the board as a bidimensional array
        :param target: positive number if it is checking a victory, alternatively is negative
        :return: None if no critical situations are spotted otherwise the action the player must take to win in this
        round or to prevent the opponent from winning in the following round.
        """
        # Example of padding plus convolution
        # |     ...     |    |     ...     |
        # |1|1|1|1|1|1|1| -> |2|3|3|3|3|3|2|
        state_horizontal = np.pad(state, [(0, 0), (1, 1)], mode='constant')
        horizontal_image = convolve2d(state_horizontal, self.horizontal_kernel, mode="valid")

        return check_horizontally(state, horizontal_image, target)

    def check_win_loss_vertical(self, state: np.array, target: int) -> Optional[int]:
        """
        Check the board columns.

        :param state: the board as a bidimensional array
        :param target: positive number if it is checking a victory, alternatively is negative
        :return: None if no critical situations are spotted otherwise the action the player must take to win in this
        round or to prevent the opponent from winning in the following round.
        """
        # Example of padding plus convolution
        # |0| ..       |0| ..
        # |0| ..       |0| ..
        # |0| ..  -->  |1| ..
        # |1| ..       |2| ..
        # |1| ..       |3| ..
        # |1| ..       |2| ..
        state_vertical = np.pad(state, [(1, 1), (0, 0)], mode='constant')
        vertical_image = convolve2d(state_vertical, self.vertical_kernel, mode="valid")
        return check_vertically(state, vertical_image, target)

    def check_win_loss_diagonals(self, state: np.array, target: int) -> Optional[int]:
        """
        Check the board diagonals.

        :param state: the board as a bidimensional array
        :param target: positive number if it is checking a victory, alternatively is negative
        :return: None if no critical situations are spotted otherwise the action the player must take to win in this
        round or to prevent the opponent from winning in the following round.
        """

        # Check diagonal
        for k_i, k in enumerate(self.diag1_kernels):
            k_w = k.shape[0]
            k_h = k.shape[1]

            k = k * int(target / abs(target))
            for i in range(state.shape[0] - k_w + 1):
                for j in range(state.shape[1] - k_h + 1):
                    if np.all((np.diag(state[i:i + k_w, j:j + k_h]) == np.diag(k))):
                        # Empty cell position is based on the ordering of the kernels and on the square nature of the
                        # kernels

                        # Check if first row
                        if (i + k_i) >= state.shape[0] - 1:
                            return j + k_i
                        # Check if the row below has the cell not empty
                        elif state[i + k_i + 1][j + k_i] != 0:
                            return j + k_i

        # Check anti-diagonal
        for k_i, k in enumerate(self.diag2_kernels):
            k_w = k.shape[0]
            k_h = k.shape[1]

            k = k * int(target / abs(target))
            for i in range(state.shape[0] - k_w + 1):
                for j in range(state.shape[1] - k_h + 1):
                    # Flip the kernel just to extract the anti-diagonal as the main diagonal
                    if np.all((np.diag(np.fliplr(state[i:i + k_w, j:j + k_h])) == np.diag(np.fliplr(k)))):
                        # Empty cell position is based on the ordering of the kernels and on the square nature of the
                        # kernels
                        empty_cell = k_h - k_i

                        # Check if first row
                        if (i + empty_cell) >= state.shape[0] - 1:
                            return j + k_i
                        # Check if the row below has the cell not empty
                        elif state[i + empty_cell + 1][j + k_i] != 0:
                            return j + k_i
        return None
