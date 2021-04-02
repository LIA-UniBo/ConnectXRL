import torch
from scipy.signal import convolve2d
import numpy as np


def check_vertically(state, vertical_image, target):
    for i in range(vertical_image.shape[0]):
        for j in range(vertical_image.shape[1]):
            if vertical_image[i][j] == target:
                if i - 1 >= 0:
                    if state[i - 1][j] == 0:
                        return torch.tensor([j]).unsqueeze(dim=1)
    return None


def check_horizontally(state, horizontal_image, target):
    for i in range(horizontal_image.shape[0]):
        for j in range(horizontal_image.shape[1]):
            if horizontal_image[i][j] == target:
                if j - 1 >= 0:
                    if i + 1 >= state.shape[0]:
                        if state[i][j - 1] == 0:
                            return torch.tensor([j - 1]).unsqueeze(dim=1)
                    else:
                        if state[i + 1][j - 1] == 0:
                            return torch.tensor([j - 1]).unsqueeze(dim=1)
                if j + 3 < state.shape[1]:
                    if i + 1 >= state.shape[0]:
                        if state[i][j + 3] == 0:
                            return torch.tensor([j + 3]).unsqueeze(dim=1)
                    else:
                        if state[i + 1][j + 3] == 0:
                            return torch.tensor([j + 3]).unsqueeze(dim=1)
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
                if i + 3 < state.shape[0] and j - 3 < state.shape[1]:
                    if i + 4 >= state.shape[0]:
                        if state[i + 3][j - 3] == 0:
                            return torch.tensor([j - 3]).unsqueeze(dim=1)
                        else:
                            if state[i + 1][j - 3] != 0 and state[i + 3][j - 3] == 0:
                                return torch.tensor([j - 3]).unsqueeze(dim=1)
    return None


def check_logic(check, state, image):
    constrained_action_win = check(state, image, target=3)
    # Priority to win if possible
    if constrained_action_win is None:
        return check(state, image, target=-3)
    else:
        return constrained_action_win


class Constraints(object):
    def __init__(self):
        self.horizontal_kernel = np.array([[1, 1, 1]])
        self.vertical_kernel = np.transpose(np.array([[1, 1, 1]]))
        self.diag1_kernel = np.eye(3, dtype=np.uint8)
        self.diag2_kernel = np.fliplr(self.diag1_kernel)

    def select_constrained_action(self, state):
        if not state.any():
            constrained_action = torch.tensor([3]).unsqueeze(dim=1)
        else:
            state[state == 2] = -1
            constrained_action = self.check_win_loss_horizontal(state)
            if constrained_action is not None:
                constrained_action = self.check_win_loss_vertical(state)
            if constrained_action is not None:
                constrained_action = self.check_win_loss_first_diagonal(state)
            if constrained_action is not None:
                constrained_action = self.check_win_loss_second_diagonal(state)

        return constrained_action

    def check_win_loss_horizontal(self, state):
        state_horizontal = np.pad(state, [(0, 0), (1, 1)], mode='constant')
        horizontal_image = convolve2d(state_horizontal, self.horizontal_kernel, mode="valid")[:, 1:(state.shape[1] - 1)]

        return check_logic(check_horizontally, state, horizontal_image)

    def check_win_loss_vertical(self, state):
        state_vertical = np.pad(state, [(1, 1), (0, 0)], mode='constant')
        vertical_image = convolve2d(state_vertical, self.vertical_kernel, mode="valid")[1:(state.shape[1] - 1), :]
        return check_logic(check_vertically, state, vertical_image)

    def check_win_loss_first_diagonal(self, state):
        state_diagonal = np.pad(state, [(1, 1), (1, 1)], mode='constant')
        diagonal_image = convolve2d(state_diagonal, self.diag1_kernel, mode="valid")[1:(state.shape[1] - 1),
                         1:(state.shape[1] - 1)]
        return check_logic(check_first_diagonal, state, diagonal_image)

    def check_win_loss_second_diagonal(self, state):
        state_diagonal = np.pad(state, [(1, 1), (1, 1)], mode='constant')
        diagonal_image = convolve2d(state_diagonal, self.diag2_kernel, mode="valid")[1:(state.shape[1] - 1),
                         1:(state.shape[1] - 1)]
        return check_logic(check_second_diagonal, state, diagonal_image)
