import torch
from scipy.signal import convolve2d
import numpy as np


def check_vertically(state, vertical_image, target):
    constrained_action = None
    for i in range(vertical_image.shape[0]):
        for j in range(vertical_image.shape[1]):
            if vertical_image[i][j] == target:
                if i - 2 >= 0:
                    if state[i - 2][j] != 0:
                        constrained_action = torch.tensor([i - 2]).unsqueeze(dim=1)

    return constrained_action


def check_horizontally(state, horizontal_image, target):
    constrained_action = None
    for i in range(horizontal_image.shape[0]):
        for j in range(horizontal_image.shape[1]):
            if horizontal_image[i][j] == target:
                if j - 2 >= 0:
                    if state[i - 1][j - 2] != 0:
                        constrained_action = torch.tensor([j - 2]).unsqueeze(dim=1)
                if j + 2 < state.shape[1]:
                    if state[i - 1][j + 2] != 0:
                        constrained_action = torch.tensor([j - 2]).unsqueeze(dim=1)

    return constrained_action


class Constraints(object):
    def __init__(self):
        self.horizontal_kernel = np.array([[1, 1, 1]])
        self.vertical_kernel = np.transpose(np.array([[1, 1, 1]]))
        # TODO: search diagonally
        # diag1_kernel = np.eye(4, dtype=np.uint8)
        # diag2_kernel = np.fliplr(diag1_kernel)

    def select_constrained_action(self, state):
        if not state.any():
            constrained_action = torch.tensor([3]).unsqueeze(dim=1)
        else:
            state[state == 2] = -1
            constrained_action = self.check_win_loss_horizontal(state)
            if constrained_action is not None:
                constrained_action = self.check_win_loss_vertical(state)

        return constrained_action

    def check_logic(self, check, state, image):
        # Look for actions to close an opponent's horizontal victory
        constrained_action_close = check(state, image, target=-3)
        # Look for actions to win horizontally
        constrained_action_win = check(state, image, target=3)

        # Priority to win if possible
        if constrained_action_win is None:
            return constrained_action_close
        else:
            return constrained_action_win

    def check_win_loss_horizontal(self, state):
        state_horizontal = np.pad(state, [(0, 0), (1, 1)], mode='constant')
        horizontal_image = convolve2d(state_horizontal, self.horizontal_kernel, mode="valid")[:, 1:(state.shape[1] - 1)]

        return self.check_logic(check_horizontally, state, horizontal_image)

    def check_win_loss_vertical(self, state):
        state_vertical = np.pad(state, [(1, 1), (0, 0)], mode='constant')
        vertical_image = convolve2d(state_vertical, self.vertical_kernel, mode="valid")[1:(state.shape[1] - 1), :]

        return self.check_logic(check_vertically, state, vertical_image)
