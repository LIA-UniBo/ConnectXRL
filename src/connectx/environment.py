from typing import Union, Callable, Any, Tuple

import gym
import numpy as np
import matplotlib.pyplot as plt
from kaggle_environments import make, get
from matplotlib.axes import Axes
from webcolors import rgb_to_name

# Color ranges, namely [0-1] vs [0-255]
VMIN = 0
VMAX = 1


def convert_state_to_image(state: np.ndarray,
                           matplotlib: bool = False,
                           num_to_rgb: np.ndarray = np.array([[VMAX, VMAX, VMAX],
                                                              [VMAX, VMIN, VMIN],
                                                              [VMIN, VMIN, VMAX]])) -> Union[np.ndarray, None]:
    """

    :param state: the original state from the environment. Will contain values 0 for empty cells, 1 and 2 for the stones
    of the 1st player and 2nd players respectively.
    :param matplotlib: if True the channel is represented with the last dimension (3rd) so it is possible to show with
    matplotlib. Otherwise the representation is more suitable for PyTorch and the matrix type is converted to float32.
    :param num_to_rgb: how to map values to rgb colors.
    :return: an RGB image representing the state.
    """
    if state is None:
        return None

    state = num_to_rgb[state]
    if matplotlib:
        return np.squeeze(state)
    else:
        return state.transpose(2, 3, 0, 1).astype(np.float32)


def show_board_grid(ax: Axes,
                    rows: int,
                    columns: int,
                    color: str = 'black') -> None:
    """
    Applies a grid on the axes. Used to add a grid in the game board.

    :param rows: the rows on the board
    :param columns: the columns on the board
    :param ax: the axes where the grid is applied
    :param color: grid color (matplotlib format)
    """
    # Major ticks
    ax.set_xticks(np.arange(0, columns, 1))
    ax.set_yticks(np.arange(0, rows, 1))

    # Labels for major ticks
    ax.set_xticklabels(np.arange(1, columns + 1, 1))
    ax.set_yticklabels(np.arange(1, rows + 1, 1))

    # Minor ticks
    ax.set_xticks(np.arange(-.5, columns, 1), minor=True)
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color=color, linestyle='-', linewidth=2)


class ConnectXGymEnv(gym.Env):
    """
    The Gym environment where agents can be trained.

    """

    def __init__(self,
                 opponent: Union[str, Callable],
                 first: bool,
                 invalid_reward: float = -1.0,
                 victory_reward: float = 1.0,
                 lost_reward: float = -1.0,
                 draw_reward: float = 0.5):
        """

        :param first: if True the agent will be trained as the first player
        :param opponent: one of the predefined agents ("random", "negamax") or a custom one (Callable)
        """
        self.opponent = opponent
        self.first = first

        self.invalid_reward = invalid_reward
        self.victory_reward = victory_reward
        self.lost_reward = lost_reward
        self.draw_reward = draw_reward

        self.kaggle_env = make('connectx', debug=True)
        self.env = self.kaggle_env.train([None, opponent] if first else [opponent, None])
        self.rows = self.kaggle_env.configuration.rows
        self.columns = self.kaggle_env.configuration.columns

        # Gym's action and observation spaces
        # Action space is a discrete discrete distribution between 0 and the number of columns
        self.action_space = gym.spaces.Discrete(self.columns)
        # Observation space is a bi-dimensional space with the size of the board. The values 0 represent free cells, 1
        # the stones of the 1st player and 2 the stones of the 2nd player.
        self.observation_space = gym.spaces.Box(low=0,
                                                high=2,
                                                shape=(self.rows, self.columns, 1), dtype=np.int32)

        self.obs = None

    def reset(self) -> np.ndarray:
        """

        Reset the environment and the trainer.
        """
        self.obs = self.env.reset()
        return np.array(self.obs['board']).reshape(self.rows, self.columns, 1)

    def reward_shaping(self, original_reward, done, draw) -> float:
        """
        Modifies original rewards.

        :param original_reward: reward from the Kaggle environment
        :param done: True if the game has ended
        :return: the modified reward
        """
        if original_reward == 1:
            # The agent has won the game
            return self.victory_reward
        elif done:
            # The game ended in a draw
            if draw:
                return self.draw_reward
            # The opponent has won the game
            else:
                return self.lost_reward
        else:
            return -1 / (self.rows * self.columns)

    def step(self, action: int) -> Tuple[np.ndarray, float, Union[bool, Any], Union[dict, Any]]:
        """

        :param action: the chosen column
        :return: the new observation (matrix), reward, flag for episode ending and info dictionary
        """

        end_status = None

        # Check if the action is valid otherwise punish the agent
        if self.obs['board'][int(action)] == 0:
            draw = False
            if all(v for v in self.obs['board']):
                draw = True
                print(self.obs['board'])
            # Perform the action
            self.obs, original_reward, done, _ = self.env.step(int(action))
            # Modify the reward
            reward = self.reward_shaping(original_reward, done, draw)
            # Set victory status
            if original_reward == 1:
                end_status = 'victory'
            elif done:
                end_status = 'lost'
        else:
            reward, done, end_status = self.invalid_reward, True, 'invalid'
        # The observed board is returned as a matrix even if internally is used as an array
        return np.array(self.obs['board']).reshape(self.rows, self.columns, 1), reward, done, {'end_status': end_status}

    def render(self, **kwargs):
        """
        Renders a visual representation of the current state of the environment. Mode can be specified with a string and
        can assume values 'html', 'ipython', 'ansi', 'human' (default), 'rgb_image'

        :return depends on the mode.
        """
        mode = get(kwargs, str, "human", path=["mode"])
        if mode == 'rgb_image':
            state = np.array(self.kaggle_env.state[0]['observation']['board']).reshape(self.rows, self.columns, 1)

            empty_color = get(kwargs, list, [VMAX * 255, VMAX * 255, VMAX * 255], path=["empty_color"])
            first_player_color = get(kwargs, list, [VMAX * 255, VMIN * 255, VMIN * 255], path=["first_player_color"])
            second_player_color = get(kwargs, list, [VMIN * 255, VMIN * 255, VMAX * 255], path=["second_player_color"])
            state = convert_state_to_image(state, matplotlib=True, num_to_rgb=np.array([empty_color,
                                                                                        first_player_color,
                                                                                        second_player_color]))
            plt.figure(2)
            plt.clf()
            plt.xlabel(f'Player1: {rgb_to_name(first_player_color)} Player2: {rgb_to_name(second_player_color)}')
            plt.title('ConnectX')
            plt.imshow(state,
                       interpolation='none',
                       vmin=VMIN,
                       vmax=VMAX,
                       aspect='equal')
            show_board_grid(plt.axes(), self.rows, self.columns)

            # Pause a bit so that plots are updated and visible
            render_waiting_time = get(kwargs, float, 1, path=["render_waiting_time"])
            plt.pause(render_waiting_time)
            return state

        return self.kaggle_env.render(**kwargs)
