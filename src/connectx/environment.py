from typing import Union, Callable, Any, Tuple, Optional

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
                                                              [VMIN, VMIN, VMAX]]),
                           first_player: bool = True,
                           keep_player_colour: bool = True) -> Union[np.ndarray, None]:
    """

    :param state: the original state from the environment. Will contain values 0 for empty cells, 1 and 2 for the stones
    of the 1st player and 2nd players respectively.
    :param matplotlib: if True the channel is represented with the last dimension (3rd) so it is possible to show with
    matplotlib. Otherwise the representation is more suitable for PyTorch and the matrix type is converted to float32.
    :param num_to_rgb: how to map number of players to RGB colors.
    :param first_player: if True the first player colour is the defined one, otherwise if keep_player_color is True
    will be swapped
    :param keep_player_colour: if True the agent color is maintained between player 1 and player 2
    :return: an RGB image representing the state.
    """
    if state is None:
        return None

    if keep_player_colour and not first_player:
        index = np.array([0, 2, 1])
        num_to_rgb = num_to_rgb[index]
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

    :param ax: the axes where the grid is applied
    :param rows: the rows on the board
    :param columns: the columns on the board
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
                 step_reward: Optional[float] = None,
                 invalid_reward: float = -1.0,
                 victory_reward: float = 1.0,
                 lost_reward: float = -1.0):
        """
        :param opponent: one of the predefined agents ("random", "negamax") or a custom one (Callable)
        :param first: if True the agent will be trained as the first player
        :param step_reward: reward at each step
        :param invalid_reward: reward get from performing an invalid action
        :param victory_reward: reward get from winning a match
        :param lost_reward: reward get from loosing a match
        """

        self.opponent = opponent
        self.first = first

        self.kaggle_env = make('connectx', debug=True)
        self.env = self.kaggle_env.train([None, opponent] if first else [opponent, None])
        self.rows = self.kaggle_env.configuration.rows
        self.columns = self.kaggle_env.configuration.columns

        self.step_reward = step_reward if step_reward is not None else -1 / (self.rows * self.columns)
        self.invalid_reward = invalid_reward
        self.victory_reward = victory_reward
        self.lost_reward = lost_reward

        # Gym's action and observation spaces
        # Action space is a discrete discrete distribution between 0 and the number of columns
        self.action_space = gym.spaces.Discrete(self.columns)
        # Observation space is a bi-dimensional space with the size of the board. The values 0 represent free cells, 1
        # the stones of the 1st player and 2 the stones of the 2nd player.
        self.observation_space = gym.spaces.Box(low=0,
                                                high=2,
                                                shape=(self.rows, self.columns, 1), dtype=np.int32)

        self.obs = None

    def set_first(self, new_first):
        """

        :param new_first: the new value to assign to first. If True the environment considers the training player as
        the first one
        """
        self.first = new_first
        self.kaggle_env = make('connectx', debug=True)
        self.env = self.kaggle_env.train([None, self.opponent] if self.first else [self.opponent, None])

    def reset(self) -> np.ndarray:
        """

        Reset the environment and the trainer.
        """
        self.obs = self.env.reset()
        return np.array(self.obs['board']).reshape((self.rows, self.columns, 1))

    def reward_shaping(self,
                       original_reward: int,
                       done: bool) -> float:
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
            # The opponent has won the game
            return self.lost_reward
        else:
            return self.step_reward

    def step(self, action: int) -> Tuple[np.ndarray, float, Union[bool, Any], Union[dict, Any]]:
        """

        :param action: the chosen column
        :return: the new observation (matrix), reward, flag for episode ending and info dictionary
        """

        end_status = None

        if not(all(v for v in self.obs['board'])) and self.obs['board'][int(action)] != 0:
            reward, done, end_status = self.invalid_reward, True, 'invalid'
        # Check if the action is valid otherwise punish the agent
        else:
            # Perform the action
            self.obs, original_reward, done, _ = self.env.step(int(action))
            # Modify the reward
            reward = self.reward_shaping(original_reward, done)
            # Set victory status
            if original_reward == 1.0:
                end_status = 'victory'
            elif done:
                end_status = 'lost'
        # The observed board is returned as a matrix even if internally is used as an array
        return np.array(self.obs['board']).reshape((self.rows, self.columns, 1)), reward, done, \
               {'end_status': end_status}

    def render(self, **kwargs):
        """
        Renders a visual representation of the current state of the environment. Mode can be specified with a string and
        can assume values 'html', 'ipython', 'ansi', 'human' (default), 'rgb_image'

        :return depends on the mode.
        """
        mode = get(kwargs, str, "human", path=["mode"])
        if mode == 'rgb_image':
            # Print a passed state otherwise retrieve the state from the environment
            state = get(kwargs,
                        np.ndarray,
                        np.array(
                            self.kaggle_env.state[0]['observation']['board']
                        ).reshape((self.rows, self.columns, 1)),
                        path=["board"])

            keep_player_colour = get(kwargs, bool, True, path=["keep_player_colour"])
            empty_color = get(kwargs, list, [VMAX * 255, VMAX * 255, VMAX * 255], path=["empty_color"])
            first_player_color = get(kwargs, list, [VMAX * 255, VMIN * 255, VMIN * 255], path=["first_player_color"])
            second_player_color = get(kwargs, list, [VMIN * 255, VMIN * 255, VMAX * 255], path=["second_player_color"])
            state = convert_state_to_image(state,
                                           matplotlib=True,
                                           num_to_rgb=np.array([empty_color,
                                                                first_player_color,
                                                                second_player_color]),
                                           first_player=self.first,
                                           keep_player_colour=keep_player_colour)
            fig = plt.figure(2)
            plt.clf()
            if not keep_player_colour:
                plt.xlabel(f'Player1: {rgb_to_name(first_player_color)}{" (you) " if self.first else " "}'
                           f'Player2: {rgb_to_name(second_player_color)}{" (you)" if not self.first else ""}')
            else:
                plt.xlabel(f'You: {rgb_to_name(first_player_color)}{" (first) " if self.first else "(second) "}'
                           f'Opponent: {rgb_to_name(second_player_color)}')
            plt.title('ConnectX')
            plt.imshow(state,
                       interpolation='none',
                       vmin=VMIN,
                       vmax=VMAX,
                       aspect='equal')
            show_board_grid(fig.get_axes()[0], self.rows, self.columns)

            # Pause a bit so that plots are updated and visible
            render_waiting_time = get(kwargs, float, 1, path=["render_waiting_time"])
            plt.pause(render_waiting_time)
            return state

        return self.kaggle_env.render(**kwargs)
