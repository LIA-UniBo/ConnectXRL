from enum import Enum
from typing import Union, Callable, Tuple

import gym
import numpy as np
from kaggle_environments import make


class Reward(Enum):
    INVALID_REWARD = -10.0
    VICTORY_REWARD = 1.0
    LOST_REWARD = -1.0


class ConnectXGymEnv(gym.Env):
    """
    The Gym environment where agents can be trained.

    """

    def __init__(self,
                 opponent: Union[str, Callable],
                 first: bool):
        """

        :param first: if True the agent will be trained as the first player
        :param opponent: one of the predefined agents ("random", "negamax") or a custom one (Callable)
        """
        self.opponent = opponent
        self.first = first

        kaggle_env = make('connectx', debug=True)
        self.env = kaggle_env.train([None, opponent] if first else [opponent, None])
        self.rows = kaggle_env.configuration.rows
        self.columns = kaggle_env.configuration.columns

        # Gym's action and observation spaces
        # Action space is a discrete discrete distribution between 0 and the number of columns
        self.action_space = gym.spaces.Discrete(self.columns)
        # Observation space is a bi-dimensional space with the size of the board. The values 0 represent free cells, 1
        # the stones of the 1st player and 2 the stones of the 2nd player.
        self.observation_space = gym.spaces.Box(low=0,
                                                high=2,
                                                shape=(self.rows, self.columns, 1), dtype=np.int32)

        self.obs = None

    def reset(self):
        """

        Reset the environment and the trainer.
        """
        self.obs = self.env.reset()
        return np.array(self.obs['board']).reshape(self.rows, self.columns, 1)

    def reward_shaping(self, original_reward, done):
        """
        Modifies original rewards.

        :param original_reward: reward from the Kaggle environment
        :param done: True if the game has ended
        :return: the modified reward
        """
        if original_reward == 1:
            # The agent has won the game
            return Reward.VICTORY_REWARD
        elif done:
            # The opponent has won the game
            return -Reward.LOST_REWARD
        else:
            return 1 / (self.rows * self.columns)

    def step(self, action: int) -> Tuple[np.ndarray, Union[int, float], bool, dict]:
        """

        :param action: the chosen column
        :return: the new observation (matrix), reward, flag for episode ending and info dictionary
        """

        # Check if the action is valid otherwise punish the agent
        if self.obs['board'][int(action)] == 0:
            # Perform the action
            self.obs, original_reward, done, _ = self.env.step(int(action))
            # Modify the reward
            reward = self.reward_shaping(original_reward, done)
        else:
            reward, done, _ = Reward.INVALID_REWARD, True, {}
        # The observed board is returned as a matrix even if internally is used as an array
        return np.array(self.obs['board']).reshape(self.rows, self.columns, 1), reward, done, _
