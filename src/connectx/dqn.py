import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F

from src.connectx.environment import ConnectXGymEnv, convert_state_to_image
from src.connectx.policy import CNNPolicy

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


def countplot(plot_id: int,
              data: list,
              labels: list,
              title: str) -> None:
    """
    Create a bar plot.

    :param plot_id: plot id
    :param data: the list containing the values
    :param labels: the list containing the labels
    :param title: the title of the bar plot
    """

    plt.figure(plot_id)
    plt.clf()
    data_torch = torch.tensor(data, dtype=torch.float)
    plt.title(title)
    plt.bar(labels, data_torch.numpy())

    # Pause a bit so that plots are updated
    plt.pause(0.001)


def lineplot(plot_id: int,
             data: list,
             title: str,
             xlabel: str,
             ylabel: str,
             points: list = (),
             points_style: list = (),
             hline: float = None) -> None:
    """
    Plot the data of the last episodes.

    :param plot_id: plot id
    :param data: list to plot
    :param title: title of the plot
    :param xlabel: title of the x-axis
    :param ylabel: title of the y-axis
    :param points: list of lists of positions where to draw points on the line
    :param points_style: list of dicts representing the style of each list of positions
    :param hline: vertical coordinate where to draw a line
    """

    plt.figure(plot_id)
    plt.clf()
    data_torch = torch.tensor(data, dtype=torch.float)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(data_torch.numpy())
    for i, p in enumerate(points):
        plt.scatter(p, data_torch[p], **points_style[i])
    if hline is not None:
        plt.axhline(hline, color='red', linestyle='dashed')

    # Pause a bit so that plots are updated
    plt.pause(0.001)

    """
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
    """


class ReplayMemory(object):
    """
    A cyclic buffer of bounded size that holds the transitions observed recently which are used to train the agent.
    """

    def __init__(self, capacity: int):
        """

        :param capacity: the size of the replay memory
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """
        Saves a transition. When the total space is filled the older data is replaced.
        Usually a transition it is something like (state, action) pair mapped to its (next_state, reward) result.

        :param args: the transition components
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> list:
        """
        Randomly sample some batch_size elements from the memory.

        :param batch_size: the size of the batch.
        :return: the sampled batch.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """

        :return: the current size of the memory
        """
        return len(self.memory)


class DQN(object):
    def __init__(self,
                 env,
                 batch_size: int = 128,
                 gamma: float = 0.999,
                 eps_start: float = 0.9,
                 eps_end: float = 0.05,
                 eps_decay: float = 200,
                 memory_size: int = 10000,
                 target_update: int = 10,
                 learning_rate: float = 1e-3,
                 epochs: int = 5,
                 device: str = 'cpu'):
        """

        TODO
        :param env: the Gym environment where it is applied
        :param batch_size:
        :param gamma:
        :param eps_start:
        :param eps_end:
        :param eps_decay:
        :param memory_size:
        :param target_update:
        :param learning_rate:
        :param device: the device where the training occurs, 'cpu', 'gpu' ...
        """

        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.epochs = epochs
        self.device = device

        # Get number of actions from gym action space
        self.n_actions = env.action_space.n

        self.memory = ReplayMemory(memory_size)

        # Policy and optimizer
        # Get initial state and its size
        init_screen = convert_state_to_image(self.env.reset())
        screen_shape = (init_screen.shape[1], init_screen.shape[2], init_screen.shape[3])

        self.policy_net = CNNPolicy(self.n_actions, screen_shape).to(device)
        self.target_net = CNNPolicy(self.n_actions, screen_shape).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Put target network on evaluation mode
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Metrics
        # Steps done is used for action selection and computation of eps threshold
        self.steps_done = 0

    def optimize_model(self, losses):
        """
        Optimize the policy's neural network.

        :param losses: list keeping track of the loss
        """

        # If there are not enough samples exit
        if len(self.memory) < self.batch_size:
            return

        for epoch in range(self.epochs):
            transitions = self.memory.sample(self.batch_size)
            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts batch-array of Transitions
            # to Transition of batch-arrays.
            batch = Transition(*zip(*transitions))

            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), device=self.device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1)[0].
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(self.batch_size, device=self.device)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch

            # Compute Huber loss
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

            # Track loss
            losses.append(loss.detach().item())

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

    def training_loop(self, num_episodes: int = 50,
                      save_path: str = None,
                      save_frequency: int = 1000,
                      render_env: bool = False,
                      render_waiting_time: float = 1,
                      plot_duration: bool = True,
                      plot_mean_reward: bool = True,
                      plot_actions_count: bool = True):
        """
        The DQN training algorithm.

        :param num_episodes: the number of episodes to train
        :param save_path: path where the model is saved at the end
        :param render_env: If true render the game board at each step
        :param render_waiting_time: paused time between a step and another
        :param plot_duration: if True plot the duration of each episode at the end
        :param plot_mean_reward: if True tracks and plots the average reward at each episode
        :param plot_actions_count: if True plots a bar plot representing the counter of actions taken
        """
        # Keep track of rewards
        episodes_rewards = []
        # Keep track of victories and losses
        episodes_victories = []
        episodes_losts = []
        # Keep track of the episodes' durations
        episode_durations = []
        # Keep track of actions taken
        action_counts = {i: 0 for i in range(1, self.n_actions + 1)}
        # Losses
        losses = []
        # Eps
        eps = []

        for i_episode in range(num_episodes):
            # Initialize the environment and state
            state = self.env.reset()
            # Get image and convert to torch tensor
            screen = torch.from_numpy(convert_state_to_image(state))

            step_rewards = []

            for t in count():
                # Select and perform an action on the environment
                action = self.select_action(screen, eps)
                new_state, reward, done, info = self.env.step(action.item())
                reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
                new_screen = torch.from_numpy(convert_state_to_image(new_state))

                # Metrics
                step_rewards.append(reward.detach().item())
                action_counts[action.detach().item() + 1] += 1

                if done:
                    new_state = None
                    new_screen = None
                    if info['end_status'] == 'victory':
                        episodes_victories.append(i_episode)
                    elif info['end_status'] == 'lost':
                        episodes_losts.append(i_episode)

                # Store the transition in memory
                self.memory.push(screen, action, new_screen, reward)

                # Update old state and image
                # state = new_state
                screen = new_screen

                # Perform one step of the optimization (on the target network)
                self.optimize_model(losses)

                # Rendering
                if render_env:
                    self.env.render(mode='rgb_image', render_waiting_time=render_waiting_time)

                if done:
                    episode_durations.append(t + 1)
                    episodes_rewards.append(np.sum(step_rewards))

                    lineplot(0, losses, 'Losses', 'Optimization steps', 'Value')
                    lineplot(1, eps, 'Eps', 'Steps', 'Value')
                    if plot_duration:
                        lineplot(2, episode_durations, 'Episodes durations', 'Episodes', 'Durations')
                    if plot_mean_reward:
                        lineplot(3,
                                 episodes_rewards,
                                 'Episode cumulative rewards',
                                 f'Episodes (Victories: {len(episodes_victories)} / {i_episode + 1})',
                                 'Rewards',
                                 points=[episodes_victories, episodes_losts],
                                 points_style=[{'c': 'green', 'marker': '*'}, {'c': 'red', 'marker': '.'}],
                                 hline=0.0)
                    if plot_actions_count:
                        countplot(4, list(action_counts.values()), list(action_counts.keys()), 'Actions taken')
                    break

            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if save_path and (i_episode + 1) % save_frequency == 0:
                save_filename = save_path + f'weights_{i_episode + 1}.pt'
                print(f'Model saved at {save_filename}')
                torch.save(self.target_net.state_dict(), save_filename)

        plt.figure(5)
        plt.title('Rolling average (over 1000 episodes) of cumulative rewards.')
        pd.DataFrame({'r': episodes_rewards})['r'].rolling(window=1000).mean().plot()
        plt.show()
        print('Training complete')

    def select_action(self, state, eps) -> torch.Tensor:
        """
        Apply the policy on the observed state to decide the next action or explore by choosing a random action.

        :param state: the current state
        :return: the chosen action
        """
        sample = random.random()
        # Push the agent to exploit after many steps rather than explore
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)

        eps.append(eps_threshold)

        self.steps_done += 1

        if sample > eps_threshold:
            # Exploitation
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # Exploration
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)


env = ConnectXGymEnv('random', True)
dqn = DQN(env)

dqn.training_loop(10000, save_path='./')
