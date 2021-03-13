import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.optim as optim
import torch.nn.functional as F

from src.connectx.environment import ConnectXGymEnv, convert_state_to_image
from src.connectx.policy import CNNPolicy

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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
                 eps_deacy: float = 200,
                 target_update: int = 10,
                 device: str = 'cpu'):
        """

        :param env: the Gym environment where it is applied
        :param batch_size:
        :param gamma:
        :param eps_start:
        :param eps_end:
        :param eps_deacy:
        :param target_update:
        :param device: the device where the training occurs, 'cpu', 'gpu' ...
        """

        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_deacy
        self.target_update = target_update
        self.device = device

        # Get number of actions from gym action space
        self.n_actions = env.action_space.n

        self.memory = ReplayMemory(10000)

        # Policy and optimizer
        # Get initial state and its size
        init_screen = convert_state_to_image(self.env.reset())
        screen_shape = (init_screen.shape[1], init_screen.shape[2], init_screen.shape[3])

        self.policy_net = CNNPolicy(self.n_actions, screen_shape).to(device)
        self.target_net = CNNPolicy(self.n_actions, screen_shape).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Put target network on evaluation mode
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())

        # Metrics
        # Steps done is used for action selection and computation of eps threshold
        self.steps_done = 0
        # Keep track of the episodes' durations
        self.episode_durations = []

    def optimize_model(self):
        """
        Optimize the policy's neural network.
        """

        # If there are not enough samples exit
        if len(self.memory) < self.batch_size:
            return

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

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def training_loop(self, num_episodes: int = 50,
                      render_env: bool = False,
                      render_waiting_time: float = 1,
                      plot_duration: bool = True):
        """
        The DQN training algorithm.

        :param num_episodes: the number of episodes to train
        :param render_env: If true render the game board at each step
        :param render_waiting_time: paused time between a step and another
        :param plot_duration: if True plot the duration of each episode at the end
        """

        for i_episode in range(num_episodes):
            # Initialize the environment and state
            state = self.env.reset()
            # Get image and convert to torch tensor
            screen = torch.from_numpy(convert_state_to_image(state))

            for t in count():
                # Select and perform an action on the environment
                action = self.select_action(screen)
                new_state, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
                new_screen = torch.from_numpy(convert_state_to_image(new_state))

                if done:
                    new_state = None
                    new_screen = None

                # Store the transition in memory
                self.memory.push(screen, action, new_screen, reward)

                # Update old state and image
                # state = new_state
                screen = new_screen

                # Perform one step of the optimization (on the target network)
                self.optimize_model()

                # Rendering
                if render_env:
                    self.env.render(mode='rgb_image', render_waiting_time=render_waiting_time)

                if done:
                    self.episode_durations.append(t + 1)
                    if plot_duration:
                        self.plot_durations()
                    break

            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        print('Training complete')

    def select_action(self, state) -> torch.Tensor:
        """
        Apply the policy on the observed state to decide the next action or explore by choosing a random action.

        :param state: the current state
        :return: the chosen action
        """
        sample = random.random()
        # Push the agent to exploit after many steps rather than explore
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)

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

    def plot_durations(self):
        """
        Plot the durations of the last episodes.
        """

        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())

        # Pause a bit so that plots are updated
        plt.pause(0.001)

        """
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())
        """


env = ConnectXGymEnv('random', True)
dqn = DQN(env)

dqn.training_loop(1000)
