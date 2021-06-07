import copy
import math
import os
import random
from typing import List, Optional, Tuple

import pylab as pl
import matplotlib.pyplot as plt
from IPython import display
from collections import namedtuple
from itertools import count

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from IPython.core.display import clear_output

from src.connectx.constraints import Constraints, ConstraintType
from src.connectx.environment import ConnectXGymEnv, convert_state_to_image
from src.connectx.plots import lineplot, countplot
from src.connectx.policy import Policy

from torchviz import make_dot

# Screen is the RGB image (3, rows, columns) representing the colorful board
# Board is the (rows, columns) representing the logical board (containing 1 and 2)
Transition = namedtuple('Transition', ('screen',
                                       'action',
                                       'next_screen',
                                       'reward',
                                       'board',
                                       'next_board'))

Statistics = namedtuple('Statistics', ('episodes_rewards',
                                       'episodes_victories',
                                       'episodes_losts',
                                       'episodes_performed_actions',
                                       'action_counts',
                                       'losses',
                                       'dual_losses',
                                       'sbr_coeffs',
                                       'eps'))


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

    def push(self, *args) -> None:
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


class SBRLagrangianDualLoss(nn.Module):
    """
    nn.Module used to train the SBR coefficient using the Lagrangian duality.
    """

    def __init__(self):
        super().__init__()
        self.sbr_coeff = nn.Parameter(torch.tensor([0.0]))

    def forward(self,
                state_action_values: torch.Tensor,
                expected_state_action_values: torch.Tensor,
                sbr_term: torch.Tensor) -> tuple:
        """
        :param state_action_values: the state values
        :param expected_state_action_values: the values of the non final states computed from the target network
        :param sbr_term: the sbr term obtained from the constraints
        :return: the dual loss
        """

        return -1 * (F.smooth_l1_loss(state_action_values, expected_state_action_values) +
                     (self.sbr_coeff * sbr_term).mean())


class DQN(object):
    def __init__(self,
                 env: ConnectXGymEnv,
                 policy: Policy,
                 batch_size: int = 128,
                 gamma: float = 0.99,
                 eps_start: float = 1.0,
                 eps_end: float = 0.01,
                 eps_decay: float = 10000,
                 memory_size: int = 10000,
                 target_update: int = 10,
                 learning_rate: float = 1e-2,
                 epochs: int = 2,
                 sbr_coeff: Optional[float] = None,
                 constraint_type: Optional[ConstraintType] = ConstraintType.LOGIC_TRAIN,
                 keep_player_colour: bool = True,
                 device: str = 'cpu',
                 notebook: bool = False):
        """

        :param env: the Gym environment used to initialize the network
        :param policy: the policy to train
        :param batch_size: size of samples from  the memory
        :param gamma: discount factor
        :param eps_start: epsilon-greedy initial value
        :param eps_end: epsilon-greedy final value
        :param eps_decay: epsilon-greedy dacay
        :param memory_size: size of the experience replay
        :param target_update: after how many episodes the target network is updated
        :param learning_rate: optimizer learning rate
        :param epochs: number of training epochs
        :param sbr_coeff: if not None is used for the SBR constraint regularization
        :param constraint_type: if not None indicates the constraint type
        :param keep_player_colour: if True the agent color is maintained between player 1 and player 2. e.g. You will
        always be the red player, otherwise the 1st player will be always the red one
        :param device: the device where the training occurs, 'cpu', 'gpu' ...
        :param notebook: if True it formats for notebook execution
        """

        self.env = env
        # Used to restore the initial value in the environment
        self.is_initially_first = env.first
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.epochs = epochs
        self.sbr_coeff = sbr_coeff
        self.device = device
        self.constraints = Constraints(constraint_type) if constraint_type else None
        self.keep_player_colour = keep_player_colour
        self.notebook = notebook

        # Get number of actions from gym action space
        self.n_actions = env.action_space.n

        self.memory = ReplayMemory(memory_size)

        # Policy and optimizer

        self.policy_net = policy.to(device)
        self.target_net = copy.deepcopy(self.policy_net).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Put target network on evaluation mode
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Metrics
        # Steps done is used for action selection and computation of eps threshold
        self.steps_done = 0

        self.lagrangian_dual_loss = SBRLagrangianDualLoss().to(device)
        self.lagrangian_optimizer = optim.Adam(self.lagrangian_dual_loss.parameters(), lr=learning_rate)

    def __compute_state_action_values__(self, transitions: list) -> tuple:
        """
        :param transitions: list of Transitions
        :return: state action values, state action values from the target network of the non final state and SBR terms
        from constraints
        """

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_screen)), device=self.device, dtype=torch.bool)

        # Find non final states (images and boards)
        if not all(v is None for v in batch.next_screen):
            non_final_next_states = torch.cat([ns if ns is not None else s
                                               for ns, s in zip(batch.next_screen, batch.screen)]).to(self.device)
        else:
            non_final_next_states = None
        if not all(v is None for v in batch.next_board):
            non_final_next_boards = torch.cat([nb if nb is not None else b
                                               for nb, b in zip(batch.next_board, batch.board)]).to(self.device)
        else:
            non_final_next_boards = None

        state_batch = torch.cat(batch.screen).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        board_batch = torch.cat(batch.board).to(self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t)
        state_action_values = self.policy_net(state_batch, board_batch)
        # SBR regularization term
        if self.constraints is not None and self.constraints.c_type is ConstraintType.SBR:
            constraint_actions = torch.stack([self.constraints.select_constrained_action(b.squeeze(),
                                                                                         self.env.first)
                                              for b in batch.board]).to(self.device)

            # The sbr_term penalty is equal to 0 when the chosen action is valid according to action mask
            # otherwise it is equal to sbr_coeff * Q_value ** 2 of the chosen action
            sbr_term = (1 * constraint_actions.gather(1, torch.argmax(state_action_values, dim=1)
                                                      .unsqueeze(dim=1).to(self.device))
                        == torch.zeros(state_action_values.shape[0]).to(self.device)) * \
                       state_action_values.gather(1, action_batch) ** 2
        else:
            sbr_term = None

        # Select the columns of the actions taken
        state_action_values = state_action_values.gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        # If all next states are None next_state_values is set to 0 in all cases of constraints
        if non_final_next_states is None:
            next_state_values = 0.0
        else:
            # CDQN action selection in the Q-update
            if self.constraints is not None and self.constraints.c_type is ConstraintType.CDQN:
                # Compute action masks for non final next states
                constraints = torch.stack([self.constraints.select_constrained_action(b.squeeze(), self.env.first)
                                           for b in batch.board]).to(self.device)
                # Get action values and set to -inf those who are masked out
                constrained_action = self.target_net(non_final_next_states, non_final_next_boards)
                constrained_action = torch.where(constraints == 0,
                                                 torch.full(constrained_action.shape, -np.inf).to(self.device),
                                                 constrained_action)
                next_state_values = torch.where(non_final_mask,
                                                constrained_action.max(1)[0].detach(),
                                                next_state_values)
            else:
                next_state_values = torch.where(non_final_mask,
                                                self.target_net(non_final_next_states,
                                                                non_final_next_boards).max(1)[0].detach(),
                                                next_state_values)

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        return state_action_values, expected_state_action_values, sbr_term

    def optimize_model(self,
                       losses: List[float],
                       dual_losses: List[float]) -> None:
        """
        Optimize the policy's neural network.

        :param losses: list keeping track of the loss
        :param dual_losses: list keeping track of the loss of the dual problem
        """

        # If there are not enough samples exit
        if len(self.memory) < self.batch_size:
            losses.append(0)
            return

        for epoch in range(self.epochs):
            transitions = self.memory.sample(self.batch_size)

            state_action_values, expected_state_action_values, sbr_term = \
                self.__compute_state_action_values__(transitions)

            # Get the SBR coefficient without recording gradients
            with torch.no_grad():
                sbr_primal = (self.lagrangian_dual_loss.sbr_coeff if self.sbr_coeff is None else self.sbr_coeff) \
                             * (sbr_term.mean() if sbr_term is not None else 0)
            # Compute Huber loss
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1)) +\
                   sbr_primal.to(self.device)

            # Track loss
            losses.append(loss.detach().item())

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            # make_dot(loss).render("primal_loss", format="png")

            # Optimize the SBR coefficient using the dual problem of the original optimization problem
            if self.constraints is not None \
                    and self.constraints.c_type is ConstraintType.SBR \
                    and self.sbr_coeff is None:
                # Recompute the values using the updated network
                state_action_values, expected_state_action_values, sbr_term = \
                    self.__compute_state_action_values__(transitions)

                # Detach because gradients from the network are not important here
                dual_loss = self.lagrangian_dual_loss(state_action_values.detach(),
                                                      expected_state_action_values.unsqueeze(1).detach(),
                                                      sbr_term.detach())
                # Track loss
                dual_losses.append(dual_loss.detach().item())

                # Optimize the dual for the SBR coefficient
                self.lagrangian_optimizer.zero_grad()
                dual_loss.backward()
                self.lagrangian_optimizer.step()
                # make_dot(dual_loss).render("dual_loss", format="png")

    def training_loop(self,
                      n_episodes_as_1st_player: int = 25,
                      n_episodes_as_2nd_player: int = 25,
                      save_path: Optional[str] = None,
                      save_frequency: int = 1000,
                      render_env: bool = False,
                      render_waiting_time: float = 1,
                      update_plots_frequency: int = 100,
                      plot_performed_actions: bool = True,
                      plot_mean_reward: bool = True,
                      plot_actions_count: bool = True,
                      avg_roll_window: int = 100) -> Statistics:
        """
        The DQN training algorithm.

        :param n_episodes_as_1st_player: the number of episodes to train the agent as the 1st player
        :param n_episodes_as_2nd_player: the number of episodes to train the agent as the 2nd player
        :param save_path: path where the model is saved at the end
        :param save_frequency: how many episodes between each weight saving
        :param render_env: If true render the game board at each step
        :param render_waiting_time: paused time between a step and another
        :param update_plots_frequency: how many episodes between each update of the plots
        :param plot_performed_actions: if True plot the number of performed actions of each episode at the end
        :param plot_mean_reward: if True tracks and plots the average reward at each episode
        :param plot_actions_count: if True plots a bar plot representing the counter of actions taken
        :param avg_roll_window: the window used to smooth the plots of the performed actions, losses and reward rolling
        averages
        """
        # Keep track of rewards
        episodes_rewards = []
        # Keep track of victories and losses
        episodes_victories = []
        episodes_losts = []
        # Keep track of the episodes' number of performed actions
        episodes_performed_actions = []
        # Keep track of actions taken
        action_counts = {i: 0 for i in range(1, self.n_actions + 1)}
        # Losses
        losses = []
        dual_losses = []
        sbr_coeffs = []
        # Eps
        eps = []

        if self.notebook:
            fig, axs = pl.subplots(3, 2, figsize=(22, 12), constrained_layout=True)
        else:
            fig, axs = plt.subplots(3, 2, constrained_layout=True)
        fig.suptitle(f'DQN ({self.constraints.c_type if self.constraints else None}), batch size = {self.batch_size}, '
                     f'Opponent {self.env.opponent}, '
                     f'memory_size = {self.memory.capacity}, gamma = {self.gamma}, target update every '
                     f'{self.target_update}\noptimizer = {self.optimizer.__str__().replace(os.linesep, "")}',
                     fontsize=9)

        # List used to randomly select if the agent will play as the first (True) or as second player (False)
        first_or_second = [True for _ in range(n_episodes_as_1st_player)]
        first_or_second += [False for _ in range(n_episodes_as_2nd_player)]
        random.shuffle(first_or_second)

        for i_episode in range(n_episodes_as_1st_player + n_episodes_as_2nd_player):

            # Change first or second player
            self.env.set_first(first_or_second[i_episode])

            # Initialize the environment and state
            state = self.env.reset()
            # Get image and convert to torch tensor
            screen = torch.from_numpy(
                convert_state_to_image(state=state,
                                       first_player=first_or_second[i_episode],
                                       keep_player_colour=self.keep_player_colour)
            ).to(self.device)

            step_rewards = []
            step_eps = []

            for t in count():
                # Select and perform an action on the environment
                action, logic_pure_action = self.select_action(screen, state, step_eps)

                new_state, reward, done, info = self.env.step(action.item())
                reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
                new_screen = torch.from_numpy(
                    convert_state_to_image(state=new_state,
                                           first_player=first_or_second[i_episode],
                                           keep_player_colour=self.keep_player_colour)
                ).to(self.device)

                # Metrics
                step_rewards.append(reward.detach().item())
                action_counts[action.detach().item() + 1] += 1
                push = True
                if done:
                    if info['end_status'] == 'victory':
                        episodes_victories.append(i_episode)
                    elif info['end_status'] == 'lost':
                        episodes_losts.append(i_episode)
                        self.memory.memory[self.memory.position - 1]._replace(reward=reward)
                        push = False
                    elif info['end_status'] == 'invalid':
                        episodes_losts.append(i_episode)
                    new_screen = None

                # Store the transition in memory if match has not ended and constrained action is from LOGIC_PURE method
                if push and not logic_pure_action:
                    self.memory.push(screen,
                                     action.to(self.device),
                                     new_screen,
                                     reward,
                                     torch.from_numpy(state).squeeze().unsqueeze(0),
                                     torch.from_numpy(new_state).squeeze().unsqueeze(0))

                # Update old state and image
                # state = new_state
                screen = new_screen
                state = new_state
                # Perform one step of the optimization (on the target network)
                self.optimize_model(losses, dual_losses)

                # Rendering
                if render_env:
                    self.env.render(mode='rgb_image',
                                    render_waiting_time=render_waiting_time,
                                    keep_player_colour=self.keep_player_colour)

                if done:
                    episodes_performed_actions.append(t + 1)
                    episodes_rewards.append(np.sum(step_rewards))
                    eps.append(step_eps[0])
                    sbr_coeffs.append(self.sbr_coeff if self.sbr_coeff is not None
                                      else self.lagrangian_dual_loss.sbr_coeff.item())

                    ep_metrics_df = pd.DataFrame({'d': episodes_performed_actions,
                                                  'r': episodes_rewards})

                    step_metrics_df = pd.DataFrame({'l': losses})

                    if i_episode % update_plots_frequency == 0:
                        clear_output(wait=True)
                        axs[0, 0].clear()
                        lineplot(axs[0, 0],
                                 step_metrics_df['l'].rolling(window=avg_roll_window).mean(),
                                 f'Loss average smoothed with windows of {avg_roll_window} episodes',
                                 'Optimization steps',
                                 'Value')
                        axs[0, 1].clear()
                        lineplot(axs[0, 1], eps, 'Epsilon', 'Episodes', 'Eps')
                        if plot_performed_actions:
                            axs[1, 0].clear()
                            lineplot(axs[1, 0],
                                     ep_metrics_df['d'].rolling(window=avg_roll_window).mean(),
                                     f'Average of the number of performed actions for episode, smoothed with windows '
                                     f'of {avg_roll_window} episodes',
                                     'Episodes',
                                     'Number of performed actions')
                        if plot_mean_reward:
                            axs[1, 1].clear()
                            lineplot(axs[1, 1],
                                     episodes_rewards,
                                     'Episode cumulative rewards',
                                     f'Episodes (Victories: {len(episodes_victories)} / {i_episode + 1})',
                                     'Rewards',
                                     points=[episodes_victories, episodes_losts],
                                     points_style=[{'c': 'green', 'marker': '*'}, {'c': 'red', 'marker': '.'}],
                                     hline=0.0)
                        if plot_actions_count:
                            axs[2, 0].clear()
                            countplot(axs[2, 0], list(action_counts.values()), list(action_counts.keys()),
                                      'Actions taken')

                        axs[2, 1].clear()
                        axs[2, 1].title.set_text(f'Reward average smoothed with windows of {avg_roll_window} episodes')
                        axs[2, 1].plot(ep_metrics_df['r'].rolling(window=avg_roll_window).mean())

                        # Pause a bit so that plots are updated
                        if self.notebook:
                            display.clear_output(wait=True)
                            display.display(pl.gcf())
                        else:
                            display.clear_output(wait=True)
                            display.display(plt.gcf())
                            plt.pause(0.001)

                    break

            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if save_path and (i_episode + 1) % save_frequency == 0:
                save_filename = save_path + f'weights_{i_episode + 1}.pt'
                print(f'Model saved at {save_filename}')
                torch.save(self.target_net.state_dict(), save_filename)

        self.env.set_first(self.is_initially_first)
        # Avoid displaying a duplicate of the last plot
        clear_output(wait=True)
        print('Training complete')

        return Statistics(episodes_rewards,
                          episodes_victories,
                          episodes_losts,
                          episodes_performed_actions,
                          action_counts,
                          losses,
                          dual_losses,
                          sbr_coeffs,
                          eps)

    def select_action(self,
                      screen: torch.Tensor,
                      state: np.array,
                      eps: List[float]) -> Tuple[torch.Tensor, bool]:
        """
        Apply the policy on the observed state to decide the next action or explore by choosing a random action.

        :param screen: the current screen
        :param state: the board
        :param eps: the list containing the computed eps (metrics)
        :return: the chosen action and a boolean representing if the action has been chosen by a LOGIC_PURE constraint
        """
        sample = random.random()
        # Push the agent to exploit the policy after many steps rather than explore
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)

        eps.append(eps_threshold)
        self.steps_done += 1

        if sample > eps_threshold:
            # Exploitation
            with torch.no_grad():
                action = None
                # If True the action is from LOGIC_PURE
                logic_pure_action = False

                # Compute constraints if necessary
                if self.constraints is not None:
                    constrained_actions = self.constraints.select_constrained_action(
                        state.squeeze(),
                        self.env.first
                    ).to(self.device)
                else:
                    constrained_actions = None

                # LOGIC_PURE and LOGIC_TRAIN constraints and constraints include only single available actions
                if constrained_actions is not None and \
                        self.constraints.c_type in [ConstraintType.LOGIC_PURE, ConstraintType.LOGIC_TRAIN] and \
                        constrained_actions.sum().item() == 1:
                    action = constrained_actions.max(0)[1].view(1, 1)
                    logic_pure_action = self.constraints.c_type is ConstraintType.LOGIC_PURE
                # SPE and CDQN constraints
                elif constrained_actions is not None and self.constraints.c_type in [ConstraintType.SPE,
                                                                                     ConstraintType.CDQN]:
                    action = self.policy_net(screen,
                                             torch.from_numpy(state).squeeze().unsqueeze(0)).squeeze()
                    action = torch.where(constrained_actions == 0,
                                         torch.full(action.shape, -np.inf).to(self.device),
                                         action)
                    action = action.max(0)[1].view(1, 1)

                # If there are no constraints or the action is still None due to LOGIC_PURE and LOGIC_TRAIN condition
                # use the network
                if constrained_actions is None or action is None:
                    action = self.policy_net(screen,
                                             torch.from_numpy(state).squeeze().unsqueeze(0)).max(1)[1].view(1, 1)

                return action, logic_pure_action
        else:
            # Exploration
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long), False
