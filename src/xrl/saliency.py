from itertools import count
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from src.connectx.environment import convert_state_to_image, ConnectXGymEnv, show_board_grid, VMIN, VMAX
from src.connectx.opponents import interactive_player
from src.connectx.policy import CNNPolicy


class CAM_wrapper(nn.Module):
    """
    A wrapper fot a deep-based policy to record gradients used by the CAM-Grad technique.
    """

    def __init__(self, policy_network: CNNPolicy):
        """

        :param policy_network: a network to model an agent policy. Must contain a feature_extractor and a fc_head. The
        feature extractor is expected to be a cnn.
        """
        super(CAM_wrapper, self).__init__()
        self.net = policy_network
        self.net.eval()

        # Gradients saved
        self.gradients = None

    def activations_hook(self, grad: torch.Tensor):
        """
        Used when register_hook is called to hook the gradients of the activations.

        :param grad: the gradients
        """
        self.gradients = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward and hook registration.

        :param x: input image
        :return: logits for each action
        """
        # If only 3 dims the batch is created adding one
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        # Extract features
        x = self.net.feature_extractor(x)
        # Register the hook
        _ = x.register_hook(self.activations_hook)
        # Flatten and pass them to fc heads
        return self.net.fc_head(x.view(x.size(0), -1))

    def get_activations_gradient(self) -> torch.Tensor:
        """

        :return: the gradients hooked at the end of the feature extractor
        """
        return self.gradients

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: the input image
        :return: the activation on the last layer of the feature extraction (e.g. cnn)
        """
        return self.net.feature_extractor(x)


def cam_saliency_map(screen: np.array,
                     policy_network: CNNPolicy) -> Tuple[torch.Tensor, np.array]:
    """
    Grad-CAM saliency map.
    https://arxiv.org/pdf/1610.02391.pdf

    :param screen: the game board screen image
    :param policy_network: the network representing the policy
    :return: the action logits from the network and the saliency map
    """

    # Select and perform an action on the environment
    action = policy_network(screen)
    i = action.argmax().view(1, 1).item()

    # Compute gradients
    action[0, i].backward()

    # Get the gradients from the model
    gradients = policy_network.get_activations_gradient()

    # Globally pool the gradients obtaining a value for each channel
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # Get the features from the policy
    features = policy_network.get_features(screen).detach()

    # Weight each feature map "pixel" by the gradient for the chosen action
    for i in range(gradients.shape[1]):
        features[:, i, :, :] *= pooled_gradients[i]

    # Average the features
    saliency_map = torch.mean(features, dim=1).squeeze()

    # Apply ReLU
    saliency_map = np.maximum(saliency_map, 0)

    # Normalize the heatmap avoiding to divide by zero
    if torch.sum(saliency_map) != 0:
        saliency_map /= torch.max(saliency_map)

    """
    # Draw the saliency map
    if torch.sum(gradients) != 0 or torch.sum(saliency_map) != 0:
        plt.matshow(saliency_map.squeeze())
    """

    return action, saliency_map.squeeze().data.numpy()


def vanilla_saliency_map(screen: np.array,
                         policy_network: CNNPolicy) -> Tuple[torch.Tensor, np.array]:
    """
    Vanilla saliency map based on gradients absolute values.
    https://arxiv.org/abs/1312.6034

    :param screen: the game board screen image
    :param policy_network: the network representing the policy
    :return: the action logits from the network and the saliency map
    """

    # Prepare input and reset gradients
    screen.requires_grad_()

    # Select and perform an action on the environment
    action = policy_network(screen)
    i = action.argmax().view(1, 1).item()

    # Compute gradients
    action[0, i].backward()

    return action, torch.abs(screen.grad).max(1)[0].squeeze().data.numpy()


def show_saliency_map(env: ConnectXGymEnv,
                      policy: CNNPolicy,
                      saliency_type: str = 'vanilla',
                      see_saliency_on_input: bool = True,
                      num_episodes: int = 10,
                      render_waiting_time: float = 0.001,
                      device: str = 'cpu',
                      above: bool = True) -> None:
    """

    :param env: Gym environment
    :param policy: policy (network)
    :param saliency_type: type of saliency ('vanilla', 'cam')
    :param see_saliency_on_input: if True the saliency map is applied to the input that the policy actually see,
    otherwise it is applied to the resulting image resulting from the choice of the action (relative to the saliency
    map) from the policy. If True the saliency is removed when the enemy is playing, to get the best from this option
    consider a longer render_waiting_time.
    :param num_episodes: how many episodes
    :param render_waiting_time: seconds of pause at each rendering
    :param device: the device used to store tensors
    :param above: if True the saliency map is applied over the board
    """

    def update_rendering(s, sm):
        # Update rendering
        im_g.set_data(s.squeeze().permute(1, 2, 0).data.numpy())
        im_s.set_data(sm)
        fig.canvas.draw_idle()
        plt.pause(render_waiting_time)

    def extract_saliency_map():
        # Extract saliency map in the correct way
        if saliency_type == 'vanilla':
            a, sm = vanilla_saliency_map(screen, policy)
        elif saliency_type == 'cam':
            a, sm = cam_saliency_map(screen, policy)
        else:
            raise ValueError(f'Unknown saliency_type: {saliency_type}!')
        return a, sm

    # Network setup
    if saliency_type == 'cam':
        policy = CAM_wrapper(policy)
    policy.eval()

    # Rendering setup
    fig, ax = plt.subplots(1, 1 if above else 2)

    if above:
        axes_g = axes_s = ax
    else:
        axes_g = ax[0]
        axes_s = ax[1]
        show_board_grid(axes_s, env.rows, env.columns)

    show_board_grid(axes_g, env.rows, env.columns)

    # Loop
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        if not env.first and env.opponent is interactive_player:
            print('\n - Table is empty, make your first move! -')
        state = env.reset()
        # Get image and convert to torch tensor
        screen = torch.from_numpy(convert_state_to_image(state)).to(device)

        # Rendering
        im_g = axes_g.imshow(screen.squeeze().permute(1, 2, 0),
                             interpolation='none',
                             vmin=VMIN,
                             vmax=VMAX,
                             aspect='equal')
        im_s = axes_s.imshow(torch.ones((screen.shape[2], screen.shape[3])).data.numpy(),
                             interpolation='none',
                             cmap='binary' if above else 'Greens',
                             vmin=0,
                             vmax=1,
                             alpha=0.75 if above else 1,
                             aspect='equal')

        # Add colorbar only once
        if i_episode == 0:
            cax = fig.add_axes([axes_s.get_position().x1 + 0.01,
                                axes_s.get_position().y0, 0.02,
                                axes_s.get_position().height])

            plt.colorbar(im_s, cax=cax)

        for _ in count():
            # See saliency on the input state
            if not see_saliency_on_input:
                action, saliency_map = extract_saliency_map()
            else:
                _, saliency_map = extract_saliency_map()
                action = policy(screen)
                saliency_map.fill(0)

            # Render the board as updated by the agent
            # Create the board to render
            render_state = state.copy()
            ir = 0
            for r in render_state:
                # Spot column to update
                if r[action.max(1)[1].view(1, 1)] != [0]:
                    break
                ir += 1
            if ir - 1 >= 0:
                render_state[ir - 1, action.max(1)[1].view(1, 1)] = [1] if env.first else [2]

            # Render the board after the policy has played
            update_rendering(torch.from_numpy(convert_state_to_image(render_state)).to(device),
                             saliency_map)

            action = action.max(1)[1].view(1, 1)
            # Continue the game
            new_state, _, done, _ = env.step(action.item())
            new_screen = torch.from_numpy(convert_state_to_image(new_state)).to(device)

            # Update screen and state
            screen = new_screen
            state = new_state

            # See saliency together with the output state
            if see_saliency_on_input:
                action, saliency_map = extract_saliency_map()

            # Render the board after the opponent has played
            update_rendering(screen,
                             saliency_map)

            if done:
                break

    print('Analysis complete')
