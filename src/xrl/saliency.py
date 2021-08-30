from itertools import count
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from lime.wrappers.scikit_image import SegmentationAlgorithm
from torchvision.transforms import Resize

from src.connectx.environment import convert_state_to_image, ConnectXGymEnv, show_board_grid, VMIN, VMAX
from src.connectx.opponents import interactive_player
from src.connectx.policy import CNNPolicy

from lime import lime_image


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

    return action, saliency_map.squeeze()


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
    saliency_map = torch.abs(screen.grad).max(1)[0].squeeze()
    # print("max_grad:", torch.max(saliency_map))
    # print("min_grad:", torch.min(saliency_map))

    return action, (saliency_map - torch.min(saliency_map)) / (torch.max(saliency_map) - torch.min(saliency_map)).data.numpy()


class LIME_wrapper(nn.Module):
    """
    A wrapper fot a deep-based policy to perform LIME explanation.
    Step 1: Generate random perturbations for input image
    Step 2: Predict class for perturbations
    Step 3: Compute weights (importance) for the perturbations
    Step 4: Fit a explainable linear model using the perturbations, predictions and weights
    """

    def __init__(self, policy_network: CNNPolicy):
        """

        :param policy_network: a network to model an agent policy. Must contain a feature_extractor and a fc_head. The
        feature extractor is expected to be a cnn.
        """
        super(LIME_wrapper, self).__init__()
        self.explainer = lime_image.LimeImageExplainer()

        # The segmentation algorithm is used to identify superpixels
        # 'quickshift': Given an image, the algorithm calculates a forest of pixels whose branches are labeled with a
        # distance value. This specifies a hierarchical segmentation of the image with segments corresponding to
        # subtrees. Useful superpixels can be identified by cutting the branches whose distance label is above a given
        # threshold.
        # 'slic': This algorithm generates superpixels by clustering pixels based on their color similarity and
        # proximity in the image plane.
        # 'felzenszwalb': Edges are considered in increasing order of weight; their endpoint pixels are merged into a
        # region if this doesn't cause a cycle in the graph and if the pixels are similar to the existing regions pixels

        # compactness balances color proximity and space proximity. Higher values give more weight to space proximity,
        # making superpixel shapes more square/cubic.
        self.segmentation_fn = SegmentationAlgorithm('slic',
                                                     start_label=1,
                                                     compactness=0.001,
                                                     min_size_factor=0,
                                                     n_segments=20,
                                                     random_seed=42)
        self.net = policy_network
        self.net.eval()

    def forward(self, x: np.array) -> np.array:
        """
        For LIME np.array are passed and returned compared to the classic forward because the library method
        requires this format.

        :param x: the game board screen image
        :return: The probability of actions
        """
        x = torch.tensor(x, dtype=torch.float32)
        # If only 3 dims the batch is created adding one
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        x = torch.reshape(x, (x.shape[0], x.shape[3], x.shape[1], x.shape[2]))
        x = F.softmax(self.net(x), dim=1).type(torch.float64).detach().numpy()

        return x

    def lime_explanation(self,
                         screen: np.array) -> Tuple[torch.Tensor, np.array]:
        """
        https://arxiv.org/pdf/1602.04938v1.pdf

        :param screen: the game board screen image
        :return: the action logits from the network and the mask
        """
        action = self.net(screen)
        img = np.transpose(np.squeeze(np.array(screen), axis=0), (1, 2, 0)).astype('float64')

        # hide_color is set to 1 in order to perturb images with white pixels
        explanation = self.explainer.explain_instance(img,
                                                      self.forward,
                                                      batch_size=1,
                                                      hide_color=1,
                                                      segmentation_fn=self.segmentation_fn,
                                                      top_labels=7,
                                                      num_samples=100)

        # num_features is the number of superpixels to include in explanation. It is set to 3 because
        # the choice of an action depends on patterns of 3 pixels (in order to win or in order to avoid the victory
        # of the opponent)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                    positive_only=False,
                                                    num_features=3,
                                                    hide_rest=False)

        return action, mask


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
            sm = Resize(size=(screen.shape[2], screen.shape[3]))(sm.unsqueeze(dim=0)).squeeze()
        elif saliency_type == 'lime':
            a, sm = policy.lime_explanation(screen)
        else:
            raise ValueError(f'Unknown saliency_type: {saliency_type}!')
        return a, sm

    # Network setup
    if saliency_type == 'cam':
        policy = CAM_wrapper(policy)
    elif saliency_type == 'lime':
        policy = LIME_wrapper(policy)
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
                # Get the shape of the saliency map
                _, saliency_map = extract_saliency_map()
                # Compute action separately because is needed
                action = policy(screen)
                # Put saliency to 0 because is not used
                saliency_map.fill_(0)

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
