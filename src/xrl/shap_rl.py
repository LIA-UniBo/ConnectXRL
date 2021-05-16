from typing import Tuple, List

import numpy as np
import shap
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from matplotlib import cm
from src.connectx.environment import convert_state_to_image, ConnectXGymEnv
from src.connectx.policy import Policy


def interactive_player(observation: dict,
                       configuration: dict):
    """
    TODO: fix delayed images

    :param observation: turn's data (board as a list, mark as 1 or 2)
    :param configuration: environment's data (num of columns, num of rows)
    :return: action performed
    """
    return int(input(f'{np.array(observation["board"]).reshape((configuration["rows"], configuration["columns"]))}\n'
                     f'Decide an action ({0}-{configuration["columns"]}): '))


def record_matches(env: ConnectXGymEnv,
                   policy: Policy,
                   configuration: dict = None,
                   play_as_first_player: bool = True,
                   num_matches: int = 1,
                   keep_player_colour: bool = True,
                   device: str = 'cpu') -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Record screens and actions performed,

    :param env: the gym environment defining the board size and opponent, used only for testing here
    :param policy: the policy of the agent as a nn.Module implementing predict method
    :param configuration: game and agent config, default is {'columns': 7, 'rows': 6, 'inarow': 4, 'c_type': None}
    :param play_as_first_player: if True the agent is the first player
    :param num_matches: the number of matches
    :param keep_player_colour: if True the agent color is maintained between player 1 and player 2. e.g. Your agent
    will always be the red player, otherwise the 1st player will always be the red one
    :param device: the device where the recording occurs, 'cpu', 'gpu' ...
    :return: two list of length num_matches where the first contains the states observed and the second the associate
    actions performed
    """

    if configuration is None:
        configuration = {'columns': 7, 'rows': 6, 'inarow': 4, 'c_type': None}

    # Results
    action_recording = [torch.Tensor([]) for _ in range(num_matches)]
    state_recording = [torch.Tensor([]) for _ in range(num_matches)]

    for m in tqdm(range(num_matches)):

        # Change first or second player
        env.set_first(play_as_first_player)
        # Initialize the environment and state
        state = env.reset()
        # Get image and convert to torch tensor
        screen = torch.from_numpy(convert_state_to_image(state=state,
                                                         first_player=play_as_first_player,
                                                         keep_player_colour=keep_player_colour)).to(device)
        done = False
        while not done:

            # Get action probs
            observation = {'board': list(state.ravel()), 'mark': 1 if env.first else 2}
            action = policy.predict(observation,
                                    configuration)
            # Get action
            action = action.max(1)[1].item()
            # Update env
            state, _, done, _ = env.step(action)

            # Update results
            action_recording[m] = torch.cat((action_recording[m], torch.Tensor([action])))
            if len(state_recording[m]) == 0:
                state_recording[m] = screen
            else:
                state_recording[m] = torch.cat((state_recording[m], screen), dim=0)

            # Update screen for next iteration
            screen = torch.from_numpy(convert_state_to_image(state=state,
                                                             first_player=play_as_first_player,
                                                             keep_player_colour=keep_player_colour)).to(device)

    return state_recording, action_recording


def show_recordings(state_recording: torch.Tensor,
                    action_recording: torch.Tensor,
                    render_waiting_time: float = 1):
    """
    :param state_recording: the states
    :param action_recording: the associated actions
    :param render_waiting_time: in seconds
    :return:
    """

    image = None
    for i, im in enumerate(state_recording):
        im = im.permute(1, 2, 0)
        if image is None:
            image = plt.imshow(im)
        else:
            image.set_data(im)
        plt.title(f'Frame {i} action performed {action_recording[i]}')
        plt.pause(render_waiting_time)
        plt.draw()


def explain(policy: Policy,
            state_recording: list,
            action_recording: list) -> List[torch.Tensor]:
    """

    :param policy: the policy involved
    :param state_recording: a list with tensors containing the states seen during each match
    :param action_recording: TODO
    :return: the shapley values associated to each action for each state seen during the matches
    """

    # If the input is a list, the states are concatenated
    if type(state_recording) == type(action_recording) == list:
        state_recording = torch.cat(state_recording)

    # Create explainer
    explainer = shap.DeepExplainer(policy, state_recording)
    shap_values = explainer.shap_values(state_recording)

    # A list with the size of possible actions containing as Numpy arrays the shapley values for each state. Here the
    # Numpy array is transformed to torch.
    shap_values = list(map(lambda x: torch.from_numpy(x), shap_values))

    return shap_values
