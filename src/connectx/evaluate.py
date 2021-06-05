import random
from typing import Union, Callable, Optional, Dict, List, Tuple

import numpy as np
import torch
from kaggle_environments import evaluate, Environment

from matplotlib import pyplot as plt
from tqdm import tqdm
from src.connectx.environment import convert_state_to_image, ConnectXGymEnv
from src.connectx.opponents import interactive_player
from src.connectx.policy import Policy

WIDTH = 256
HEIGHT = 256


def fix_random(seed: int) -> None:
    """
    Fix all the possible sources of randomness.

    :param seed: the seed to use.
    """

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def play(env: Environment,
         player1: Union[str, Callable],
         player2: Union[str, Callable],
         width: Optional[int] = WIDTH,
         height: Optional[int] = HEIGHT) -> None:
    """
    Render the game between two agents.

    :param env: Kaggle environment
    :param player1: one of the predefined agents ("random", "negamax") or a custom one (Callable)
    :param player2: one of the predefined agents ("random", "negamax") or a custom one (Callable)
    :param width: board rendering width
    :param height: board rendering width
    """

    env.run([player1, player2])
    env.render(mode='ipython', width=width, height=height)


def interactive_play(env: Environment,
                     player1: Optional[Union[str, Callable]] = None,
                     player2: Optional[Union[str, Callable]] = None,
                     width: Optional[int] = WIDTH,
                     height: Optional[int] = HEIGHT) -> None:
    """
    Interactively play with the computer. The interactive player MUST be passed as None.

    :param env: Kaggle environment
    :param player1: one of the predefined agents ("random", "negamax") or a custom one (Callable)
    :param player2: one of the predefined agents ("random", "negamax") or a custom one (Callable)
    :param width: board rendering width
    :param height: board rendering width
    """

    if player1 is not None and player2 is not None:
        raise ValueError('One of the player must be None to be assigned to the interactive player.')

    env.play([player1, player2], width=width, height=height)


def evaluate_matches(matches: List[Dict[str, int]]):
    """
    Evaluate multiple matches.

    :param matches: List containing dictionaries specifying the matches. Each match defines the 2 players 'ply1' and
    'ply2' and how many times the game is played. A mean of the rewards of each play is displayed for each match.
    """

    for m in matches:
        print(f' {m["ply1"]} vs {m["ply1"]} ({m["eps"]} eps):', np.mean(evaluate('connectx',
                                                                                 [m['ply1'], m['ply2']],
                                                                                 num_episodes=m['eps'])))


def get_win_percentages(player: Union[str, Callable],
                        opponents: Dict[str, Union[str, Callable]],
                        player_name: Optional[str] = None,
                        config: Optional[Dict] = None,
                        n_rounds_as_1st_player: int = 100,
                        n_rounds_as_2nd_player: int = 100):
    """
    Print win/loss percentages between the player and the list of opponents.

    :param player: Your player
    :param player_name: Your player's name
    :param opponents: List of opponents
    :param config: dict specifying the board ('rows' and 'columns') and the stones to win ('inarow'). Default values are
    {'rows': 6, 'columns': 7, 'inarow': 4}
    :param n_rounds_as_1st_player: number of rounds where agent 1 play as the first player
    :param n_rounds_as_2nd_player: number of rounds where agent 1 play as the second player
    """

    if config is None:
        config = {'rows': 6, 'columns': 7, 'inarow': 4}

    if player_name is None:
        player_name = player.__name__ if callable(player) else player

    for opponent_name, opponent in opponents.items():
        print(f'{player_name} VS {opponent_name}')

        outcomes = evaluate('connectx', [player, opponent], config, [], n_rounds_as_1st_player)
        outcomes += [[b, a] for [a, b] in evaluate('connectx', [opponent, player], config, [], n_rounds_as_2nd_player)]

        # print(outcomes)
        print()
        print(f'{player_name} win percentage: {np.round(outcomes.count([1, -1]) / len(outcomes), 2)} - '
              f'{outcomes.count([1, -1])}/{len(outcomes)}')
        print(f'{opponent_name} win percentage: {np.round(outcomes.count([-1, 1]) / len(outcomes), 2)} - '
              f'{outcomes.count([-1, 1])}/{len(outcomes)}')
        print(f'Draw percentage: {np.round(outcomes.count([0, 0]) / len(outcomes), 2)} - '
              f'{outcomes.count([0, 0])}/{len(outcomes)}')
        print(f'Number of invalid moves by {player_name}: {outcomes.count([None, 0])}')
        print(f'Number of invalid moves by {opponent_name}: {outcomes.count([0, None])}\n')


def record_matches(env: ConnectXGymEnv,
                   policy: Policy,
                   configuration: dict = None,
                   play_as_first_player: bool = True,
                   num_matches: int = 1,
                   render_env: bool = True,
                   keep_player_colour: bool = True,
                   device: str = 'cpu') -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Record screens and actions performed.

    :param env: the gym environment defining the board size and opponent, used only for testing here
    :param policy: the policy of the agent as a nn.Module implementing predict method
    :param configuration: game and agent config, default is {'columns': 7, 'rows': 6, 'inarow': 4, 'c_type': None}
    :param play_as_first_player: if True the agent is the first player
    :param num_matches: the number of matches
    :param render_env: if True the environment is rendered
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
        if not play_as_first_player and env.opponent is interactive_player:
            print('\n - Game is on ... Press any button -')
        env.set_first(play_as_first_player)
        # Initialize the environment and state
        if not play_as_first_player and env.opponent is interactive_player:
            print('\n - Table is empty, make your first move! -')
        state = env.reset()
        # Get image and convert to torch tensor
        screen = torch.from_numpy(convert_state_to_image(state=state,
                                                         first_player=play_as_first_player,
                                                         keep_player_colour=keep_player_colour)).to(device)
        done = False
        if render_env:
            env.render(board=np.array(state).reshape((env.rows, env.columns, 1)),
                       mode='rgb_image',
                       render_waiting_time=1,
                       keep_player_colour=keep_player_colour)
        while not done:

            # Get action
            observation = {'board': list(state.ravel()), 'mark': 1 if env.first else 2}
            action = policy.predict(observation,
                                    configuration)

            # Render the board as updated by the agent
            if render_env:
                # Create the board to render
                render_state = state.copy()
                ir = 0
                for r in render_state:
                    # Spot column to update
                    if r[action] != [0]:
                        break
                    ir += 1
                if ir - 1 >= 0:
                    render_state[ir - 1, action] = [1] if play_as_first_player else [2]
                env.render(board=render_state,
                           mode='rgb_image',
                           render_waiting_time=1,
                           keep_player_colour=keep_player_colour)

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

    if render_env and play_as_first_player:
        # Create the board to render
        render_state = state.copy()
        ir = 0
        for r in render_state:
            # Spot column to update
            if r[action] != [0]:
                break
            ir += 1
        if ir - 1 >= 0:
            render_state[ir - 1, action] = [1] if play_as_first_player else [2]
        env.render(board=render_state,
                   mode='rgb_image',
                   render_waiting_time=1,
                   keep_player_colour=keep_player_colour)
    input('\n - Game finished! -')
    return state_recording, action_recording


def show_recordings(state_recording: torch.Tensor,
                    action_recording: torch.Tensor,
                    render_waiting_time: float = 1) -> None:
    """
    :param state_recording: the states
    :param action_recording: the associated actions
    :param render_waiting_time: in seconds
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
