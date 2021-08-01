from typing import Callable, Union, List, Tuple

import torch
from torch import Tensor

from src.connectx.environment import ConnectXGymEnv, convert_state_to_image
from src.connectx.evaluate import record_matches, show_recordings
from src.connectx.opponents import interactive_player, random_player, always_one_position_player
from src.connectx.policy import CNNPolicy, Policy
from src.xrl.shapley import explain


def create_recordings(agent: Union[Policy, Callable],
                      opponent: Union[str, Callable],
                      play_as_first_player: bool,
                      matches: int) -> Tuple[List[Tensor], List[str]]:

    """

    :param agent: the trained policy or a generic Callable
    :param opponent: the opponent
    :param play_as_first_player: if True it plays as first
    :param matches: number of matches to record
    :return: the recorded screen states and the final status
    """

    if not play_as_first_player and opponent is interactive_player:
        print('Creation of environment ... Press any button')
    env = ConnectXGymEnv(opponent, play_as_first_player)

    print(f'Recording {matches} matches ...')
    states_recording, actions_recording, results_recording = record_matches(env,
                                                                            agent,
                                                                            play_as_first_player=play_as_first_player,
                                                                            num_matches=matches,
                                                                            render_env=False,
                                                                            interactive_progress=False,
                                                                            keep_player_colour=True)

    """
    for i, (sr, ar) in enumerate(zip(state_recording, action_recording)):
        print(f'Play recording {i + 1}')
        show_recordings(sr, ar)
    """

    return states_recording, results_recording


def main():
    # Create the agent policy
    env = ConnectXGymEnv('random', True)

    init_screen = convert_state_to_image(env.reset())
    screen_shape = (init_screen.shape[1], init_screen.shape[2], init_screen.shape[3])

    agent = CNNPolicy(env.action_space.n,
                      screen_shape)

    device = 'cpu'
    weight_path = './models/curriculum.pt'
    agent.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))

    background_images = []

    # Create recordings for background images
    if input('Use recording as background images for the DeepExplainer (y/n)') == 'y':
        recordings_combinations = [
            {'agent': random_player,
             'opponent': 'random',
             'first': True,
             'matches': 750},

            {'agent': random_player,
             'opponent': 'random',
             'first': False,
             'matches': 750},

            {'agent': always_one_position_player,
             'opponent': 'random',
             'first': True,
             'matches': 75},

            {'agent': always_one_position_player,
             'opponent': 'random',
             'first': False,
             'matches': 75},

            {'agent': agent,
             'opponent': 'negamax',
             'first': True,
             'matches': 75},

            {'agent': agent,
             'opponent': 'negamax',
             'first': False,
             'matches': 75}
        ]

        for i, tc in enumerate(recordings_combinations):
            print(f'Combination {i + 1}/{len(recordings_combinations)}\n{tc}')
            background_images.append(create_recordings(tc['agent'],
                                                       tc['opponent'],
                                                       tc['first'],
                                                       tc['matches'])[0])
        # Flatten
        background_images = [s_tensor for sr_sublist in background_images for s_tensor in sr_sublist]

    # White images
    empty_boards = int(input('Number of empty boards to add in the background dataset for the DeepExplainer'))
    background_images += [torch.clone(torch.from_numpy(init_screen)) for _ in range(empty_boards)]

    # Create explainer
    explainer = explain(agent,
                        background_images=background_images,
                        explain_images=None)

    is_continue = True
    while is_continue:
        opponent = int(input('1 to play against random\n2 to play against negamax\nany button to play against the '
                             'interactive player'))

        test_states_recording, test_results_recording = create_recordings(
            agent,
            'random' if opponent == 1 else ('negamax' if opponent == 2 else '3'),
            input('Play as first player (y/n)') == 'y',
            1
        )

        print(f'The match ended with {test_results_recording[0]} status')

        _ = explain(agent,
                    background_images=None,
                    explain_images=test_states_recording,
                    explainer=explainer)

        is_continue = input('Continue (y/n)').lower() == 'y'


if __name__ == "__main__":
    main()
