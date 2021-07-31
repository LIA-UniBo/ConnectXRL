from typing import Callable, Union, List

import torch
from torch import Tensor

from src.connectx.environment import ConnectXGymEnv, convert_state_to_image
from src.connectx.evaluate import record_matches, show_recordings
from src.connectx.opponents import interactive_player
from src.connectx.policy import CNNPolicy, Policy
from src.xrl.shapley import explain


def create_recordings(agent: Policy,
                      opponent: Union[str, Callable],
                      play_as_first_player: bool,
                      matches: int) -> List[Tensor]:

    """

    :param agent: the trained policy
    :param opponent: the opponent
    :param play_as_first_player: if True it plays as first
    :param matches: number of matches to record
    :return: the recorded screen states
    """

    if not play_as_first_player and opponent is interactive_player:
        print('Creation of environment ... Press any button')
    env = ConnectXGymEnv(opponent, play_as_first_player)

    print(f'Recording {matches} matches ...')
    states_recording, actions_recording = record_matches(env,
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

    return states_recording


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

    # Create recordings for training
    train_matches = 1000

    training_combinations = [
        {'opponent': 'random',
         'first': True,
         'matches': train_matches},

        {'opponent': 'random',
         'first': False,
         'matches': train_matches},

        {'opponent': 'negamax',
         'first': True,
         'matches': train_matches},

        {'opponent': 'negamax',
         'first': False,
         'matches': train_matches}
    ]

    states_recording = []

    for i, tc in enumerate(training_combinations):
        print(f'Combination {i + 1}/{len(training_combinations)}\n{tc}')
        states_recording.append(create_recordings(agent,
                                                  tc['opponent'],
                                                  tc['first'],
                                                  tc['matches']))
    # Flatten
    states_recording = [s_tensor for sr_sublist in states_recording for s_tensor in sr_sublist]

    # Train explainer
    explainer = explain(agent,
                        states_training=states_recording,
                        states_test=None)

    opponent = 'random'
    play_as_first_player = True

    test_states_recording = create_recordings(agent,
                                              opponent,
                                              play_as_first_player,
                                              1)
    _ = explain(agent,
                states_training=None,
                states_test=test_states_recording,
                explainer=explainer)


if __name__ == "__main__":
    main()
