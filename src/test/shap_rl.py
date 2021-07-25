import torch

from src.connectx.environment import ConnectXGymEnv, convert_state_to_image
from src.connectx.evaluate import record_matches, show_recordings
from src.connectx.opponents import interactive_player
from src.connectx.policy import CNNPolicy
from src.xrl.shap_rl import explain


def main():
    opponent = 'random'
    play_as_first_player = True

    if not play_as_first_player and opponent is interactive_player:
        print('Creation of environment ... Press any button')
    env = ConnectXGymEnv(opponent, play_as_first_player)

    if not play_as_first_player and opponent is interactive_player:
        print('Initialize policy ... Press any button')

    init_screen = convert_state_to_image(env.reset())
    screen_shape = (init_screen.shape[1], init_screen.shape[2], init_screen.shape[3])

    agent = CNNPolicy(env.action_space.n,
                      screen_shape)

    device = 'cpu'
    weight_path = './models/curriculum.pt'
    agent.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))

    state_recording, action_recording = record_matches(env,
                                                       agent,
                                                       play_as_first_player=play_as_first_player,
                                                       num_matches=20,
                                                       render_env=False,
                                                       interactive_progress=False,
                                                       keep_player_colour=True)

    print(state_recording, action_recording)

    shap_values = explain(agent, state_recording[:-1], state_recording[-1:])

    """
    for i, (sr, ar) in enumerate(zip(state_recording, action_recording)):
        print(f'Play recording {i + 1}')
        show_recordings(sr, ar)
    """


if __name__ == "__main__":
    main()
