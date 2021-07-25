import torch

from src.connectx.environment import ConnectXGymEnv, convert_state_to_image
from src.connectx.opponents import interactive_player
from src.connectx.policy import CNNPolicy
from src.xrl.saliency import show_saliency_map


def main():
    # Environment creation
    play_as_first_player = True
    opponent = interactive_player
    if not play_as_first_player and opponent is interactive_player:
        print('Creation of environment ... Press any button')
    env = ConnectXGymEnv(opponent, play_as_first_player)

    # Initialize environment
    if not env.first and env.opponent is interactive_player:
        print('Initialization of environment ... Press any button')
    if env.first and env.opponent is interactive_player:
        print('\n - Game is ready make your first move! -')
    init_screen = convert_state_to_image(env.reset())

    # Initialize agent
    screen_shape = (init_screen.shape[1], init_screen.shape[2], init_screen.shape[3])
    agent = CNNPolicy(env.action_space.n,
                      screen_shape)
    device = 'cpu'
    weight_path = './models/curriculum.pt'
    agent.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))

    # Go
    if opponent is interactive_player:
        render_waiting_time = 0.001
    else:
        render_waiting_time = 2

    show_saliency_map(env,
                      agent,
                      'lime',
                      see_saliency_on_input=False,
                      num_episodes=30,
                      render_waiting_time=render_waiting_time,
                      device=device)


if __name__ == "__main__":
    main()
