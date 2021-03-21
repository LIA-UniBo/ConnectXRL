from itertools import count

import matplotlib.pyplot as plt

import torch

from src.connectx.policy import CNNPolicy
from src.connectx.environment import convert_state_to_image, ConnectXGymEnv


def show_saliency_map(env: ConnectXGymEnv,
                      policy: torch.nn.Module,
                      num_episodes: int = 10,
                      render_waiting_time: float = 0,
                      device: str = 'cpu'):
    """

    :param env: Gym environment
    :param policy: policy (network)
    :param num_episodes: how many episodes
    :param render_waiting_time: if 0 or None you can skip frames manually otherwise frames are displayed automatically
    :param device: the device used to store tensors
    """

    policy.eval()
    fig, ax = plt.subplots(1, 2)
    plt.grid()

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        state = env.reset()
        # Get image and convert to torch tensor
        screen = torch.from_numpy(convert_state_to_image(state)).to(device)

        # Rendering
        im_g = ax[0].imshow(screen.squeeze().permute(1, 2, 0))
        im_s = ax[1].imshow(torch.ones((screen.shape[2], screen.shape[3])).data.numpy(), cmap='gray', vmin=0, vmax=1)

        for t in count():

            # Prepare input and reset gradients
            screen.requires_grad_()

            # Select and perform an action on the environment
            action = policy(screen)
            i = action.argmax().view(1, 1).item()

            # Compute gradients
            action[0, i].backward()

            # Extract saliency map
            print(torch.abs(screen.grad))
            print(torch.ones((screen.shape[2], screen.shape[3])).data.numpy())

            # Rendering
            im_g.set_data(screen.squeeze().permute(1, 2, 0).data.numpy())
            im_s.set_data(torch.abs(screen.grad).max(1)[0].squeeze().data.numpy())
            fig.canvas.draw_idle()
            if render_waiting_time:
                plt.pause(render_waiting_time)
            else:
                input(f"{t}/{i_episode} Press Enter to continue...")
                plt.pause(0.000001)

            action = action.max(1)[1].view(1, 1)
            # Continue the game
            new_state, _, done, _ = env.step(action.item())
            new_screen = torch.from_numpy(convert_state_to_image(new_state)).to(device)

            screen = new_screen

            if done:
                break

    print('Analysis complete')


env = ConnectXGymEnv('random', True)
init_screen = convert_state_to_image(env.reset())
screen_shape = (init_screen.shape[1], init_screen.shape[2], init_screen.shape[3])

agent = CNNPolicy(env.action_space.n,
                  screen_shape)

device = 'cpu'
weight_path = 'TODO'
agent.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))

show_saliency_map(env, agent, 30, device=device)
