from itertools import count

import matplotlib.pyplot as plt

import torch

from src.connectx.environment import convert_state_to_image, ConnectXGymEnv, show_board_grid


def show_saliency_map(env: ConnectXGymEnv,
                      policy: torch.nn.Module,
                      num_episodes: int = 10,
                      render_waiting_time: float = 0,
                      device: str = 'cpu',
                      above: bool = False) -> None:
    """

    :param env: Gym environment
    :param policy: policy (network)
    :param num_episodes: how many episodes
    :param render_waiting_time: if 0 or None you can skip frames manually otherwise frames are displayed automatically
    :param device: the device used to store tensors
    :param above: if True the saliency map is applied over the board
    """

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

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        state = env.reset()
        # Get image and convert to torch tensor
        screen = torch.from_numpy(convert_state_to_image(state)).to(device)

        # Rendering
        im_g = axes_g.imshow(screen.squeeze().permute(1, 2, 0))
        im_s = axes_s.imshow(torch.ones((screen.shape[2], screen.shape[3])).data.numpy(),
                             cmap='binary' if above else 'Greens',
                             vmin=0,
                             vmax=1,
                             alpha=0.5 if above else 1)

        # Add colorbar only once
        if i_episode == 0:
            cax = fig.add_axes([axes_s.get_position().x1 + 0.01,
                                axes_s.get_position().y0, 0.02,
                                axes_s.get_position().height])

            plt.colorbar(im_s, cax=cax)

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

            # Update rendering
            im_g.set_data(screen.squeeze().permute(1, 2, 0).data.numpy())
            im_s.set_data(torch.abs(screen.grad).max(1)[0].squeeze().data.numpy())
            fig.canvas.draw_idle()
            if render_waiting_time:
                plt.pause(render_waiting_time)
            else:
                plt.pause(0.000001)
                input(f"{t}/{i_episode} Press Enter to continue...")

            action = action.max(1)[1].view(1, 1)
            # Continue the game
            new_state, _, done, _ = env.step(action.item())
            new_screen = torch.from_numpy(convert_state_to_image(new_state)).to(device)

            screen = new_screen

            if done:
                break

    print('Analysis complete')
