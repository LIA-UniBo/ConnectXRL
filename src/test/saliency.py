import torch

from src.connectx.environment import ConnectXGymEnv, convert_state_to_image
from src.connectx.policy import CNNPolicy
from src.xrl.saliency import show_saliency_map

env = ConnectXGymEnv('random', True)
init_screen = convert_state_to_image(env.reset())
screen_shape = (init_screen.shape[1], init_screen.shape[2], init_screen.shape[3])

agent = CNNPolicy(env.action_space.n,
                  screen_shape)

device = 'cpu'
weight_path = './weights_2000.pt'
agent.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))

show_saliency_map(env, agent, 30, device=device)
