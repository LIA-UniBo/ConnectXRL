import torch

from src.connectx.environment import ConnectXGymEnv, convert_state_to_image
from src.connectx.policy import CNNPolicy
from src.xrl.shap_rl import record_matches, interactive_player, show_recordings, explain

opponent = 'random'  # interactive_player
play_as_first_player = True

env = ConnectXGymEnv(opponent,
                     play_as_first_player)

init_screen = convert_state_to_image(env.reset())
screen_shape = (init_screen.shape[1], init_screen.shape[2], init_screen.shape[3])

agent = CNNPolicy(env.action_space.n,
                  screen_shape)

device = 'cpu'
weight_path = './weights_5000.pt'
agent.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))

state_recording, action_recording = record_matches(env, agent)

print(state_recording, action_recording)

show_recordings(state_recording[0], action_recording[0])

shap_values = explain(agent, state_recording, action_recording)

show_recordings(shap_values[0], action_recording[0])
