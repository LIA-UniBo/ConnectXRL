import torch

from src.connectx.constraints import ConstraintType
from src.connectx.environment import ConnectXGymEnv, convert_state_to_image
from src.connectx.evaluate import record_matches, show_recordings
from src.connectx.opponents import interactive_player
from src.connectx.policy import CNNPolicy

#############################
# Play with a trained model #
#############################

# Use the interactive player to play against the computer, otherwise 'random' or 'negamax' or declare your own function
# as in src.connectx.opponents.py
opponent = interactive_player
play_as_first_player = str(input('Do you want your trained agent to play as first (y/n)?')).lower() == 'y'

num_matches = int(input('How many match do you want to play?'))

# When the interactive player play as the 1st player some steps must actions must be skipped at the beginning
if not play_as_first_player and opponent is interactive_player:
    print('Creation of environment ... Press any button')
env = ConnectXGymEnv(opponent, play_as_first_player)

if not play_as_first_player and opponent is interactive_player:
    print('Initialize policy ... Press any button')

# Policy
init_screen = convert_state_to_image(env.reset())
screen_shape = (init_screen.shape[1], init_screen.shape[2], init_screen.shape[3])
agent = CNNPolicy(env.action_space.n,
                  screen_shape)

constraint_type = int(input('Insert number of the agent you want to play:\n'
                            f'0 - DQN\n'
                            f'{ConstraintType.LOGIC_PURE.value} - {ConstraintType.LOGIC_PURE.name}\n'
                            f'{ConstraintType.LOGIC_TRAIN.value} - {ConstraintType.LOGIC_TRAIN.name}\n'
                            f'{ConstraintType.SBR.value} - {ConstraintType.SBR.name}\n'
                            f'{ConstraintType.SPE.value} - {ConstraintType.SPE.name}\n'
                            f'{ConstraintType.CDQN.value} - {ConstraintType.CDQN.name}\n'))

device = 'cpu'
weight_path = '../../models/' + (ConstraintType(constraint_type).name if constraint_type > 0 else 'dqn').lower() + '.pt'
agent.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))

config = {'columns': 7,
          'rows': 6,
          'inarow': 4,
          'c_type': ConstraintType(constraint_type) if constraint_type > 0 else None}

state_recording, action_recording = record_matches(env,
                                                   agent,
                                                   config,
                                                   play_as_first_player=play_as_first_player,
                                                   num_matches=num_matches,
                                                   render_env=True,
                                                   keep_player_colour=True)

for i, (sr, ar) in enumerate(zip(state_recording, action_recording)):
    print(f'Play recording {i + 1}')
    show_recordings(sr, ar)
