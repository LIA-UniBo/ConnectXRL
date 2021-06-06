from src.connectx.constraints import ConstraintType
from src.connectx.dqn import DQN
from src.connectx.environment import ConnectXGymEnv, convert_state_to_image
from src.connectx.policy import FeedForward, CNNPolicy

device = 'cpu'
env = ConnectXGymEnv('random', True)

# Get initial state and its size
init_screen = convert_state_to_image(env.reset())
screen_shape = (init_screen.shape[1], init_screen.shape[2], init_screen.shape[3])
cnn_policy = CNNPolicy(env.action_space.n, screen_shape, non_local=False).to(device)

ffnn_policy = FeedForward([6 * 7, 32, 32, 7])

dqn = DQN(env,
          ffnn_policy,
          constraint_type=ConstraintType.SBR)

dqn.training_loop(n_episodes_as_1st_player=100,
                  n_episodes_as_2nd_player=100,
                  render_env=False,
                  save_path='../',
                  update_plots_frequency=200)
input('Press to close ...')
