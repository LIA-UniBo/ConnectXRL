from src.connectx.constraints import ConstraintType
from src.connectx.dqn import DQN
from src.connectx.environment import ConnectXGymEnv

env = ConnectXGymEnv('random', True)
dqn = DQN(env,
          non_local=True,
          constraint_type=None)

dqn.training_loop(n_episodes_as_1st_player=100,
                  n_episodes_as_2nd_player=100,
                  render_env=False,
                  save_path='../',
                  update_plots_frequency=200)
input('Press to close ...')
