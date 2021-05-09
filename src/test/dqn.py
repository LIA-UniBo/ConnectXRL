from src.connectx.constraints import ConstraintType
from src.connectx.dqn import DQN
from src.connectx.environment import ConnectXGymEnv

env = ConnectXGymEnv('random', True)
dqn = DQN(env,
          non_local=True,
          constraint_type=ConstraintType.SBR)

dqn.training_loop(n_episodes_as_1st_player=10,
                  n_episodes_as_2nd_player=10,
                  render_env=False,
                  save_path='../',
                  update_plots_frequency=200)
input('Press to close ...')
