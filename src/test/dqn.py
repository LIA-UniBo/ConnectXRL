from src.connectx.dqn import DQN
from src.connectx.environment import ConnectXGymEnv

env = ConnectXGymEnv('random', True)
dqn = DQN(env)

dqn.training_loop(20000,
                  render_env=False,
                  save_path='../',
                  update_plots_frequency=200)
input('Press to close ...')
