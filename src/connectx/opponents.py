import random


def always_one_position_agent(observation: dict,
                              configuration: dict) -> int:
    """
    Agent that choose always the same column at each match.

    :param observation: turn's data (board as a list, mark as 1 or 2)
    :param configuration: environment's data (num of columns, num of rows)
    :return: the action
    """

    # Choose the position at the first move
    if sum(observation['board']) in [0, 1]:
        return random.randint(0, configuration['columns'])

    return observation['board'].index(observation['mark']) % configuration['columns']
