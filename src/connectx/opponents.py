import random


def always_one_position_player(observation: dict,
                               configuration: dict) -> int:
    """
    Agent that choose always the same column at each match.

    :param observation: turn's data (board as a list, mark as 1 or 2)
    :param configuration: environment's data (num of columns, num of rows)
    :return: the action
    """

    # Choose the position at the first move
    if sum(observation['board']) in [0, 1]:
        return random.randint(0, configuration['columns'] - 1)

    return observation['board'].index(observation['mark']) % configuration['columns']


def interactive_player(observation: dict,
                       configuration: dict):
    """
    Agent playable from the console (actions from 1 to configuration["columns"] + 1). Prints on the console are delayed,
    render the environment externally.

    :param observation: turn's data (board as a list, mark as 1 or 2)
    :param configuration: environment's data (num of columns, num of rows)
    :return: action performed
    """

    return int(input(f'Decide an action ({1}-{configuration["columns"]}): ')) - 1


def random_player(observation: dict,
                  configuration: dict):
    """
    Performs random actions without any check.

    :param observation: turn's data (board as a list, mark as 1 or 2)
    :param configuration: environment's data (num of columns, num of rows)
    :return: action performed
    """

    return random.randint(0, configuration['columns'] - 1)
