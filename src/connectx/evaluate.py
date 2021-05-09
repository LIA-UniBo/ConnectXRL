from typing import Union, Callable, Optional, Dict, List

import numpy as np
from kaggle_environments import evaluate, Environment

WIDTH = 256
HEIGHT = 256


def play(env: Environment,
         player1: Union[str, Callable],
         player2: Union[str, Callable],
         width: Optional[int] = WIDTH,
         height: Optional[int] = HEIGHT) -> None:
    """
    Render the game between two agents.

    :param env: Kaggle environment
    :param player1: one of the predefined agents ("random", "negamax") or a custom one (Callable)
    :param player2: one of the predefined agents ("random", "negamax") or a custom one (Callable)
    :param width: board rendering width
    :param height: board rendering width
    """

    env.run([player1, player2])
    env.render(mode='ipython', width=width, height=height)


def interactive_play(env: Environment,
                     player1: Optional[Union[str, Callable]] = None,
                     player2: Optional[Union[str, Callable]] = None,
                     width: Optional[int] = WIDTH,
                     height: Optional[int] = HEIGHT) -> None:
    """
    Interactively play with the computer. The interactive player MUST be passed as None.

    :param env: Kaggle environment
    :param player1: one of the predefined agents ("random", "negamax") or a custom one (Callable)
    :param player2: one of the predefined agents ("random", "negamax") or a custom one (Callable)
    :param width: board rendering width
    :param height: board rendering width
    """

    if player1 is not None and player2 is not None:
        raise ValueError('One of the player must be None to be assigned to the interactive player.')

    env.play([player1, player2], width=width, height=height)


def evaluate_matches(matches: List[Dict[str, int]]):
    """
    Evaluate multiple matches.

    :param matches: List containing dictionaries specifying the matches. Each match defines the 2 players 'ply1' and
    'ply2' and how many times the game is played. A mean of the rewards of each play is displayed for each match.
    """

    for m in matches:
        print(f' {m["ply1"]} vs {m["ply1"]} ({m["eps"]} eps):', np.mean(evaluate('connectx',
                                                                                 [m['ply1'], m['ply2']],
                                                                                 num_episodes=m['eps'])))


def get_win_percentages(player: Union[str, Callable],
                        opponents: List[Union[str, Callable]],
                        player_name: Optional[str] = None,
                        config: Optional[Dict] = None,
                        n_rounds_as_1st_player: int = 100,
                        n_rounds_as_2nd_player: int = 100):
    """
    Print win/loss percentages between the player and the list of opponents.

    :param player: Your player
    :param player_name: Your player's name
    :param opponents: List of opponents
    :param config: dict specifying the board ('rows' and 'columns') and the stones to win ('inarow'). Default values are
    {'rows': 6, 'columns': 7, 'inarow': 4}
    :param n_rounds_as_1st_player: number of rounds where agent 1 play as the first player
    :param n_rounds_as_2nd_player: number of rounds where agent 1 play as the second player
    """

    if config is None:
        config = {'rows': 6, 'columns': 7, 'inarow': 4}

    for opponent in opponents:
        outcomes = evaluate('connectx', [player, opponent], config, [], n_rounds_as_1st_player)
        outcomes += [[b, a] for [a, b] in evaluate('connectx', [opponent, player], config, [], n_rounds_as_2nd_player)]

        opponent_name = opponent.__name__ if callable(opponent) else opponent
        if player_name is None:
            player_name = player.__name__ if callable(player) else player

        # print(outcomes)
        print()
        print(f'{player_name} win percentage: {np.round(outcomes.count([1, -1]) / len(outcomes), 2)} - '
              f'{outcomes.count([1, -1])}/{len(outcomes)}')
        print(f'{opponent_name} win percentage: {np.round(outcomes.count([-1, 1]) / len(outcomes), 2)} - '
              f'{outcomes.count([-1, 1])}/{len(outcomes)}')
        print(f'Draw percentage: {np.round(outcomes.count([0, 0]) / len(outcomes), 2)} - '
              f'{outcomes.count([0, 0])}/{len(outcomes)}')
        print(f'Number of invalid moves by {player_name}: {outcomes.count([None, 0])}')
        print(f'Number of invalid moves by {opponent_name}: {outcomes.count([0, None])}\n')
