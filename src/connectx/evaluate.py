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
