from clemcore.clemgame import GameInstanceIterator
from pettingzoo import AECEnv
from pettingzoo.utils import BaseWrapper


class GameInstanceIteratorWrapper(BaseWrapper):
    """
    A wrapper that iterates through a GameInstanceIterator, either once or infinitely.

    Args:
        wrapped_env: A pettingzoo AECEnv instance.
        game_iterator: An instance of GameInstanceIterator pre-loaded with instances.
        single_pass: If True, the iterator stops after passed once through all instances (e.g., for evaluation).
                     If False (default), the iterator cycles infinitely (e.g., for RL training).
    """

    def __init__(self, wrapped_env: AECEnv, game_iterator: GameInstanceIterator, single_pass: bool = False):
        super().__init__(wrapped_env)
        self.game_iterator = game_iterator.__deepcopy__()
        self.game_iterator.reset()
        self.options = {}
        if not single_pass:
            from itertools import cycle
            self.game_iterator = cycle(self.game_iterator)

    def reset(self, seed: int | None = None, options: dict | None = None):
        experiment, game_instance = next(self.game_iterator)
        self.options = options or {}
        self.options["experiment"] = experiment
        self.options["game_instance"] = game_instance
        super().reset(seed=seed, options=options)
