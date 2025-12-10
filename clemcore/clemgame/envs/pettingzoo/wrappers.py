from typing import Callable

from clemcore.clemgame import GameInstanceIterator, GameSpec, GameBenchmark
from pettingzoo import AECEnv
from pettingzoo.utils import BaseWrapper


class GameBenchmarkWrapper(BaseWrapper):
    """
    A wrapper that loads a GameBenchmark from a GameSpec and passes it to the wrapped environment.
    """

    def __init__(self, env_class: Callable[[GameBenchmark], AECEnv], game_spec: GameSpec, **env_kwargs):
        self.game_benchmark = GameBenchmark.load_from_spec(game_spec)
        super().__init__(env_class(self.game_benchmark, **env_kwargs))

    def close(self) -> None:
        super().close()
        self.game_benchmark.close()


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
