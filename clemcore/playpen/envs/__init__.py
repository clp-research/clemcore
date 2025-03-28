import abc
from contextlib import contextmanager
from typing import List, Tuple, Callable, Union, Dict

from clemcore.backends import Model
from clemcore.clemgame import GameSpec, benchmark

from clemcore.playpen.envs.game_env import GameEnv
from clemcore.playpen.envs.tree_env import GameTreeEnv


@contextmanager
def make_env(game_spec: GameSpec, players: List[Model],
             instances_name: str = None, shuffle_instances: bool = False):
    with benchmark.load_from_spec(game_spec, do_setup=True, instances_name=instances_name) as game:
        task_iterator = game.create_game_instance_iterator(shuffle_instances)
        yield GameEnv(game, players, task_iterator)


@contextmanager
def make_tree_env(game_spec: GameSpec, players: List[Model],
                  instances_name: str = None, shuffle_instances: bool = False,
                  branching_factor: int = 2, pruning_fn=lambda candidates: candidates):
    with benchmark.load_from_spec(game_spec, do_setup=True, instances_name=instances_name) as game:
        assert branching_factor > 1, "The branching factor must be greater than one"
        task_iterator = game.create_game_instance_iterator(shuffle_instances)
        yield GameTreeEnv(game, players, task_iterator, branching_factor=branching_factor, pruning_fn=pruning_fn)


class PlayPenEnv(abc.ABC):

    def __init__(self):
        self._done: bool = False

    def is_done(self) -> bool:
        return self._done

    @abc.abstractmethod
    def reset(self) -> None:
        pass

    @abc.abstractmethod
    def observe(self) -> Tuple[Callable, Union[Dict, List[Dict]]]:
        pass

    @abc.abstractmethod
    def step(self, responses: Union[str, List]) -> Tuple[Union[bool, List], Union[Dict, List]]:
        pass
