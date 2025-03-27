import abc
from contextlib import contextmanager
from typing import List, Tuple, Callable, Union, Dict

from clemcore.backends import Model
from clemcore.clemgame import GameSpec, benchmark
from clemcore.playpen import GameEnv
from clemcore.playpen.envs.env import GameTreeEnv


@contextmanager
def make_env(game_spec: GameSpec, players: List[Model],
             instances_name: str = None, shuffle_instances: bool = False,
             branching_factor: int = 1, pruning_fn=lambda candidates: candidates):
    with benchmark.load_from_spec(game_spec, do_setup=True, instances_name=instances_name) as game:
        assert branching_factor > 0, "The branching factor must be greater than zero"
        if branching_factor == 1:
            # this could also resolve to a tree env with branching factor one, but
            # for clarity we instantiate here the game benchmark environment directly
            yield GameEnv(game, players=players, shuffle_instances=shuffle_instances)
        else:
            yield GameTreeEnv(game, players=players, shuffle_instances=shuffle_instances,
                              branching_factor=branching_factor, pruning_fn=pruning_fn)


class PlayPenEnv(abc.ABC):

    @abc.abstractmethod
    def reset(self) -> None:
        pass

    @abc.abstractmethod
    def observe(self) -> Tuple[Callable, Union[List, Dict]]:
        pass

    @abc.abstractmethod
    def step(self, response: Union[str, List]) -> Tuple[bool, Dict]:
        pass
