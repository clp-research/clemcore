from contextlib import contextmanager
from typing import List, Tuple, Dict

from clemcore.backends import Model
from clemcore.clemgame import GameSpec, benchmark, GameBenchmark, Player


@contextmanager
def make_env(game_spec: GameSpec, players: List[Model],
             instances_name: str = None, shuffle_instances: bool = False):
    with benchmark.load_from_spec(game_spec, do_setup=True, instances_name=instances_name) as game:
        yield GameEnv(game, players=players, shuffle_instances=shuffle_instances)


class GameEnv:

    def __init__(self, game: GameBenchmark, players: List[Model], shuffle_instances: bool = False):
        self.game = game
        self.players = players
        # setup iterator to go through tasks / game instances
        self.task_iterator = game.create_game_instance_iterator(shuffle_instances)
        if len(self.task_iterator) < 1:
            raise RuntimeError(f"No game instances given for the game: '{self.game.game_name}'")
        # variables initialized on reset()
        self.game_instance = None
        self.experiment_config = None
        self.master = None
        # reset here so that game env is fully functional after init
        self.reset()

    def reset(self):
        try:
            self.experiment_config, self.game_instance = next(self.task_iterator)
            self.master = self.game.create_game_master(self.experiment_config, self.players)
            self.master.setup(**self.game_instance)
        except StopIteration:
            self.task_iterator.reset()
            self.reset()

    def get_observation(self) -> Tuple[Player, Dict, Dict]:
        player = self.master.get_current_player()
        context = self.master.get_context_for(player)
        state = self.master.get_game_state()
        return player, context, state

    def step(self, action: str) -> Tuple[bool, Dict]:
        return self.master.step(action)
