from typing import List, Tuple, Dict, Callable, Union

from clemcore.backends import Model
from clemcore.clemgame import GameBenchmark, DialogueGameMaster, GameInstanceIterator
from clemcore.playpen.envs import PlayPenEnv


class GameEnv(PlayPenEnv):

    def __init__(self, game: GameBenchmark, player_models: List[Model], task_iterator: GameInstanceIterator,
                 reset=True):
        super().__init__()
        self._game = game
        self._player_models = player_models
        self._dialogue_pair_descriptor = game.get_dialogue_pair_descriptor(player_models)
        # setup iterator to go through tasks / game instances
        self._task_iterator = task_iterator
        if len(self._task_iterator) < 1:
            raise RuntimeError(f"No game instances given for the game: '{self._game.game_name}'")
        # variables initialized on reset()
        self._game_instance: Dict = None
        self._experiment: Dict = None
        self._master: DialogueGameMaster = None
        if reset:  # if reset, then the game env is fully functional after init
            self.reset()

    @property
    def experiment(self):
        return self._experiment

    @property
    def master(self):
        return self._master

    @master.setter
    def master(self, master):
        self._master = master

    def reset(self) -> None:
        try:
            self._experiment, self._game_instance = next(self._task_iterator)
            self.master = self._game.create_game_master(self._experiment, self._player_models)
            self.master.setup(**self._game_instance)
        except StopIteration:
            self._task_iterator.reset()
            self.reset()

    def observe(self) -> Tuple[Callable, Union[Dict, List[Dict]]]:
        player = self.master.get_current_player()
        context = self.master.get_context_for(player)
        return player, context

    def step(self, response: Union[str, List]) -> Tuple[Union[bool, List], Union[Dict, List]]:
        self._done, info = self.master.step(response)
        return self._done, info

    def clone(self) -> "GameEnv":
        _clone = GameEnv(self._game,  # should not be mutated
                         self._player_models,  # should not be mutated
                         self._task_iterator.clone(),
                         reset=False)
        _clone._game_instance = self._game_instance  # should not be mutated
        _clone._dialogue_pair_descriptor = self._dialogue_pair_descriptor  # should not be mutated
        _clone._experiment = self._experiment  # should not be mutated
        _clone._master = self._master.clone()
        return _clone

    def store_experiment_config(self, experiment_dir: str, results_dir: str):
        self._game.store_results_file(self.experiment,
                                      f"experiment_{self.experiment['name']}.json",
                                      self._dialogue_pair_descriptor,
                                      sub_dir=experiment_dir,
                                      results_dir=results_dir)

    def store_game_instance(self, episode_dir, results_dir):
        self._game.store_results_file(self._game_instance,
                                      f"instance.json",
                                      self._dialogue_pair_descriptor,
                                      sub_dir=episode_dir,
                                      results_dir=results_dir)

    def store_game_interactions(self, episode_dir, results_dir):
        self.master.store_records(results_dir, self._dialogue_pair_descriptor, episode_dir)
