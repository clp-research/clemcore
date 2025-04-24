import collections
from typing import Dict

from clemcore.clemgame.resources import GameResourceLocator


class GameInstanceGenerator(GameResourceLocator):
    """Create all game instances for a game benchmark.
    Results in an instances.json with the following structure:

    "experiments": [ # this is required
        {
            "name": <experiment-name>, # this is required
            "param1": "value1", # optional
            "param2": "value2", # optional
            "game_instances": [ # this is required
                {"id": <value>, "initial_prompt": ... },
                {"id": <value>, "initial_prompt": ... }
            ]
        }
    ]
    """

    def __init__(self, path: str):
        """
        Args:
            path: The path to the game.
        """
        super().__init__(path=path)
        self.instances = dict(experiments=list())

    def add_experiment(self, experiment_name: str) -> Dict:
        """Add an experiment to the game benchmark.
        Experiments are sets of instances, usually with different experimental variables than other experiments in a
        game benchmark.
        Call this method and adjust the returned dict to configure the experiment.
        For game instances use add_game_instance!
        Args:
            experiment_name: Name of the new game experiment.
        Returns:
            A new game experiment dict.
        """
        experiment = collections.OrderedDict(name=experiment_name)
        experiment["game_instances"] = list()
        self.instances["experiments"].append(experiment)
        return experiment

    def add_game_instance(self, experiment: Dict, game_id):
        """Add an instance to an experiment.
        An instance holds all data to run a single episode of a game.
        Call this method and adjust the returned dict to configure the instance.
        Args:
            experiment: The experiment to which a new game instance should be added.
            game_id: Identifier of the new game instance.
        Returns:
            A new game instance dict.
        """
        game_instance = dict(game_id=game_id)
        experiment["game_instances"].append(game_instance)
        return game_instance

    def on_generate(self, **kwargs):
        """Game-specific instance generation.
        This method is intended for creation of instances and experiments for a game benchmark. Use the add_experiment
        and add_game_instance methods to create the game benchmark.
        Must be implemented!
        Args:
            kwargs: Keyword arguments (or dict) with data controlling instance generation.
        """
        raise NotImplementedError()

    def generate(self, filename="instances.json", **kwargs):
        """Generate the game benchmark and store the instances JSON file.
        Intended to not be modified by inheriting classes, modify on_generate instead.
        Args:
            filename: The name of the instances JSON file to be stored in the 'in' subdirectory. Defaults to
                'instances.json'.
            kwargs: Keyword arguments (or dict) to pass to the on_generate method.
        """
        self.on_generate(**kwargs)
        self.store_file(self.instances, filename, sub_dir="in")
