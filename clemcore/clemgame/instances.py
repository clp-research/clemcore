import abc
import collections
import logging
import os
import random
from copy import copy
from typing import Dict, final, Optional, Callable, List, Tuple

import numpy as np
from clemcore.clemgame.registry import GameSpec

from clemcore.clemgame.resources import GameResourceLocator, load_json

stdout_logger = logging.getLogger("clemcore.run")


def to_instance_filter(dataset) -> Callable[[str, str], List[int]]:
    """
    Converts the given dataset into a game instance filter function.

    Args:
        dataset: a list of dict-like rows with game, experiment, task_id values

    Returns:
        A callable mapping of (game_name, experiment_name) tuples to lists of task ids (game instance ids)
    """
    tasks_by_group = collections.defaultdict(list)
    for row in dataset:
        key = (row['game'], row['experiment'])
        tasks_by_group[key].append(int(row['task_id']))
    return lambda game, experiment: tasks_by_group[(game, experiment)]


def to_rows(instances: dict) -> list[dict]:
    """Transforms a hierarchical instances dict into a flat list of row dicts.

    Each row has two keys:
        - "experiment": the experiment metadata (all fields except "game_instances")
        - "game_instance": the individual instance data (game_id and instance-specific fields)

    The instances dict must follow this structure:
        {
            "experiments": [
                {
                    "name": <experiment-name>,
                    "param1": "value1",
                    "game_instances": [
                        {"game_id": <value>, ...},
                        {"game_id": <value>, ...}
                    ]
                }
            ]
        }

    Raises:
        ValueError: If the instances dict is missing "experiments", it is not a list, or it is empty.
    """
    if "experiments" not in instances:
        raise ValueError("No 'experiments' key in instances")
    if not isinstance(instances["experiments"], list):
        raise ValueError("'experiments' must be a list")
    if len(instances["experiments"]) == 0:
        raise ValueError("'experiments' list is empty")
    results = []
    experiment_names = []
    for experiment in instances["experiments"]:
        experiment_names.append(experiment["name"])
        for game_instance in experiment["game_instances"]:
            experiment_data = {k: experiment[k] for k in experiment if k != 'game_instances'}
            results.append({"experiment": experiment_data, "game_instance": game_instance})
    return results


class GameInstances:
    """A collection of game instance rows for a single game, loaded from instances.json.

    Each row is a dict with two keys:
        - "experiment": the experiment metadata (name and parameters, excluding game_instances)
        - "game_instance": the individual instance data (game_id and instance-specific parameters)

    Rows are produced by `to_rows()` from the hierarchical instances.json structure and held
    eagerly in memory. Use `filter()` to sub-select rows, and `find_by_game_id()` for direct lookup.

    Args:
        game_name: The name of the game these instances belong to.
        rows: Flat list of row dicts as returned by `to_rows()`.
    """

    def __init__(self, game_name: str, rows: list):
        assert game_name is not None, "Game name must be given as 'game_name'"
        assert rows is not None, "Instances must be given as 'rows'"
        self._game_name = game_name
        self._rows: list[dict] = rows
        self._experiment_names = list({row["experiment"]["name"] for row in rows})

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)

    def __str__(self):
        return f"GameInstances({self._game_name}, {len(self._experiment_names)} experiments, {len(self._rows)} rows)"

    def describe(self) -> str:
        """Returns a detailed description, including experiment names, for logging."""
        return (f"{self._game_name}: {len(self._rows)} rows "
                f"from {len(self._experiment_names)} experiments: {self._experiment_names}")

    def filter(self, condition: Callable[[dict], bool]) -> "GameInstances":
        """Returns a new GameInstances containing only rows for which the condition returns True.

        The condition receives a single row dict with "experiment" and "game_instance" keys,
        aligned with the HuggingFace Dataset.filter() signature for future compatibility.

        Args:
            condition: A callable that takes a row dict and returns True to keep the row.
        """
        rows = [row for row in self._rows if condition(row)]
        return GameInstances(self._game_name, rows)

    def find_by_game_id(self, game_id) -> dict:
        """Returns the row dict for the given game_id or raises ValueError if not found.

        Args:
            game_id: The game_id to look up (as stored in row["game_instance"]["game_id"]).
        """
        for row in self._rows:
            if row["game_instance"]["game_id"] == game_id:
                return row
        raise ValueError(f"game_id={game_id!r} not found in game instances for {self._game_name}")

    @classmethod
    def from_game_spec(cls, game_spec: GameSpec) -> "GameInstances":
        """Load game instances from the path and file name defined in the given GameSpec.

        Args:
            game_spec: The game spec providing game_name, game_path, and optional instances file name.
        """
        if not hasattr(game_spec, "instances"):
            game_spec.instances = "instances"
        return cls.from_file(
            game_spec.game_name,
            os.path.join(game_spec.game_path, "in"),
            game_spec.instances
        )

    @classmethod
    def from_file(cls,
                  game_name: str,
                  instance_dir_path: str,
                  instance_file_name: str = "instances") -> "GameInstances":
        """Load game instances from a JSON file on disk.

        Args:
            game_name: The name of the game these instances belong to.
            instance_dir_path: Path to the directory containing the instances JSON file.
            instance_file_name: Name of the instances file (without .json extension).
        """
        file_path = os.path.join(instance_dir_path, instance_file_name)
        instances = load_json(file_path)
        rows = to_rows(instances)
        return cls(game_name, rows)


class GameInstanceIterator:
    """
    The instances.json must follow the structure:
        "experiments": [ # this is required
            {
                "name": <experiment-name>, # this is required
                "param1": "value1", # optional
                "param2": "value2", # optional
                "game_instances": [ # this is required
                    {"game_id": <value>, "initial_prompt": ... },
                    {"game_id": <value>, "initial_prompt": ... }
                ]
            }

    Args:
        game_name: The name of the game to which the instances belong to.
        instances: The instances dict with experiments and game instances.
        sub_selector: A callable mapping from (game_name, experiment_name) tuples to lists of game instance ids.
            If a mapping returns None, then all game instances will be used.
    """

    def __init__(self,
                 game_name: str,
                 instances: Dict,
                 *,
                 sub_selector: Optional[Callable[[str, str], List[int]]] = None):
        assert game_name is not None, "Game name must be given"
        assert instances is not None, "Instances must be given"
        self._game_name = game_name
        self._instances: Dict = instances
        self._sub_selector: Optional[Callable[[str, str], List[int]]] = sub_selector
        self._queue = []

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[Dict, Dict]:
        try:
            return self._queue.pop(0)
        except IndexError:
            raise StopIteration()

    def __len__(self):
        return len(self._queue)

    def __deepcopy__(self) -> "GameInstanceIterator":
        _copy = type(self).__new__(self.__class__)
        _copy._game_name = self._game_name
        _copy._instances = self._instances
        _copy._sub_selector = self._sub_selector
        _copy._queue = copy(self._queue)  # no need to copy the underlying instances
        return _copy

    def reset(self, verbose: bool = False) -> "GameInstanceIterator":
        self._queue = []
        experiment_names = []
        num_instances = 0
        for index, experiment in enumerate(self._instances["experiments"]):
            filtered_experiment = {k: experiment[k] for k in experiment if k != 'game_instances'}
            selected_ids: Optional[List[int]] = None
            # some bookkeeping and logging
            if self._sub_selector is None:
                experiment_names.append(experiment["name"])
            else:
                selected_ids = self._sub_selector(self._game_name, experiment["name"])
                if selected_ids is None:  # use all instances
                    experiment_names.append(experiment["name"])
                elif len(selected_ids) == 0:
                    if verbose:
                        stdout_logger.info("Skip experiment %s for %s", experiment["name"], self._game_name)
                else:
                    experiment_names.append(experiment["name"])
                    if verbose:
                        stdout_logger.info("Sub-select for %s experiment %s instances with game_ids: %s",
                                           self._game_name, experiment["name"], selected_ids)
            # add instances to queue, if eligible
            for game_instance in experiment["game_instances"]:
                if selected_ids is None or game_instance["game_id"] in selected_ids:
                    self._queue.append((filtered_experiment, game_instance))
                    num_instances += 1
        if verbose:
            stdout_logger.info("Prepared instance queue for %s using %s experiments %s and %s instances in total.",
                               self._game_name, len(experiment_names), experiment_names, num_instances)
        return self

    @classmethod
    def from_game_spec(cls,
                       game_spec: GameSpec,
                       *,
                       sub_selector: Optional[Callable[[str, str], List[int]]] = None):
        """Load a game instance iterator using information from the given game spec.

        Args:
            game_spec: The game spec with a game path and instance file name.
            sub_selector: A callable mapping from (game_name, experiment_name) tuples to lists of game instance ids.
                If a mapping returns None, then all game instances will be used.
        """
        if not hasattr(game_spec, "instances"):
            game_spec.instances = "instances"  # if not already set, fallback to default file name
        return cls.from_file(
            game_spec.game_name,
            os.path.join(game_spec.game_path, "in"),
            game_spec.instances,
            sub_selector=sub_selector
        )

    @classmethod
    def from_file(cls,
                  game_name: str,
                  instance_dir_path: str,
                  instance_file_name: str = "instances",
                  *,
                  sub_selector: Optional[Callable[[str, str], List[int]]] = None):
        """Load a game instance iterator using the given file path.

        Args:
            game_name: The name of the game to which the instances belong to. Necessary for the sub_selector to work.
            instance_dir_path: The path the directory containing a JSON file with the game instances.
            instance_file_name: The name of the instance file to load.
            sub_selector: A callable mapping from (game_name, experiment_name) tuples to lists of game instance ids.
                If a mapping returns None, then all game instances will be used.
        """
        file_path = os.path.join(instance_dir_path, instance_file_name)
        instances = load_json(file_path)
        if "experiments" not in instances:
            raise ValueError(f"{game_name}: No 'experiments' key in {instance_file_name}")
        experiments = instances["experiments"]
        if not isinstance(experiments, list):
            raise ValueError(f"{game_name}: Experiments in {instance_file_name} is not a list")
        if len(experiments) == 0:
            raise ValueError(f"{game_name}: Experiments list in {instance_file_name} is empty")
        return cls(game_name, instances, sub_selector=sub_selector)


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

    @final
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

    @final
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

    @abc.abstractmethod
    def on_generate(self, seed: int, **kwargs):
        """Game-specific instance generation.
        This method is intended for creation of instances and experiments for a game benchmark. Use the add_experiment
        and add_game_instance methods to create the game benchmark.
        Must be implemented!
        Args:
            seed: The random seed set for `random` and `np.random`. Defaults to None.
            kwargs: Keyword arguments (or dict) with data controlling instance generation.
        """
        pass

    @final
    def generate(self, filename="instances.json", seed=None, **kwargs) -> str:
        """Generate the game benchmark and store the instances JSON file.
        Intended to not be modified by inheriting classes, modify on_generate instead.
        Args:
            filename: The name of the instances JSON file to be stored in the 'in' subdirectory. Defaults to
                'instances.json'.
            seed: The random seed to be set. Defaults to None.
            kwargs: Keyword arguments (or dict) to pass to the on_generate method.
        """
        random.seed(seed)
        np.random.seed(seed)
        self.on_generate(seed, **kwargs)
        file_path = self.store_file(self.instances, filename, sub_dir="in")
        return file_path
