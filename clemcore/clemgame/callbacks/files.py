from datetime import datetime
from pathlib import Path
from typing import Dict, TYPE_CHECKING, Any

from clemcore import get_version

if TYPE_CHECKING:  # to satisfy pycharm
    from clemcore.clemgame import GameMaster, GameBenchmark

from clemcore.clemgame.recorder import GameInteractionsRecorder
from clemcore.clemgame.callbacks.base import GameBenchmarkCallback
from clemcore.clemgame.resources import store_json, load_json


# for pycharm: suppress could be static checks, because methods might be overwritten
# noinspection PyMethodMayBeStatic
class ResultsFolder:
    """
    Instance-bound, single-pass results layout.

    Default assumptions:
        - Exactly one episode per instance
        - episode_id == instance_id
        - Each game instance has a single result location (repeated runs overwrite previous results)

    Structure:
        - results_dir (root)
            - <run_dir>
                - game_name
                    - experiment_name
                        - experiment.json
                        - instance_<id>
                            - instance.json
                            - interaction.json
    """

    def __init__(self, result_dir_path: Path, run_dir: str):
        self.results_dir_path: Path = result_dir_path
        self.run_dir: str = run_dir

    def to_results_dir_path(self) -> Path:
        return self.results_dir_path

    def to_run_dir_path(self) -> Path:
        return self.results_dir_path / self.run_dir

    def to_experiment_dir_path(self, game_master: "GameMaster") -> Path:
        game_dir = self.to_game_dir(game_master)
        experiment_dir = self.to_experiment_dir(game_master.experiment)
        return self.to_run_dir_path() / game_dir / experiment_dir

    def to_instance_dir_path(self, game_master: "GameMaster", game_instance: Dict) -> Path:
        experiment_path = self.to_experiment_dir_path(game_master)
        instance_dir = self.to_instance_dir(game_instance)
        return experiment_path / instance_dir

    def to_game_dir(self, game_master: "GameMaster") -> str:
        return game_master.game_spec.game_name

    def to_experiment_dir(self, experiment: Dict) -> str:
        return experiment["name"]

    def to_instance_dir(self, game_instance: Dict) -> str:
        return f"instance_{game_instance['game_id']:05d}"


class EpisodeResultsFolder(ResultsFolder):
    """
    Episode-based results layout.

    Allows iterating over the same game instance multiple times.
    Each game run corresponds to a new episode directory, independent of the underlying instance identity.

    Note:
    This aligns with repeated exposure to an initial state, e.g., reinforcement learning or stochastic evaluation.
    """

    def __init__(self, result_dir_path: Path, run_dir: str):
        super().__init__(result_dir_path, run_dir)
        self.episode_id = 0  # reset per process; overwrites on rerun

    def increment_episode_id(self):
        self.episode_id += 1

    def to_episode_dir(self) -> str:
        return f"episode_{self.episode_id:05d}"

    def to_instance_dir(self, game_instance: dict) -> str:
        """
        In episode-based layouts, the 'instance directory' corresponds
        to an episode rather than a unique game instance.
        """
        return self.to_episode_dir()


class EpisodeResultsFolderCallback(GameBenchmarkCallback):

    def __init__(self, results_folder: EpisodeResultsFolder):
        self.results_folder = results_folder

    def on_game_start(self, game_master: "GameMaster", game_instance: dict):
        # One game execution == one episode
        self.results_folder.increment_episode_id()


class EpochResultsFolder(ResultsFolder):
    """
    Epoch-based results layout.

    Each benchmark run corresponds to a new epoch.
    Within an epoch, each game instance is evaluated exactly once.

    This aligns with dataset-style training loops, e.g., supervised learning.
    """

    def __init__(self, result_dir_path: Path, run_dir: str):
        super().__init__(result_dir_path, run_dir)
        self.epoch_id = 0  # reset per process; overwrites on rerun

    def increment_epoch_id(self):
        self.epoch_id += 1

    def to_run_dir_path(self):
        models_dir_path = super().to_run_dir_path() / f"epoch_{self.epoch_id:05d}"
        return models_dir_path


class EpochResultsFolderCallback(GameBenchmarkCallback):

    def __init__(self, results_folder: EpochResultsFolder):
        self.results_folder = results_folder

    def on_benchmark_start(self, game_benchmark: "GameBenchmark"):
        # assuming every benchmark run corresponds to an epoch
        self.results_folder.increment_epoch_id()


class RunFileSaver(GameBenchmarkCallback):

    def __init__(self, results_folder: ResultsFolder, *, player_model_infos: Any = None):
        self.results_folder = results_folder
        self.game_info = None
        self.benchmark_start = None
        self.num_instances = 0
        self.data = dict(clem_version=get_version(),
                         created=datetime.now().isoformat(),
                         player_models=player_model_infos,
                         games={})

        model_dir_path = self.results_folder.to_run_dir_path()
        run_file_path = Path(model_dir_path / "run.json")
        if run_file_path.exists():
            self.data = load_json(str(run_file_path))  # keep already stored values
        else:
            store_json(self.data, "run.json", model_dir_path)  # create file

    def on_benchmark_start(self, game_benchmark: "GameBenchmark"):
        self.benchmark_start = datetime.now()
        self.game_info = dict(game_path=game_benchmark.game_path, benchmark_start=self.benchmark_start.isoformat())
        self.data["games"][game_benchmark.game_name] = self.game_info
        store_json(self.data, "run.json", self.results_folder.to_run_dir_path())  # overwrite

    def on_game_start(self, game_master: "GameMaster", game_instance: Dict):
        self.num_instances += 1  # the instance iterator is not necessarily yet initialized, so we count here

    def on_benchmark_end(self, game_benchmark: "GameBenchmark"):
        benchmark_end = datetime.now()
        benchmark_duration = benchmark_end - self.benchmark_start
        self.game_info["benchmark_end"] = benchmark_end.isoformat()
        self.game_info["duration"] = str(benchmark_duration)
        self.game_info["duration_seconds"] = benchmark_duration.total_seconds()
        self.game_info["num_instances"] = self.num_instances
        store_json(self.data, "run.json", self.results_folder.to_run_dir_path())  # overwrite
        self.num_instances = 0
        self.game_info = None


class InstanceFileSaver(GameBenchmarkCallback):

    def __init__(self, results_folder: ResultsFolder):
        self.results_folder = results_folder

    def on_game_start(self, game_master: "GameMaster", game_instance: Dict):
        instance_dir_path = self.results_folder.to_instance_dir_path(game_master, game_instance)
        store_json(game_instance, "instance.json", instance_dir_path)


class ExperimentFileSaver(GameBenchmarkCallback):

    def __init__(self, results_folder: ResultsFolder, *, player_model_infos: Any = None):
        self.results_folder = results_folder
        self.player_models_infos = player_model_infos

    def on_game_start(self, game_master: "GameMaster", game_instance: Dict):
        experiment_dir_path = self.results_folder.to_experiment_dir_path(game_master)
        experiment_file_path = experiment_dir_path / "experiment.json"
        if experiment_file_path.is_file():
            return  # file already exists; only store once for all game instances
        experiment = game_master.experiment
        experiment_config = {k: experiment[k] for k in experiment if k != 'game_instances'}  # ignore instances
        experiment_config["timestamp"] = datetime.now().isoformat()
        experiment_config["game_name"] = game_master.game_spec.game_name
        experiment_config["player_models"] = self.player_models_infos
        store_json(experiment_config, "experiment.json", experiment_dir_path)


class InteractionsFileSaver(GameBenchmarkCallback):

    def __init__(self, results_folder: ResultsFolder, *, player_model_infos: Any = None):
        self.results_folder = results_folder
        self.player_models_infos = player_model_infos
        self._recorders: Dict[str, GameInteractionsRecorder] = {}

    @staticmethod
    def to_key(game_name: str, experiment_name: str, game_id: int):
        return f"{game_name}-{experiment_name}-{game_id}"

    def on_game_start(self, game_master: "GameMaster", game_instance: Dict):
        game_name = game_master.game_spec.game_name
        experiment_name = game_master.experiment["name"]
        game_id = game_instance["game_id"]
        # create, inject and register new game recorder
        game_recorder = GameInteractionsRecorder(game_name,
                                                 experiment_name,  # meta info for transcribe
                                                 game_id,  # meta info for transcribe
                                                 self.results_folder.run_dir,  # meta info for transcribe
                                                 self.player_models_infos)
        for player in game_master.get_players():
            game_recorder.log_player(player.name, player.game_role, player.model.name)
        game_master.register(game_recorder)

        _key = InteractionsFileSaver.to_key(game_name, experiment_name, game_id)
        self._recorders[_key] = game_recorder

    def on_game_end(self, game_master: "GameMaster", game_instance: Dict):
        game_name = game_master.game_spec.game_name
        experiment_name = game_master.experiment["name"]
        game_id = game_instance["game_id"]
        _key = InteractionsFileSaver.to_key(game_name, experiment_name, game_id)
        assert _key in self._recorders, f"Recorder must be registered on_game_start, but wasn't for: {_key}"
        recorder = self._recorders.pop(_key)  # auto-remove recorder from registry
        self._store_files(recorder, game_master, game_instance)

    def _store_files(self, recorder, game_master, game_instance):
        instance_dir_path = self.results_folder.to_instance_dir_path(game_master, game_instance)
        store_json(recorder.interactions, "interactions.json", instance_dir_path)
        store_json(recorder.requests, "requests.json", instance_dir_path)
