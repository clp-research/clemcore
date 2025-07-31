import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, TYPE_CHECKING

if TYPE_CHECKING:  # to satisfy pycharm
    from clemcore.clemgame import GameMaster

from clemcore.backends import Model
from clemcore.clemgame.recorder import GameInteractionsRecorder
from clemcore.clemgame.callbacks.base import GameBenchmarkCallback
from clemcore.clemgame.resources import store_json


def to_model_results_folder(player_models: List[Model]):
    def to_descriptor(model: Model):
        return f"{model.name}-t{model.temperature}"

    model_descriptors = [to_descriptor(m) for m in player_models]
    folder_name = "--".join(model_descriptors)
    if len(player_models) <= 2:
        return folder_name
    _hash = hashlib.sha1(folder_name.encode()).hexdigest()[:8]
    return f"group-{len(player_models)}p-{_hash}"


class ResultsFolder:
    """
        Represents the following structure:
            - results_dir (root)
                - model_folder_name
                    - game_name
                        - experiment_name
                            - experiment.json
                            - episode_id
                                - instance.json
                                - interaction.json
    """

    def __init__(self, result_dir_path: Path, player_models: List[Model]):
        self.results_dir_path: Path = result_dir_path
        self.models_dir: str = to_model_results_folder(player_models)

    def to_results_dir_path(self) -> Path:
        return self.results_dir_path

    def to_models_dir_path(self) -> Path:
        return self.results_dir_path / self.models_dir

    def to_experiment_dir_path(self, game_master: "GameMaster") -> Path:
        game_dir = ResultsFolder.to_game_dir(game_master)
        experiment_dir = ResultsFolder.to_experiment_dir(game_master.experiment)
        return self.to_models_dir_path() / game_dir / experiment_dir

    def to_instance_dir_path(self, game_master: "GameMaster", game_instance: Dict) -> Path:
        experiment_path = self.to_experiment_dir_path(game_master)
        instance_dir = ResultsFolder.to_instance_dir(game_instance)
        return experiment_path / instance_dir

    @staticmethod
    def to_game_dir(game_master: "GameMaster") -> str:
        return game_master.game_spec.game_name

    @staticmethod
    def to_experiment_dir(experiment: Dict) -> str:
        return experiment["name"]

    @staticmethod
    def to_instance_dir(game_instance: Dict) -> str:
        return f"instance_{game_instance['game_id']:05d}"


class InstanceFileSaver(GameBenchmarkCallback):
    def __init__(self, result_dir_path: Path, player_models: List[Model]):
        self.results_folder = ResultsFolder(result_dir_path, player_models)

    def on_game_start(self, game_master: "GameMaster", game_instance: Dict):
        instance_dir_path = self.results_folder.to_instance_dir_path(game_master, game_instance)
        store_json(game_instance, "instance.json", instance_dir_path)


class ExperimentFileSaver(GameBenchmarkCallback):

    def __init__(self, result_dir_path: Path, player_models: List[Model]):
        self.results_folder = ResultsFolder(result_dir_path, player_models)
        self.player_models_infos = Model.to_infos(player_models)

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

    def __init__(self, result_dir_path: Path, player_models: List[Model]):
        self.results_folder = ResultsFolder(result_dir_path, player_models)
        self.player_models_infos = Model.to_infos(player_models)
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
                                                 self.results_folder.models_dir,  # meta info for transcribe
                                                 self.player_models_infos)
        game_master.register(game_recorder)
        _key = InteractionsFileSaver.to_key(game_name, experiment_name, game_id)
        self._recorders[_key] = game_recorder

    def on_game_end(self, game_master: "GameMaster", game_instance: Dict):
        game_name = game_master.game_spec.game_name
        experiment_name = game_master.experiment["name"]
        game_id = game_instance["game_id"]
        _key = InteractionsFileSaver.to_key(game_name, experiment_name, game_id)
        assert _key in self._recorders, f"Recoder must be registered on_game_start, but wasn't for: {_key}"
        recorder = self._recorders.pop(_key)  # auto-remove recorder from registry
        instance_dir_path = self.results_folder.to_instance_dir_path(game_master, game_instance)
        store_json(recorder.interactions, "interactions.json", instance_dir_path)
        store_json(recorder.requests, "requests.json", instance_dir_path)
