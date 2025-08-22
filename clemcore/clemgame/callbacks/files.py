import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, TYPE_CHECKING

from clemcore import get_version

if TYPE_CHECKING:  # to satisfy pycharm
    from clemcore.clemgame import GameMaster, GameBenchmark

from clemcore.backends import Model
from clemcore.clemgame.recorder import GameInteractionsRecorder
from clemcore.clemgame.callbacks.base import GameBenchmarkCallback
from clemcore.clemgame.resources import store_json, load_json


def to_model_results_folder(player_models: List[Model]):
    def to_descriptor(model: Model):
        return f"{model.name}-t{model.temperature}"

    model_descriptors = [to_descriptor(m) for m in player_models]
    folder_name = "--".join(model_descriptors)
    if len(player_models) <= 2:
        return folder_name
    _hash = hashlib.sha1(folder_name.encode()).hexdigest()[:8]
    return f"group-{len(player_models)}p-{_hash}"


# for pycharm: suppress could be static checks, because methods might be overwritten
# noinspection PyMethodMayBeStatic
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
        game_dir = self.to_game_dir(game_master)
        experiment_dir = self.to_experiment_dir(game_master.experiment)
        return self.to_models_dir_path() / game_dir / experiment_dir

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


class RunFileSaver(GameBenchmarkCallback):

    def __init__(self, results_folder: ResultsFolder, player_model_infos: Dict):
        self.results_folder = results_folder
        self.game_info = None
        self.benchmark_start = None
        self.num_instances = 0
        self.data = dict(clem_version=get_version(),
                         created=datetime.now().isoformat(),
                         player_models=player_model_infos,
                         games={})

        model_dir_path = self.results_folder.to_models_dir_path()
        run_file_path = Path(model_dir_path / "run.json")
        if run_file_path.exists():
            self.data = load_json(str(run_file_path))  # keep already stored values
        else:
            store_json(self.data, "run.json", model_dir_path)  # create file

    def on_benchmark_start(self, game_benchmark: "GameBenchmark"):
        self.benchmark_start = datetime.now()
        self.game_info = dict(game_path=game_benchmark.game_path, benchmark_start=self.benchmark_start.isoformat())
        self.data["games"][game_benchmark.game_name] = self.game_info
        store_json(self.data, "run.json", self.results_folder.to_models_dir_path())  # overwrite

    def on_game_start(self, game_master: "GameMaster", game_instance: Dict):
        self.num_instances += 1  # the instance iterator is not necessarily yet initialized, so we count here

    def on_benchmark_end(self, game_benchmark: "GameBenchmark"):
        benchmark_end = datetime.now()
        benchmark_duration = benchmark_end - self.benchmark_start
        self.game_info["benchmark_end"] = benchmark_end.isoformat()
        self.game_info["duration"] = str(benchmark_duration)
        self.game_info["duration_seconds"] = benchmark_duration.total_seconds()
        self.game_info["num_instances"] = self.num_instances
        store_json(self.data, "run.json", self.results_folder.to_models_dir_path())  # overwrite
        self.num_instances = 0
        self.game_info = None


class InstanceFileSaver(GameBenchmarkCallback):
    def __init__(self, results_folder: ResultsFolder):
        self.results_folder = results_folder

    def on_game_start(self, game_master: "GameMaster", game_instance: Dict):
        instance_dir_path = self.results_folder.to_instance_dir_path(game_master, game_instance)
        store_json(game_instance, "instance.json", instance_dir_path)


class ExperimentFileSaver(GameBenchmarkCallback):

    def __init__(self, results_folder: ResultsFolder, player_model_infos: Dict):
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

    def __init__(self, results_folder: ResultsFolder, player_model_infos: Dict):
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
        assert _key in self._recorders, f"Recorder must be registered on_game_start, but wasn't for: {_key}"
        recorder = self._recorders.pop(_key)  # auto-remove recorder from registry
        self._store_files(recorder, game_master, game_instance)

    def _store_files(self, recorder, game_master, game_instance):
        instance_dir_path = self.results_folder.to_instance_dir_path(game_master, game_instance)
        store_json(recorder.interactions, "interactions.json", instance_dir_path)
        store_json(recorder.requests, "requests.json", instance_dir_path)


class ImageFileSaver(GameBenchmarkCallback):
    def __init__(self, results_folder: ResultsFolder):
        self.results_folder = results_folder

    def on_game_end(self, game_master: "GameMaster", game_instance: Dict):
        game_dir = Path(game_master.game_spec.game_name)
        local_images_dir = game_dir / "images"

        if not local_images_dir.exists() or not local_images_dir.is_dir():
            return

        instance_dir_path = self.results_folder.to_instance_dir_path(game_master, game_instance)
        results_images_dir = instance_dir_path / "images"
        results_images_dir.mkdir(parents=True, exist_ok=True)

        for image_file in local_images_dir.glob("*"):
            if image_file.is_file():
                shutil.move(str(image_file), str(results_images_dir / image_file.name))

        try:
            local_images_dir.rmdir()
        except OSError:
            pass
