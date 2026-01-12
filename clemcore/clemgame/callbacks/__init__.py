from pathlib import Path
from typing import Any

from clemcore.clemgame.callbacks.base import GameBenchmarkCallbackList
from clemcore.clemgame.callbacks.files import (
    ExperimentFileSaver,
    InstanceFileSaver,
    InteractionsFileSaver,
    EpisodeResultsFolderCallback,
    EpisodeResultsFolder
)


def episode_results_folder_callbacks(
        *,
        run_dir: str,
        result_dir_path: str | Path = "episode-records",
        player_model_infos: Any = None
) -> GameBenchmarkCallbackList:
    """
    Generates a list of callbacks to handle results processing and
    serialization for episodes. The callbacks are responsible for tasks such as
    managing result directories, saving configurations, and logging interaction
    data.

    The function ensures that the results folder callback runs before any file
    saving callbacks, enabling consistent and structured file organization during the process.

    Args:
        run_dir: The directory where the current run's data will be stored.
        result_dir_path: Path to the parent directory for storing result files. Defaults to "episode-records".
        player_model_infos: Additional information about the models, if any, to be stored in the experiment file
        included in the experiment and interaction data files. Defaults to None.

    Returns:
        A list of callbacks that manage result directories and save relevant interaction data.
    """
    result_dir_path = Path(result_dir_path)
    results_folder = EpisodeResultsFolder(result_dir_path, run_dir)
    # IMPORTANT: EpisodeResultsFolderCallback must run before any file savers
    return GameBenchmarkCallbackList([
        # a callback to increase the episode number in the result folder
        EpisodeResultsFolderCallback(results_folder),
        # a callback to save the instance.json
        InstanceFileSaver(results_folder),
        # a callback to save the experiment.json
        ExperimentFileSaver(results_folder, player_model_infos=player_model_infos),
        # a callback to save the interactions.json and requests.json
        InteractionsFileSaver(results_folder, player_model_infos=player_model_infos)
    ])
