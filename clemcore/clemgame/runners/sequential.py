import logging
from typing import List

from tqdm import tqdm

from clemcore.backends import Model
from clemcore.clemgame import GameBenchmark, GameBenchmarkCallbackList, GameInstanceIterator

module_logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("clemcore.run")


def run(game_benchmark: GameBenchmark,
        game_instance_iterator: GameInstanceIterator,
        player_models: List[Model],
        *,
        callbacks: GameBenchmarkCallbackList):
    callbacks.on_benchmark_start(game_benchmark)
    error_count = 0
    for experiment, game_instance in tqdm(game_instance_iterator, desc="Playing game instances"):
        try:
            game_master = game_benchmark.create_game_master(experiment, player_models)
            callbacks.on_game_start(game_master, game_instance)
            game_master.setup(**game_instance)
            game_master.play()
            callbacks.on_game_end(game_master, game_instance)
        except Exception:  # continue with other instances if something goes wrong
            message = f"{game_benchmark.game_name}: Exception for instance {game_instance['game_id']} (but continue)"
            module_logger.exception(message)
            error_count += 1
    if error_count > 0:
        stdout_logger.error(
            f"{game_benchmark.game_name}: '{error_count}' exceptions occurred: See clembench.log for details.")
    callbacks.on_benchmark_end(game_benchmark)
