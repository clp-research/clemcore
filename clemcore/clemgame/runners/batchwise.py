import logging
from typing import List

from tqdm import tqdm

from clemcore.backends import Model
from clemcore.clemgame import GameBenchmark, GameBenchmarkCallbackList, GameMaster

module_logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("clemcore.run")


def run(game_benchmark: GameBenchmark,
        player_models: List[Model],
        *,
        callbacks: GameBenchmarkCallbackList,
        batch_size: int | str):
    assert batch_size > 0 or batch_size == "auto", f"batch_size must be >0 or 'auto' but is {batch_size}"
    # If not all support batching, then this doesn't help, because the models have to wait for the slowest one
    assert all(player_model.supports_batching() for player_model in player_models), \
        "Not all player_models support batching"  # todo
    # Note: Here we already assume that both models allow batching
    callbacks.on_benchmark_start(game_benchmark)
    # Setup game_masters
    error_count = 0
    game_masters: List[GameMaster] = []
    for experiment, game_instance in tqdm(game_benchmark.game_instance_iterator, desc="Playing games"):
        try:
            game_master = game_benchmark.create_game_master(experiment, player_models)
            callbacks.on_game_start(game_master, game_instance)
            game_master.setup(**game_instance)
            game_masters.append(game_master)
        except Exception:  # continue with other instances if something goes wrong
            message = f"{game_benchmark.game_name}: Exception for instance {game_instance['game_id']} (but continue)"
            module_logger.exception(message)
            error_count += 1
    # Play # todo use torch.data Dataset
    for game_master in game_masters:
        player, context = game_master.observe()
        # group context by player
        ...
        # step game_masters
        game_master.step(...)
    callbacks.on_benchmark_end(game_benchmark)
