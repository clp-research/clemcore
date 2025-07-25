import logging
from typing import List

from clemcore.backends import Model
from clemcore.clemgame import GameBenchmark, GameBenchmarkCallbackList
from clemcore.clemgame.runners import sequential

stdout_logger = logging.getLogger("clemcore.run")


def run(game_benchmark: GameBenchmark,
        player_models: List[Model],
        *,
        callbacks: GameBenchmarkCallbackList = None,
        force_sequential: bool = False
        ):
    """
        The dispatch run method automatically checks if all models support batching:
        (1) If all models support batching, then will delegate to the batchwise runner.
        (2) If at least one of the models does not support batching, then will delegate to the sequential runner.

        If you want to have more control over the runner selection, then invoke them directly.

        Note: Slurk backends do not support batching, hence will run always sequentially (for now).
    Args:
        force_sequential: Force to use the sequential runner.
        game_benchmark: The game benchmark to run.
        player_models: A list of backends.Model instances to run the game with.
        callbacks: Callbacks to be invoked during the benchmark run.
    """
    callbacks = callbacks or GameBenchmarkCallbackList()
    if force_sequential:
        stdout_logger.info("Start sequential runner for %s with models=[%s] (forced)",
                           game_benchmark.game_name,
                           ",".join(player_model.name for player_model in player_models))
        sequential.run(game_benchmark, player_models, callbacks=callbacks)
    elif not Model.all_support_batching(player_models):
        stdout_logger.info("Not all models support batching: models=%s, support=%s", player_models,
                           [model.supports_batching() for model in player_models])
        stdout_logger.info("Start sequential runner (fallback)")
        sequential.run(game_benchmark, player_models, callbacks=callbacks)
    else:
        stdout_logger.info("Start batchwise runner for %s with models=[%s]",
                           game_benchmark.game_name,
                           ",".join(player_model.name for player_model in player_models))
        from clemcore.clemgame.runners import batchwise  # requires torch (for now)
        batchwise.run(game_benchmark, player_models, callbacks=callbacks)
