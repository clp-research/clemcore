import logging
from typing import List

from clemcore.backends import Model, BatchGenerativeModel
from clemcore.clemgame import GameBenchmark, GameBenchmarkCallbackList, GameInstanceIterator

stdout_logger = logging.getLogger("clemcore.run")


def run(game_benchmark: GameBenchmark,
        game_instance_iterator: GameInstanceIterator,
        player_models: List[Model | BatchGenerativeModel],
        *,
        callbacks: GameBenchmarkCallbackList = None,
        batch_size: int = 1
        ):
    """
        The dispatch run method checks if batchwise processing is possible:

        - If (a) all models support batching and (b) batch size is >1, then will delegate to the batchwise runner.
        - Otherwise, will delegate to the sequential runner.

        If you want to have more control over the runner selection, then invoke them directly.

        Note: Slurk backends do not support batching, hence will run always sequentially (for now).
    Args:
        game_benchmark: The game benchmark to run, that is, a factory to create the proper game master.
        game_instance_iterator: An iterator over the game instances to be played.
        player_models: A list of backends.Model instances to run the game with.
        callbacks: Callbacks to be invoked during the benchmark run.
        batch_size: The batch size to use (default: 1).
    """
    callbacks = callbacks or GameBenchmarkCallbackList()
    if batch_size > 1 and Model.all_support_batching(player_models):
        from clemcore.clemgame.runners import batchwise  # lazy import
        stdout_logger.info("Start batchwise runner for %s with models=[%s]  (batch_size=%s)",
                           game_benchmark.game_name,
                           ",".join(player_model.name for player_model in player_models),
                           batch_size)
        batchwise.run(game_benchmark, game_instance_iterator, player_models, callbacks=callbacks, batch_size=batch_size)
    else:
        from clemcore.clemgame.runners import sequential  # lazy import
        if not Model.all_support_batching(player_models):
            stdout_logger.info("Fallback to sequential because not all models support batching: "
                               "models=%s, support=%s", player_models,
                               [model.supports_batching() for model in player_models])
        stdout_logger.info("Start sequential runner for %s with models=[%s] (batch_size=%s)",
                           game_benchmark.game_name,
                           ",".join(player_model.name for player_model in player_models),
                           batch_size)
        sequential.run(game_benchmark, game_instance_iterator, player_models, callbacks=callbacks)
