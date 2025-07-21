from typing import List

from clemcore.backends import Model
from clemcore.clemgame import GameBenchmark, GameBenchmarkCallbackList
from clemcore.clemgame.runners import sequential, batchwise
from clemcore.clemgame.runners.batchwise import support_batching


def run(game_benchmark: GameBenchmark,
        player_models: List[Model],
        *,
        callbacks: GameBenchmarkCallbackList = None,
        batch_size: int | str = None):
    """
        The dispatch run method automatically checks if all models support batching:
        (1) If all models support batching, then will delegate to the batchwise runner.
        (2) If at least one of the models does not support batching, then will delegate to the sequential runner.
            The batchwise runner will use the batch_size parameter of the model is use.
            You can set the batch size via the model_spec unification mechanism, for example,
            passing a model spec to the run command like -m '{"model_name": "llama3-8b", "batch_size": 8}'.
            If batch_size is not given, then the batchwise runner determines the best batch_size automatically.
        If you want to have more control over the runner selection, then invoke them directly.

        Note: Slurk backends do not support batching, hence will run always sequentially (for now).
    Args:
        game_benchmark: The game benchmark to run.
        player_models: A list of backends.Model instances to run the game with.
        callbacks: Callbacks to be invoked during the benchmark run.
        batch_size: The batch size to be used for batch processing.

            - When greater than 1 or "auto", checks if backend support batch processing (or fails otherwise).
            - When set to "auto", then the optimal batch size will be determined automatically.
            - When batch_size is None or 1, then the runner falls back to sequential processing.
            Default: None.
    """
    callbacks = callbacks or GameBenchmarkCallbackList()
    if all(support_batching(player_model) for player_model in player_models):
        batchwise.run(game_benchmark, player_models, callbacks=callbacks, batch_size=batch_size)
    else:
        sequential.run(game_benchmark, player_models, callbacks=callbacks)
