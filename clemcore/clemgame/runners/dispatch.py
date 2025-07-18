from typing import List

from clemcore.backends import Model
from clemcore.clemgame import GameBenchmark, GameBenchmarkCallbackList
from clemcore.clemgame.runners import sequential, batchwise


def run(game_benchmark: GameBenchmark,
        player_models: List[Model],
        *,
        callbacks: GameBenchmarkCallbackList = None,
        batch_size: int | str = None):
    """ Runs game-play on the game instances of a game benchmark.
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
    if batch_size is None or batch_size == 1:
        sequential.run(game_benchmark, player_models, callbacks=callbacks)
    else:
        batchwise.run(game_benchmark, player_models, callbacks=callbacks, batch_size=batch_size)
