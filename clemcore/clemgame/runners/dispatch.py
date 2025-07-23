from typing import List

from clemcore.backends import Model
from clemcore.clemgame import GameBenchmark, GameBenchmarkCallbackList
from clemcore.clemgame.runners import sequential, batchwise


def run(game_benchmark: GameBenchmark,
        player_models: List[Model],
        *,
        callbacks: GameBenchmarkCallbackList = None
        ):
    """
        The dispatch run method automatically checks if all models support batching:
        (1) If all models support batching, then will delegate to the batchwise runner.
        (2) If at least one of the models does not support batching, then will delegate to the sequential runner.

        If you want to have more control over the runner selection, then invoke them directly.

        Note: Slurk backends do not support batching, hence will run always sequentially (for now).
    Args:
        game_benchmark: The game benchmark to run.
        player_models: A list of backends.Model instances to run the game with.
        callbacks: Callbacks to be invoked during the benchmark run.
    """
    callbacks = callbacks or GameBenchmarkCallbackList()
    if Model.all_support_batching(player_models):
        batchwise.run(game_benchmark, player_models, callbacks=callbacks)
    else:
        sequential.run(game_benchmark, player_models, callbacks=callbacks)
