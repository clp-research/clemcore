import logging
from typing import List

from tqdm import tqdm

from clemcore.backends import Model
from clemcore.clemgame import GameBenchmark, GameBenchmarkCallbackList, GameInstanceIterator, GameStep

module_logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("clemcore.run")


def run(game_benchmark: GameBenchmark,
        game_instance_iterator: GameInstanceIterator,
        player_models: List[Model],
        *,
        callbacks: GameBenchmarkCallbackList):

    # Automatic player expansion: When only a single model is given, then use this model given for each game role.
    if len(player_models) == 1 and game_spec.players > 1:
        player_models = [player_models[0]] * game_spec.players  # keeps original list untouched
    if len(player_models) != game_spec.players:
        raise ValueError(f"{game_spec.game_name} requires {game_spec.players} players, "
                         f"but {len(player_models)} were given: {[m.name for m in player_models]}")
    self.player_models: List[backends.Model] = player_models

    callbacks.on_benchmark_start(game_benchmark)
    error_count = 0
    for experiment, game_instance in tqdm(game_instance_iterator, desc="Playing game instances"):
        try:
            game_master = game_benchmark.create_game_master(experiment, player_models)
            callbacks.on_game_start(game_master, game_instance)
            game_master.setup(**game_instance)
            done = False
            while not done:
                player, context = game_master.observe()
                response = player(context)
                done, info = game_master.step(response)
                game_step = GameStep(context, response, done, info)
                callbacks.on_game_step(game_master, game_instance, game_step)
            callbacks.on_game_end(game_master, game_instance)
        except Exception:  # continue with other instances if something goes wrong
            message = f"{game_benchmark.game_name}: Exception for instance {game_instance['game_id']} (but continue)"
            module_logger.exception(message)
            error_count += 1
    if error_count > 0:
        stdout_logger.error(
            f"{game_benchmark.game_name}: '{error_count}' exceptions occurred: See clembench.log for details.")
    callbacks.on_benchmark_end(game_benchmark)
