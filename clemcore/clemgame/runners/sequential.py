import logging
from typing import List

from tqdm import tqdm

from clemcore.backends import Model
from clemcore.clemgame import GameBenchmarkCallbackList, GameInstanceIterator, GameStep, GameBenchmark
from clemcore.clemgame.envs.pettingzoo.master import GameMasterEnv

module_logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("clemcore.run")


def run(game_benchmark: GameBenchmark,
        game_instance_iterator: GameInstanceIterator,
        player_models: List[Model],
        *,
        callbacks: GameBenchmarkCallbackList
        ):
    callbacks.on_benchmark_start(game_benchmark)
    game_env = GameMasterEnv(game_benchmark, callbacks=callbacks)
    error_count = 0
    for experiment, game_instance in tqdm(game_instance_iterator, desc="Playing game instances"):
        try:
            game_env.reset(options={
                "player_models": player_models,
                "experiment": experiment,
                "game_instance": game_instance
            })
            for model in player_models:
                model.reset()  # this is mainly to notify slurk backends; other models are state-less anyway
            for agent_id in game_env.agent_iter():  # when there is no agent left, the episode is done
                context, reward, termination, truncation, info = game_env.last(observe=True)
                if termination or truncation:
                    # None actions remove the agent from the game during step(None)
                    # This is essential to observe the final reward, e.g., for the describer, when the guesser wins
                    response = None
                else:
                    player = game_env.player_by_agent_id[agent_id]
                    response = player(context)
                game_env.step(response)
        except Exception:  # continue with other instances if something goes wrong
            message = f"{game_benchmark.game_name}: Exception for instance {game_instance['game_id']} (but continue)"
            module_logger.exception(message)
            error_count += 1
            for model in player_models:
                model.reset()
    game_env.close()
    if error_count > 0:
        stdout_logger.error(
            f"{game_benchmark.game_name}: '{error_count}' exceptions occurred: See clembench.log for details.")
    callbacks.on_benchmark_end(game_benchmark)
