import logging
from typing import List

from pettingzoo import AECEnv
from tqdm import tqdm

from clemcore.backends import Model
from clemcore.clemgame import GameBenchmark, GameBenchmarkCallbackList, GameInstanceIterator, GameStep, GameSpec
from clemcore.clemgame.envs.pettingzoo.wrappers import env_from_spec

module_logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("clemcore.run")


def run(game_spec: GameSpec,
        game_instance_iterator: GameInstanceIterator,
        player_models: List[Model],
        *,
        callbacks: GameBenchmarkCallbackList
        ):
    callbacks.on_benchmark_start(game_spec)
    game_env = env_from_spec(game_spec, game_instance_iterator, single_pass=True)
    error_count = 0
    try:
        for _ in range(2 ** 32):  # iterate through game instances
            game_env.reset()
            game_master = game_env.unwrapped().game_master
            game_instance = game_env.options["game_instance"]
            callbacks.on_game_start(game_master, game_instance)
            for agent_id in game_env.agent_iter():  # when there is no agent left, the episode is done
                context, reward, termination, truncation, info = game_env.last(observe=True)
                if termination or truncation:
                    response = None  # None actions remove the agent from the game during step(None)
                else:
                    # todo how to define mapping
                    agent = agent_mapping[agent_id]
                    response = agent(context)
                game_env.step(response)
                if response is not None:  # notify callbacks only for agent actions
                    done = len(game_env.agents) == 0
                    callbacks.on_game_step(
                        game_master,
                        game_instance,
                        GameStep(context, response, done, info)
                    )
            callbacks.on_game_end(game_master, game_instance)
    except StopIteration:
        pass
    except Exception:
        message = f"{game_spec.game_name}: Exception for instance {game_instance['game_id']} (but continue)"
        module_logger.exception(message)
        error_count += 1
    if error_count > 0:
        stdout_logger.error(
            f"{game_spec.game_name}: '{error_count}' exceptions occurred: See clembench.log for details.")
    callbacks.on_benchmark_end(game_spec)
