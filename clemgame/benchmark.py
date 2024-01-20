""" Main entry point """
from typing import List, Dict

import backends
import clemgame

from datetime import datetime

from clemgame.clemgame import load_benchmarks, load_benchmark

logger = clemgame.get_logger(__name__)
stdout_logger = clemgame.get_logger("benchmark.run")

backends.init_model_registry()


def list_games():
    stdout_logger.info("Listing benchmark games:")
    games_list = load_benchmarks(do_setup=False)
    if not games_list:
        stdout_logger.info(" No games found. You can create a new game module in a sibling 'games' directory.")
    games_list = sorted(games_list, key=lambda gb: gb.name)
    for game in games_list:
        stdout_logger.info(" Game: %s -> %s", game.name, game.get_description())


def run(game_name: str, model_specs: List[backends.ModelSpec] = None, experiment_name: str = None):
    if experiment_name:
        logger.info("Only running experiment: %s", experiment_name)
    try:
        player_backends = [backends.get_backend_for(model_spec) for model_spec in model_specs]
        benchmark = load_benchmark(game_name)
        logger.info("Running benchmark for: %s (backends=%s)", game_name,
                    player_backends if player_backends is not None else "see experiment configs")
        if experiment_name:
            benchmark.filter_experiment.append(experiment_name)
        time_start = datetime.now()
        benchmark.run(player_backends=player_backends)
        time_end = datetime.now()
        logger.info(f"Run {benchmark.name} took {str(time_end - time_start)}")
    except Exception as e:
        logger.error(e, exc_info=True)


def score(game_name: str, experiment_name: str = None):
    logger.info("Scoring benchmark for: %s", game_name)
    if experiment_name:
        logger.info("Only scoring experiment: %s", experiment_name)
    if game_name == "all":
        games_list = load_benchmarks(do_setup=False)
    else:
        games_list = [load_benchmark(game_name, do_setup=False)]
    total_games = len(games_list)
    for idx, benchmark in enumerate(games_list):
        try:
            if experiment_name:
                benchmark.filter_experiment.append(experiment_name)
            stdout_logger.info(f"Score game {idx + 1} of {total_games}: {benchmark.name}")
            time_start = datetime.now()
            benchmark.compute_scores()
            time_end = datetime.now()
            logger.info(f"Score {benchmark.name} took {str(time_end - time_start)}")
        except Exception as e:
            stdout_logger.exception(e)
            logger.error(e, exc_info=True)


def transcripts(game_name: str, experiment_name: str = None):
    logger.info("Building benchmark transcripts for: %s", game_name)
    if experiment_name:
        logger.info("Only transcribe experiment: %s", experiment_name)
    if game_name == "all":
        games_list = load_benchmarks(do_setup=False)
    else:
        games_list = [load_benchmark(game_name, do_setup=False)]
    total_games = len(games_list)
    for idx, benchmark in enumerate(games_list):
        try:
            if experiment_name:
                benchmark.filter_experiment.append(experiment_name)
            stdout_logger.info(f"Transcribe game {idx + 1} of {total_games}: {benchmark.name}")
            time_start = datetime.now()
            benchmark.build_transcripts()
            time_end = datetime.now()
            logger.info(f"Building transcripts {benchmark.name} took {str(time_end - time_start)}")
        except Exception as e:
            stdout_logger.exception(e)
            logger.error(e, exc_info=True)
