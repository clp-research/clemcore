""" Main entry point """
import textwrap
from typing import List, Dict, Union
import os.path
import logging
import logging.config
import yaml
from datetime import datetime
import json

import clemcore.backends as backends
import clemcore.clemgame as clemgame
import clemcore.utils.file_utils as file_utils
from clemcore.clemgame import GameSpec

BANNER = \
    r"""
      _                _                     _     
     | |              | |                   | |    
  ___| | ___ _ __ ___ | |__   ___ _ __   ___| |__  
 / __| |/ _ \ '_ ` _ \| '_ \ / _ \ '_ \ / __| '_ \ 
| (__| |  __/ | | | | | |_) |  __/ | | | (__| | | |
 \___|_|\___|_| |_| |_|_.__/ \___|_| |_|\___|_| |_|
"""  # doom font, thanks to http://patorjk.com/software/taag/

print(BANNER)

# Configure logging
with open(os.path.join(file_utils.clemcore_root(), "utils", "logging.yaml")) as f:
    conf = yaml.safe_load(f)
    # adapt path to logging relative to project root
    log_fn = conf["handlers"]["file_handler"]["filename"]
    log_fn = os.path.join(file_utils.project_root(), log_fn)
    conf["handlers"]["file_handler"]["filename"] = log_fn
    logging.config.dictConfig(conf)

logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("clemcore.run")

# look for custom user-defined models before loading the base registry
backends.load_custom_model_registry()
backends.load_model_registry()

# load available games
clemgame.load_custom_game_registry()
clemgame.load_game_registry()


def list_games():
    """List all games specified in the game registries.
    Only loads those for which master.py can be found in the specified path.
    See game registry doc for more infos (TODO: add link)
    TODO: add filtering options to see only specific games
    """
    stdout_logger.info("Listing all available games")
    for game in clemgame.game_registry:
        game_name = f'{game["game_name"]}:\n'
        preferred_width = 70
        wrapper = textwrap.TextWrapper(initial_indent="\t", width=preferred_width,
                                       subsequent_indent="\t")
        print(game_name, wrapper.fill(game["description"]))


def run(game: Union[str, Dict, GameSpec], model_specs: List[backends.ModelSpec], gen_args: Dict,
        experiment_name: str = None, instances_name: str = None, results_dir: str = None):
    """Run specific model/models with a specified clemgame.
    Args:
        game: Name of the game, matching the game's name in the game registry, OR GameSpec-like dict, OR GameSpec.
        model_specs: A list of backends.ModelSpec instances for the player models to run the game with.
        gen_args: Text generation parameters for the backend; output length and temperature are implemented for the
            majority of model backends.
        experiment_name: Name of the experiment to run. Corresponds to the experiment key in the instances JSON file.
        instances_name: Name of the instances JSON file to use for this benchmark run.
        results_dir: Path to the results directory in which to store the episode records.
    """
    try:
        player_models = []
        for model_spec in model_specs:
            model = backends.get_model_for(model_spec)
            model.set_gen_args(**gen_args)  # todo make this somehow available in generate method?
            player_models.append(model)

        game_specs = clemgame.select_game(game)
        print("Matched game specs in registry:", " ".join([game_spec.game_name for game_spec in game_specs]))
        for game_spec in game_specs:
            game_benchmark = clemgame.load_game(game_spec, instances_name=instances_name)
            logger.info(
                f'Running {game_spec["game_name"]} (models={player_models if player_models is not None else "see experiment configs"})')
            stdout_logger.info(f"Running game {game_spec['game_name']}")
            if experiment_name:  # leaving this as-is for now, needs discussion conclusions
                logger.info("Only running experiment: %s", experiment_name)
                game_benchmark.filter_experiment.append(experiment_name)
            time_start = datetime.now()
            game_benchmark.run(player_models=player_models, results_dir=results_dir)
            time_end = datetime.now()
            logger.info(f'Running {game_spec["game_name"]} took {str(time_end - time_start)}')

    except Exception as e:
        stdout_logger.exception(e)
        logger.error(e, exc_info=True)


def score(game_name: str, experiment_name: str = None, results_dir: str = None):
    """Calculate scores from a game benchmark run's records and store score files.
    Args:
        game_name: Name of the game, matching the game's name in the game registry.
        experiment_name: Name of the experiment to score. Corresponds to the experiment directory in each player pair
            subdirectory in the results directory.
        results_dir: Path to the results directory in which the benchmark records are stored.
    """
    logger.info(f"Scoring game {game_name}")
    stdout_logger.info(f"Scoring game {game_name}")

    if experiment_name:
        logger.info("Only scoring experiment: %s", experiment_name)
    game_specs = clemgame.select_game(game_name)
    for game_spec in game_specs:
        try:
            game = clemgame.load_game(game_spec, do_setup=False)
            if experiment_name:
                game.filter_experiment.append(experiment_name)
            time_start = datetime.now()
            game.compute_scores(results_dir)
            time_end = datetime.now()
            logger.info(f"Scoring {game.game_name} took {str(time_end - time_start)}")
        except Exception as e:
            stdout_logger.exception(e)
            logger.error(e, exc_info=True)


def transcripts(game_name: str, experiment_name: str = None, results_dir: str = None):
    """Create episode transcripts from a game benchmark run's records and store transcript files.
    Args:
        game_name: Name of the game, matching the game's name in the game registry.
        experiment_name: Name of the experiment to score. Corresponds to the experiment directory in each player pair
            subdirectory in the results directory.
        results_dir: Path to the results directory in which the benchmark records are stored.
    """
    logger.info(f"Transcribing game {game_name}")
    stdout_logger.info(f"Transcribing game {game_name}")
    if experiment_name:
        logger.info("Only transcribing experiment: %s", experiment_name)
    game_specs = clemgame.select_game(game_name)
    for game_spec in game_specs:
        try:
            game = clemgame.load_game(game_spec, do_setup=False)
            if experiment_name:
                game.filter_experiment.append(experiment_name)
            time_start = datetime.now()
            game.build_transcripts(results_dir)
            time_end = datetime.now()
            logger.info(f"Building transcripts for {game.game_name} took {str(time_end - time_start)}")
        except Exception as e:
            stdout_logger.exception(e)
            logger.error(e, exc_info=True)
