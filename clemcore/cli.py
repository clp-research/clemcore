import argparse
import textwrap
import logging
from datetime import datetime
from typing import List, Dict, Union

import clemcore.backends as backends
from clemcore.backends import ModelRegistry, BackendRegistry
from clemcore.clemgame import GameRegistry, GameSpec
from clemcore.clemgame import benchmark
from clemcore import clemeval

logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("clemcore.cli")


def list_backends(verbose: bool):
    """List all models specified in the models registries."""
    print("Listing all supported backends (use -v option to see full file path)")
    backend_registry = BackendRegistry.from_packaged_and_cwd_files()
    if not backend_registry:
        print("No registered backends found")
        return
    print(f"Found '{len(backend_registry)}' supported backends.")
    print("Then you can use models that specify one of the following backends:")
    wrapper = textwrap.TextWrapper(initial_indent="\t", width=70, subsequent_indent="\t")
    for backend_file in backend_registry:
        print(f'{backend_file["backend"]} '
              f'({backend_file["lookup_source"]})')
        if verbose:
            print(wrapper.fill("\nFull Path: " + backend_file["file_path"]))


def list_models(verbose: bool):
    """List all models specified in the models registries."""
    print("Listing all available models by name (use -v option to see the whole specs)")
    model_registry = ModelRegistry.from_packaged_and_cwd_files()
    if not model_registry:
        print("No registered models found")
        return
    print(f"Found '{len(model_registry)}' registered model specs:")
    wrapper = textwrap.TextWrapper(initial_indent="\t", width=70, subsequent_indent="\t")
    for model_spec in model_registry:
        print(f'{model_spec["model_name"]} '
              f'-> {model_spec["backend"]} '
              f'({model_spec["lookup_source"]})')
        if verbose:
            print(wrapper.fill("\nModelSpec: " + model_spec.to_string()))


def list_games(game_selector: str, verbose: bool):
    """List all games specified in the game registries.
    Only loads those for which master.py can be found in the specified path.
    See game registry doc for more infos (TODO: add link)
    TODO: add filtering options to see only specific games
    """
    print("Listing all available games (use -v option to see the whole specs)")
    game_registry = GameRegistry.from_directories_and_cwd_files()
    if not game_registry:
        print("No clemgames found.")
        return
    if game_selector != "all":
        game_selector = GameSpec.from_string(game_selector)
    game_specs = game_registry.get_game_specs_that_unify_with(game_selector, verbose=False)
    print(f"Found '{len(game_specs)}' game specs that match the game_selector='{game_selector}'")
    wrapper = textwrap.TextWrapper(initial_indent="\t", width=70, subsequent_indent="\t")
    for game_spec in game_specs:
        game_name = f'{game_spec["game_name"]}:\n'
        if verbose:
            print(game_name,
                  wrapper.fill(game_spec["description"]), "\n",
                  wrapper.fill("GameSpec: " + game_spec.to_string()),
                  )
        else:
            print(game_name, wrapper.fill(game_spec["description"]))


def run(game_selector: Union[str, Dict, GameSpec], model_selectors: List[backends.ModelSpec],
        gen_args: Dict, experiment_name: str = None, instances_name: str = None, results_dir: str = None):
    """Run specific model/models with a specified clemgame.
    Args:
        game_selector: Name of the game, matching the game's name in the game registry, OR GameSpec-like dict, OR GameSpec.
        model_selectors: One or two selectors for the models that are supposed to play the games.
        gen_args: Text generation parameters for the backend; output length and temperature are implemented for the
            majority of model backends.
        experiment_name: Name of the experiment to run. Corresponds to the experiment key in the instances JSON file.
        instances_name: Name of the instances JSON file to use for this benchmark run.
        results_dir: Path to the results directory in which to store the episode records.
    """
    try:
        # check games first
        game_registry = GameRegistry.from_directories_and_cwd_files()
        game_specs = game_registry.get_game_specs_that_unify_with(game_selector)  # throws error when nothing unifies
        # check models are available
        model_registry = ModelRegistry.from_packaged_and_cwd_files()
        unified_model_specs = []
        for model_selector in model_selectors:
            unified_model_spec = model_registry.get_first_model_spec_that_unify_with(model_selector)
            logger.info(f"Found registered model spec that unifies with {model_selector.to_string()} "
                        f"-> {unified_model_spec}")
            unified_model_specs.append(unified_model_spec)
        # check backends are available
        backend_registry = BackendRegistry.from_packaged_and_cwd_files()
        for unified_model_spec in unified_model_specs:
            backend_selector = unified_model_spec.backend
            if not backend_registry.is_supported(backend_selector):
                raise ValueError(f"Specified model backend '{backend_selector}' not found in backend registry.")
            logger.info(f"Found registry entry for backend {backend_selector} "
                        f"-> {backend_registry.get_first_file_matching(backend_selector)}")
        # ready to rumble, do the heavy lifting only now, that is, loading the additional modules
        player_models = []
        for unified_model_spec in unified_model_specs:
            logger.info(f"Dynamically import backend {unified_model_spec.backend}")
            backend = backend_registry.get_backend_for(unified_model_spec.backend)
            model = backend.get_model_for(unified_model_spec)
            model.set_gen_args(**gen_args)  # todo make this somehow available in generate method?
            logger.info(f"Successfully loaded {unified_model_spec.model_name} model")
            player_models.append(model)

        for game_spec in game_specs:
            with benchmark.load_from_spec(game_spec, instances_name=instances_name) as game_benchmark:
                logger.info(
                    f'Running {game_spec["game_name"]} '
                    f'(models={player_models if player_models is not None else "see experiment configs"})')
                stdout_logger.info(f"Running game {game_spec['game_name']}")
                if experiment_name:  # leaving this as-is for now, needs discussion conclusions
                    logger.info("Only running experiment: %s", experiment_name)
                    game_benchmark.filter_experiment.append(experiment_name)
                time_start = datetime.now()
                game_benchmark.run(player_models=player_models, results_dir=results_dir)
                try:
                    stdout_logger.info(f"Scoring game {game_spec['game_name']}")
                    game_benchmark.compute_scores(results_dir)
                except Exception as e:
                    stdout_logger.info(f"There was a problem during scoring. See clembench.log for details.")
                    logger.error(e, exc_info=True)
                time_end = datetime.now()
                logger.info(f'Running {game_spec["game_name"]} took {str(time_end - time_start)}')

    except Exception as e:
        stdout_logger.exception(e)
        logger.error(e, exc_info=True)


def score(game_selector: Union[str, Dict, GameSpec], experiment_name: str = None, results_dir: str = None):
    """Calculate scores from a game benchmark run's records and store score files.
    Args:
        game_selector: Name of the game, matching the game's name in the game registry, OR GameSpec-like dict, OR GameSpec.
        experiment_name: Name of the experiment to score. Corresponds to the experiment directory in each player pair
            subdirectory in the results directory.
        results_dir: Path to the results directory in which the benchmark records are stored.
    """
    logger.info(f"Scoring game {game_selector}")
    stdout_logger.info(f"Scoring game {game_selector}")

    if experiment_name:
        logger.info("Only scoring experiment: %s", experiment_name)

    game_registry = GameRegistry.from_directories_and_cwd_files()
    game_specs = game_registry.get_game_specs_that_unify_with(game_selector)
    for game_spec in game_specs:
        try:
            with benchmark.load_from_spec(game_spec, do_setup=False) as game_benchmark:
                if experiment_name:
                    game_benchmark.filter_experiment.append(experiment_name)
                time_start = datetime.now()
                game_benchmark.compute_scores(results_dir)
                time_end = datetime.now()
                logger.info(f"Scoring {game_benchmark.game_name} took {str(time_end - time_start)}")
        except Exception as e:
            stdout_logger.exception(e)
            logger.error(e, exc_info=True)


def transcripts(game_selector: Union[str, Dict, GameSpec], experiment_name: str = None, results_dir: str = None):
    """Create episode transcripts from a game benchmark run's records and store transcript files.
    Args:
        game_selector: Name of the game, matching the game's name in the game registry, OR GameSpec-like dict, OR GameSpec.
        experiment_name: Name of the experiment to score. Corresponds to the experiment directory in each player pair
            subdirectory in the results directory.
        results_dir: Path to the results directory in which the benchmark records are stored.
    """
    logger.info(f"Transcribing game {game_selector}")
    stdout_logger.info(f"Transcribing game {game_selector}")
    if experiment_name:
        logger.info("Only transcribing experiment: %s", experiment_name)

    game_registry = GameRegistry.from_directories_and_cwd_files()
    game_specs = game_registry.get_game_specs_that_unify_with(game_selector)
    for game_spec in game_specs:
        try:
            with benchmark.load_from_spec(game_spec, do_setup=False) as game_benchmark:
                if experiment_name:
                    game_benchmark.filter_experiment.append(experiment_name)
                time_start = datetime.now()
                game_benchmark.build_transcripts(results_dir)
                time_end = datetime.now()
                logger.info(f"Building transcripts for {game_benchmark.game_name} took {str(time_end - time_start)}")
        except Exception as e:
            stdout_logger.exception(e)
            logger.error(e, exc_info=True)


def read_gen_args(args: argparse.Namespace):
    """Get text generation inference parameters from CLI arguments.
    Handles sampling temperature and maximum number of tokens to generate.
    Args:
        args: CLI arguments as passed via argparse.
    Returns:
        A dict with the keys 'temperature' and 'max_tokens' with the values parsed by argparse.
    """
    return dict(temperature=args.temperature, max_tokens=args.max_tokens)


def cli(args: argparse.Namespace):
    if args.command_name == "list":
        if args.mode == "games":
            list_games(args.selector, args.verbose)
        elif args.mode == "models":
            list_models(args.verbose)
        elif args.mode == "backends":
            list_backends(args.verbose)
        else:
            print(f"Cannot list {args.mode}. Choose an option documented at 'list -h'.")
    if args.command_name == "run":
        run(args.game,
            model_selectors=backends.ModelSpec.from_strings(args.models),
            gen_args=read_gen_args(args),
            experiment_name=args.experiment_name,
            instances_name=args.instances_name,
            results_dir=args.results_dir)
    if args.command_name == "score":
        score(args.game, experiment_name=args.experiment_name, results_dir=args.results_dir)
    if args.command_name == "transcribe":
        transcripts(args.game, experiment_name=args.experiment_name, results_dir=args.results_dir)
    if args.command_name == "eval":
        clemeval.perform_evaluation(args.results_dir)


"""
    Use good old argparse to run the commands.

    To list available games: 
    $> clem list [games]

    To list available models: 
    $> clem list models
    
    To list available backends: 
    $> clem list backends

    To run a specific game with a single player:
    $> clem run -g privateshared -m mock

    To run a specific game with two players:
    $> clem run -g taboo -m mock mock

    If the game supports model expansion (using the single specified model for all players):
    $> clem run -g taboo -m mock

    To score all games:
    $> clem score

    To score a specific game:
    $> clem score -g privateshared

    To transcribe all games:
    $> clem transcribe

    To transcribe a specific game:
    $> clem transcribe -g privateshared
"""


def main():
    """Main CLI handling function.

    Handles the clembench CLI commands

    - 'ls' to list available clemgames.
    - 'run' to start a benchmark run. Takes further arguments determining the clemgame to run, which experiments,
    instances and models to use, inference parameters, and where to store the benchmark records.
    - 'score' to score benchmark results. Takes further arguments determining the clemgame and which of its experiments
    to score, and where the benchmark records are located.
    - 'transcribe' to transcribe benchmark results. Takes further arguments determining the clemgame and which of its
    experiments to transcribe, and where the benchmark records are located.

    Args:
        args: CLI arguments as passed via argparse.
    """
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest="command_name")
    list_parser = sub_parsers.add_parser("list")
    list_parser.add_argument("mode", choices=["games", "models", "backends"],
                             default="games", nargs="?", type=str,
                             help="Choose to list available games, models or backends. Default: games")
    list_parser.add_argument("-v", "--verbose", action="store_true")
    list_parser.add_argument("-s", "--selector", type=str, default="all")

    run_parser = sub_parsers.add_parser("run", formatter_class=argparse.RawTextHelpFormatter)
    run_parser.add_argument("-m", "--models", type=str, nargs="*",
                            help="""Assumes model names supported by the implemented backends.

      To run a specific game with a single player:
      $> python3 scripts/cli.py run -g privateshared -m mock

      To run a specific game with a two players:
      $> python3 scripts/cli.py run -g taboo -m mock mock

      If the game supports model expansion (using the single specified model for all players):
      $> python3 scripts/cli.py run -g taboo -m mock

      When this option is not given, then the dialogue partners configured in the experiment are used. 
      Default: None.""")
    run_parser.add_argument("-e", "--experiment_name", type=str,
                            help="Optional argument to only run a specific experiment")
    run_parser.add_argument("-g", "--game", type=str,
                            required=True, help="A specific game name (see ls), or a GameSpec-like JSON string object.")
    run_parser.add_argument("-t", "--temperature", type=float, default=0.0,
                            help="Argument to specify sampling temperature for the models. Default: 0.0.")
    run_parser.add_argument("-l", "--max_tokens", type=int, default=100,
                            help="Specify the maximum number of tokens to be generated per turn (except for cohere). "
                                 "Be careful with high values which might lead to exceed your API token limits."
                                 "Default: 100.")
    run_parser.add_argument("-i", "--instances_name", type=str, default=None,
                            help="The instances file name (.json suffix will be added automatically.")
    run_parser.add_argument("-r", "--results_dir", type=str, default="results",
                            help="A relative or absolute path to the results root directory. "
                                 "For example '-r results/v1.5/de‘ or '-r /absolute/path/for/results'. "
                                 "When not specified, then the results will be located in 'results'")

    score_parser = sub_parsers.add_parser("score")
    score_parser.add_argument("-e", "--experiment_name", type=str,
                              help="Optional argument to only run a specific experiment")
    score_parser.add_argument("-g", "--game", type=str,
                              help='A specific game name (see ls), a GameSpec-like JSON string object or "all" (default).',
                              default="all")
    score_parser.add_argument("-r", "--results_dir", type=str, default="results",
                              help="A relative or absolute path to the results root directory. "
                                   "For example '-r results/v1.5/de‘ or '-r /absolute/path/for/results'. "
                                   "When not specified, then the results will be located in 'results'")

    transcribe_parser = sub_parsers.add_parser("transcribe")
    transcribe_parser.add_argument("-e", "--experiment_name", type=str,
                                   help="Optional argument to only run a specific experiment")
    transcribe_parser.add_argument("-g", "--game", type=str,
                                   help='A specific game name (see ls), a GameSpec-like JSON string object or "all" (default).',
                                   default="all")
    transcribe_parser.add_argument("-r", "--results_dir", type=str, default="results",
                                   help="A relative or absolute path to the results root directory. "
                                        "For example '-r results/v1.5/de‘ or '-r /absolute/path/for/results'. "
                                        "When not specified, then the results will be located in 'results'")

    eval_parser = sub_parsers.add_parser("eval")
    eval_parser.add_argument("-r", "--results_dir", type=str, default="results",
                             help="A relative or absolute path to the results root directory. "
                                  "For example '-r results/v1.5/de‘ or '-r /absolute/path/for/results'. "
                                  "When not specified, then the results will be located in 'results'."
                                  "For evaluation, the directory must already contain the scores.")

    cli(parser.parse_args())


if __name__ == "__main__":
    main()
