import argparse
import sys
import textwrap
import logging
import uvicorn
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List, Dict, Union, Callable, Optional, Any

import clemcore.backends as backends
from clemcore.backends import ModelRegistry, BackendRegistry, Model, KeyRegistry
from clemcore.clemgame import GameRegistry, GameSpec, InstanceFileSaver, ExperimentFileSaver, \
    InteractionsFileSaver, GameBenchmarkCallbackList, RunFileSaver, GameInstanceIterator, ResultsFolder, \
    GameBenchmark
from clemcore import clemeval, get_version
from clemcore.clemgame.runners import dispatch
from clemcore.clemgame.transcripts.builder import build_transcripts
from clemcore.utils.string_utils import read_query_string
from clemcore.clemgame.envs.openenv.server.app import create_clemv_app

logger = logging.getLogger(__name__)  # by default also logged to console


def list_keys():
    # load key registry, then go through detected backends and show if keys are present or not
    key_registry = KeyRegistry.from_json()
    print(f"Listing all registered keys in {key_registry.key_file_path}")
    if not key_registry:
        print("No registered keys found")
        return
    print(f"The following {len(key_registry)} backends are considered [ active ] if 'api_key' is present:")
    for backend_name, key in key_registry.items():
        # overwrite key_api values to mask the secret
        status = " active " if key.has_api_key() else "inactive"
        print(f'* [{status}] {backend_name}: {key.to_json()}')
    print("")
    print("A note on how to define your own keys manually:")
    print("Create a 'key.json' in the current working directory and add {'backend_name': {'api_key':'value'}} entries.")
    print("Alternatively use: clem register key -n <backend_name> -v <api_key=value>")


def list_backends(verbose: bool = False):
    """List all backends found within the package and the current working directory."""
    print("Listing all supported backends (use -v option to see full file path)")
    backend_registry = BackendRegistry.from_packaged_and_cwd_files()
    if not backend_registry:
        print("No registered backends found")
        return
    print(f"Found '{len(backend_registry)}' supported backends.")
    key_registry = KeyRegistry.from_json()

    registered_backends = []
    unregistered_backends = []
    for backend_file in backend_registry:
        backend_name = backend_file["backend"]
        if backend_name in key_registry and not key_registry.get_key_for(backend_name).has_api_key():
            unregistered_backends.append(backend_file)
        else:
            registered_backends.append(backend_file)

    def print_backends(listing):
        wrapper = textwrap.TextWrapper(initial_indent="\t", width=70, subsequent_indent="\t")
        for backend_file in sorted(listing, key=lambda x: x["backend"]):
            print(f'* {backend_file["backend"]} '
                  f'({backend_file["lookup_source"]})')
            if verbose:
                print(wrapper.fill("\nFull Path: " + backend_file["file_path"]))

    print()
    if registered_backends:
        print("[ active ] backends:")
        print_backends(registered_backends)
        print("-> If an active backend is not functional, consider adding a respective entry to the key registry.")
        print()
    if unregistered_backends:
        print("[inactive] backends:")
        print_backends(unregistered_backends)
        print("-> To enable these, set the 'api_key' in the key registry.")
        print()
    print("A note on how to add your own backend:")
    print("Create a 'custom_api.py' in the current working directory and implement a Backend class in that file.")
    print("Then your custom backend will then be listed and usable as 'custom' backend.")


def list_models(verbose: bool = False):
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
    print("")
    print("A note on how to add a model entry manually:")
    print("Create a 'model_registry.json' in the current working directory "
          "and list {'model_name': <model_name>, 'backend': <backend_name>} entries.")
    print("Alternatively use: clem register model -n <model_name> -v <backend=backend_name>")


def list_games(game_selector: str, verbose: bool = False):
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


def experiment_filter(game: str, experiment: str, *, selected_experiment: str, game_ids: Optional[List[int]]):
    if experiment != selected_experiment:
        return []  # skip experiment
    if game_ids is None:
        return None  # allow all
    return game_ids


def run(game_selector: Union[str, Dict, GameSpec],
        model_selectors: List[backends.ModelSpec],
        *,
        gen_args: Dict,
        experiment_name: str = None,
        instances_filename: str = None,
        results_dir_path: Path = None,
        sub_selector: Callable[[str, str], List[int]] = None,
        batch_size: int = 1
        ):
    """Run specific model/models with a specified clemgame.
    Args:
        game_selector: Name of the game, matching the game's name in the game registry, OR GameSpec-like dict, OR GameSpec.
        model_selectors: One or two selectors for the models that are supposed to play the games.
        gen_args: Text generation parameters for the backend; output length and temperature are implemented for the
            majority of model backends.
        experiment_name: Name of the experiment to run. Corresponds to the experiment key in the instances JSON file.
        instances_filename: Name of the instances JSON file to use for this benchmark run.
        results_dir_path: Path to the results directory in which to store the episode records.
        sub_selector: A callable mapping from (game_name, experiment_name) tuples to lists of game instance ids.
            If a mapping returns None, then all game instances will be used.
        batch_size: A batch size to use for the run.
    """
    # check games
    game_registry = GameRegistry.from_directories_and_cwd_files()
    game_specs = game_registry.get_game_specs_that_unify_with(game_selector)  # throws error when nothing unifies

    # load models (can take some time for large local models)
    player_models = backends.load_models(model_selectors, gen_args)

    # setup reusable callbacks here once
    # we name the run directory after the participating models
    results_folder = ResultsFolder(results_dir_path, run_dir=Model.to_identifier(player_models))
    model_infos = Model.to_infos(player_models)
    callbacks = GameBenchmarkCallbackList([
        InstanceFileSaver(results_folder),
        ExperimentFileSaver(results_folder, player_model_infos=model_infos),
        InteractionsFileSaver(results_folder, player_model_infos=model_infos),
        RunFileSaver(results_folder, player_model_infos=model_infos)
    ])

    all_start = datetime.now()
    errors = []
    for game_spec in game_specs:
        try:
            # configure instance file to be used
            if instances_filename:
                game_spec.instances = instances_filename  # force the use of cli argument, when given

            if experiment_name:  # establish experiment filter, if given
                logger.info("Only running experiment: %s", experiment_name)
                if sub_selector is None:
                    sub_selector = partial(experiment_filter, selected_experiment=experiment_name, game_ids=None)
                else:
                    game_ids = sub_selector(game_spec.game_name, experiment_name)
                    sub_selector = partial(experiment_filter, selected_experiment=experiment_name, game_ids=game_ids)

            with GameBenchmark.load_from_spec(game_spec) as game_benchmark:
                time_start = datetime.now()
                logger.info(f'Running {game_spec["game_name"]} (models={player_models})')
                game_instance_iterator = GameInstanceIterator.from_game_spec(game_spec, sub_selector=sub_selector)
                game_instance_iterator.reset(verbose=True)
                dispatch.run(
                    game_benchmark,
                    game_instance_iterator,
                    player_models,
                    callbacks=callbacks,
                    batch_size=batch_size
                )
                logger.info(f"Running {game_spec['game_name']} took: %s", datetime.now() - time_start)
        except Exception as e:
            logger.exception(e)
            logger.error(e, exc_info=True)
            errors.append(e)
    logger.info("Running all benchmarks took: %s", datetime.now() - all_start)
    if errors:
        sys.exit(1)


def score(game_selector: Union[str, Dict, GameSpec], results_dir: str = None):
    """Calculate scores from a game benchmark run's records and store score files.
    Args:
        game_selector: Name of the game, matching the game's name in the game registry, OR GameSpec-like dict, OR GameSpec.
        experiment_name: Name of the experiment to score. Corresponds to the experiment directory in each player pair
            subdirectory in the results directory.
        results_dir: Path to the results directory in which the benchmark records are stored.
    """
    logger.info(f"Scoring game {game_selector}")
    errors = []
    game_registry = GameRegistry.from_directories_and_cwd_files()
    game_specs = game_registry.get_game_specs_that_unify_with(game_selector)
    for game_spec in game_specs:
        try:
            time_start = datetime.now()
            with GameBenchmark.load_from_spec(game_spec) as game_benchmark:
                game_benchmark.compute_scores(results_dir)
            logger.info(f"Scoring {game_benchmark.game_name} took: %s", datetime.now() - time_start)
        except Exception as e:
            logger.exception(e)
            errors.append(e)
    if errors:
        sys.exit(1)


def transcripts(game_selector: Union[str, Dict, GameSpec], results_dir: str = None):
    """Create episode transcripts from a game benchmark run's records and store transcript files.
    Args:
        game_selector: Name of the game, matching the game's name in the game registry, OR GameSpec-like dict, OR GameSpec.
        results_dir: Path to the results directory in which the benchmark records are stored.
    """
    logger.info(f"Transcribing game interactions that match game_selector={game_selector}")

    filter_games = []
    if game_selector != "all":
        game_registry = GameRegistry.from_directories_and_cwd_files()
        game_specs = game_registry.get_game_specs_that_unify_with(game_selector)
        filter_games = [game_spec.game_name for game_spec in game_specs]
    time_start = datetime.now()
    build_transcripts(results_dir, filter_games)
    logger.info(f"Building transcripts took: %s", datetime.now() - time_start)


def serve(game: str,
          learner_agent: str,
          env_agents: Optional[Dict[str, str]] = None,
          gen_args: Optional[Dict[str, Any]] = None,
          split: Optional[str] = None,
          single_pass: bool = False,
          host: str = "0.0.0.0",
          port: int = 5000,
          results_dir: Optional[str] = None,
          run_id: Optional[str] = None):
    logger.info(f"Starting a clem environment server for the game: {game} on {host}:{port}")
    app = create_clemv_app(
        game_name=game,
        learner_agent=learner_agent,
        env_agents=env_agents,
        game_instance_split=split,
        single_pass=single_pass,
        gen_args=gen_args,
        results_dir=results_dir,
        run_id=run_id
    )
    uvicorn.run(app, host=host, port=port)


def parse_kv(arg: str):
    if '=' not in arg:
        raise argparse.ArgumentTypeError(f"Invalid agent format: '{arg}'. Use key=value")
    return arg.split('=', 1)


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
            list_games(args.selector, verbose=args.verbose)
        elif args.mode == "models":
            list_models(verbose=args.verbose)
        elif args.mode == "backends":
            list_backends(verbose=args.verbose)
        elif args.mode == "keys":
            list_keys()
        else:
            print(f"Cannot list {args.mode}. Choose an option documented at 'list -h'.")
    if args.command_name == "register":
        if args.mode == "model":
            registry = ModelRegistry.register(args.name, reset=args.reset, **args.values)
            model_spec = registry.get_first_model_spec_that_unify_with(args.name)
            print(f"Updated model registry at {registry.get_cwd_path()} successfully: {model_spec.to_string()}")
        if args.mode == "key":
            registry = KeyRegistry.register(args.name, reset=args.reset, force_cwd=args.cwd, **args.values)
            key = registry.get_key_for(args.name)
            print(f"Updated key registry at {registry.key_file_path} successfully: {key.to_json()}")
    if args.command_name == "run":
        start = datetime.now()
        try:
            run(args.game,
                model_selectors=backends.ModelSpec.from_strings(args.models),
                gen_args=read_gen_args(args),
                experiment_name=args.experiment_name,
                instances_filename=args.instances_filename,
                results_dir_path=args.results_dir,
                batch_size=args.batch_size)
        finally:
            logger.info("clem run took: %s", datetime.now() - start)
    if args.command_name == "serve":
        serve(args.game,
              learner_agent=args.learner_agent,
              env_agents=args.env_agents,
              gen_args=args.gen_args,
              split=args.split,
              single_pass=args.single_pass,
              host=args.host,
              port=args.port,
              results_dir=args.results_dir,
              run_id=args.run_id)
    if args.command_name == "score":
        score(args.game, results_dir=args.results_dir)
    if args.command_name == "transcribe":
        transcripts(args.game, results_dir=args.results_dir)
    if args.command_name == "eval":
        clemeval.perform_evaluation(args.results_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', action='version', version=f'%(prog)s {get_version()}')
    sub_parsers = parser.add_subparsers(dest="command_name")
    list_parser = sub_parsers.add_parser("list")
    list_parser.add_argument("mode", choices=["games", "models", "backends", "keys"],
                             default="games", nargs="?", type=str,
                             help="Choose to list available games, models or backends. Default: games")
    list_parser.add_argument("-v", "--verbose", action="store_true")
    list_parser.add_argument("-s", "--selector", type=str, default="all")

    register_parser = sub_parsers.add_parser(
        "register",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Update a model or key registry entry.",
        epilog="""
Update Behavior:
  - By default, existing entries are UPDATED (merged). 
    Only the keys provided in --value will be changed; 
    unmentioned fields remain intact.
  - If --reset is used, the entry is REPLACED. 
    All existing data for that name is wiped, and only 
    the new values provided are saved.
    """
    )

    register_parser.add_argument("mode", choices=["model", "key"], type=str,
                                 help="Choose which entry type to register.")
    register_parser.add_argument("-n", "--name", type=str, required=True,
                                 help="The name of the entry to be registered. "
                                      "For 'model' entries, this is the model name, "
                                      "and for 'key' entries it is the backend name.")
    register_parser.add_argument("-v", "--values", required=True,
                                 type=read_query_string,
                                 help="Query string style values for the entry to be registered. "
                                      "Example: 'api_key=abc,base_url=localhost'.")
    register_parser.add_argument("-r", "--reset", action="store_true",
                                 help="Reset existing entries with the same name. Otherwise entries are updated.")
    register_parser.add_argument("--cwd", action="store_true",
                                 help="Create files always in current working directory.")

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
    run_parser.add_argument("-l", "--max_tokens", type=int, default=300,
                            help="Specify the maximum number of tokens to be generated per turn (except for cohere). "
                                 "Be careful with high values which might lead to exceed your API token limits."
                                 "Default: 300.")
    run_parser.add_argument("-b", "--batch_size", type=int, default=1,
                            help="The batch size for response generation, that is, "
                                 "the number of simultaneously played game instances. "
                                 "Applies to all models that support batchwise generation, "
                                 "otherwise the game instances will be played sequentially."
                                 "Default: 1 (sequential processing).")
    run_parser.add_argument("-i", "--instances_filename", type=str, default=None,
                            help="The instances file name (.json suffix will be added automatically.")
    run_parser.add_argument("-r", "--results_dir", type=Path, default="results",
                            help="A relative or absolute path to the results root directory. "
                                 "For example '-r results/v1.5/de‘ or '-r /absolute/path/for/results'. "
                                 "When not specified, then the results will be located in 'results'")

    score_parser = sub_parsers.add_parser("score")
    score_parser.add_argument("-g", "--game", type=str,
                              help='A specific game name, a GameSpec-like JSON string object or "all" (default).',
                              default="all")
    score_parser.add_argument("-r", "--results_dir", type=str, default="results",
                              help="A relative or absolute path to the results root directory. "
                                   "For example '-r results/v1.5/de‘ or '-r /absolute/path/for/results'. "
                                   "When not specified, then the results will be located in 'results'")

    transcribe_parser = sub_parsers.add_parser("transcribe")
    transcribe_parser.add_argument("-g", "--game", type=str,
                                   help='A specific game name, a GameSpec-like JSON string object or "all" (default).',
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

    serve_parser = sub_parsers.add_parser("serve")
    serve_parser.add_argument("-g", "--game",
                              type=str,
                              required=True,
                              help="A specific game name , or a GameSpec-like JSON string object.")
    serve_parser.add_argument("-l", "--learner-agent",
                              type=str,
                              default="player_0",
                              help="The player id which the learning agent is supposed to play."
                                   "Default: player_0.")
    serve_parser.add_argument("-e", "--env-agents",
                              type=read_query_string,
                              help="Fixed agents providing the training environment. "
                                   "Example: 'player_1=gpt-4o,player_2=llama3'.")
    serve_parser.add_argument("--gen-args",
                              type=read_query_string,
                              help="Sampling parameters for the models. Example: 'temperature=0.0,max_tokens=300'.")
    serve_parser.add_argument("--split",
                              type=str,
                              choices=["train", "validation"],
                              help="Specify the game instance split to use (train or validation). "
                                   "Default: None (all instances).")
    serve_parser.add_argument("--single-pass",
                              action="store_true",
                              help="If set, the environment will run in single-pass mode. "
                                   "Default: False (cycles through the games instances infinitely).")
    serve_parser.add_argument("--host",
                              type=str,
                              default="0.0.0.0",
                              help="The host to bind the server to. Default: 0.0.0.0")
    serve_parser.add_argument("--port",
                              type=int,
                              default=8000,
                              help="The port to bind the server to. Default: 8000")
    serve_parser.add_argument("-r", "--results-dir",
                              type=str,
                              default="openenv-records",
                              help="Directory to save episode interaction results. "
                                   "Default: 'openenv-records'.")
    serve_parser.add_argument("--run-id",
                              type=str,
                              help="Identifier for this run, used as subdirectory name in results-dir. "
                                   "If not provided, derived from env-agent model names (e.g., 'gpt-4o-llama3'), "
                                   "otherwise defaults to 'run'.")
    cli(parser.parse_args())


if __name__ == "__main__":
    main()
