import argparse

from clemgame import benchmark

"""
    Use good old argparse to run the commands.
    
    To list available games: 
    $> python3 scripts/cli.py ls
    
    To run all games:
    $> python3 scripts/cli.py [-m "mock"] run all
    
    To run a specific game:
    $> python3 scripts/cli.py [-m "mock"] run privateshared
    
    To score all games:
    $> python3 scripts/cli.py score all
    
    To score a specific game:
    $> python3 scripts/cli.py score privateshared
    
    To score all games:
    $> python3 scripts/cli.py transcribe all
    
    To score a specific game:
    $> python3 scripts/cli.py transcribe privateshared
"""


def main(args):
    if args.command_name == "ls":
        benchmark.list_games()
    if args.command_name == "run":
        benchmark.run(args.game_name,
                      temperature=args.temperature,
                      model_name=args.model_name,
                      experiment_name=args.experiment_name)
    if args.command_name == "score":
        benchmark.score(args.game_name,
                        experiment_name=args.experiment_name)
    if args.command_name == "transcribe":
        benchmark.transcripts(args.game_name,
                              experiment_name=args.experiment_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str,
                        help="Assumes model names supported by the implemented backends."
                             " Use 'mock--mock' to avoid using any backend."
                             " Note: '--' to configure two dialogue partners e.g. gpt-3.5-turbo--gpt-3.5-turbo."
                             " For single-player games like private-shared only provide one model e.g. gpt-3.5-turbo."
                             " When a dialog pair is given for a single-player game, then an error is thrown."
                             " When this option is not given, then the dialogue partners configured in the experiment"
                             " are used."
                             " Default: None.")
    parser.add_argument("-e", "--experiment_name", type=str,
                        help="Optional argument to only run a specific experiment")
    parser.add_argument("-t", "--temperature", type=float, default=0.0,
                        help="Argument to specify sampling temperature used for the whole benchmark run. Default: 0.0.")
    sub_parsers = parser.add_subparsers(dest="command_name")
    sub_parsers.add_parser("ls")
    run_parser = sub_parsers.add_parser("run")
    run_parser.add_argument("game_name", help="A specific game name (see ls) or 'all'."
                                              " Important: 'all' only allows self-play for now. For this mode pass"
                                              " only single model names e.g. model-a and then this will automatically"
                                              " be expanded to model-a--model-a for multi-player games.")
    run_parser = sub_parsers.add_parser("score")
    run_parser.add_argument("game_name", help="A specific game name (see ls) or 'all'")
    run_parser = sub_parsers.add_parser("transcribe")
    run_parser.add_argument("game_name", help="A specific game name (see ls) or 'all'")
    args = parser.parse_args()
    main(args)
