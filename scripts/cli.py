import argparse

from clemgame import benchmark

"""
    Use good old argparse to run the commands.
    
    To list available games: 
    $> python3 scripts/cli.py ls
    
    To run a specific game with a single player:
    $> python3 scripts/cli.py run -g privateshared -m mock
    
    To run a specific game with a two players:
    $> python3 scripts/cli.py run -g taboo -m mock mock
    
    If the game supports model expansion (using the single specified model for all players):
    $> python3 scripts/cli.py run -g taboo -m mock
    
    To score all games:
    $> python3 scripts/cli.py score
    
    To score a specific game:
    $> python3 scripts/cli.py score -g privateshared
    
    To score all games:
    $> python3 scripts/cli.py transcribe
    
    To score a specific game:
    $> python3 scripts/cli.py transcribe -g privateshared
"""


def main(args):
    if args.command_name == "ls":
        benchmark.list_games()
    if args.command_name == "run":
        benchmark.run(args.game,
                      temperature=args.temperature,
                      models=args.models,
                      experiment_name=args.experiment_name)
    if args.command_name == "score":
        benchmark.score(args.game, experiment_name=args.experiment_name)
    if args.command_name == "transcribe":
        benchmark.transcripts(args.game, experiment_name=args.experiment_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest="command_name")
    sub_parsers.add_parser("ls")

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
    run_parser.add_argument("-t", "--temperature", type=float, default=0.0,
                            help="Argument to specify sampling temperature for the models. Default: 0.0.")
    run_parser.add_argument("-e", "--experiment_name", type=str,
                            help="Optional argument to only run a specific experiment")
    run_parser.add_argument("-g", "--game", type=str,
                            required=True, help="A specific game name (see ls).")

    score_parser = sub_parsers.add_parser("score")
    score_parser.add_argument("-e", "--experiment_name", type=str,
                              help="Optional argument to only run a specific experiment")
    score_parser.add_argument("-g", "--game", type=str,
                              help="A specific game name (see ls).", default="all")

    transcribe_parser = sub_parsers.add_parser("transcribe")
    transcribe_parser.add_argument("-e", "--experiment_name", type=str,
                                   help="Optional argument to only run a specific experiment")
    transcribe_parser.add_argument("-g", "--game", type=str,
                                   help="A specific game name (see ls).", default="all")

    args = parser.parse_args()
    main(args)
