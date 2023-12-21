""" Initial checks on models to be added to the HuggingFace local backend """

import argparse
from transformers import AutoTokenizer

def model_pre_check(args):
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
                        help="The full HuggingFace model ID/link. "
                             "Example: google/")
    parser.add_argument("-e", "--experiment_name", type=str,
                        help="Optional argument to only run a specific experiment")
    parser.add_argument("-t", "--temperature", type=float, default=0.0,
                        help="Argument to specify sampling temperature used for the whole benchmark run. Default: 0.0.")

    args = parser.parse_args()
    main(args)