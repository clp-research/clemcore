from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

import evaluation.evalutils as utils
from clemgame.metrics import *
from games.codenames.constants import *
from collections import Counter
import json
from evaluation.bencheval import PlayedScoreError

REQUEST_METRICS = [METRIC_REQUEST_COUNT, METRIC_REQUEST_COUNT_PARSED, METRIC_REQUEST_COUNT_VIOLATED, METRIC_REQUEST_SUCCESS]
GAME_METRICS = [METRIC_ABORTED, METRIC_PLAYED, METRIC_SUCCESS, METRIC_LOSE, GAME_ENDED_THROUGH_ASSASSIN, NUMBER_OF_TURNS, Episode_Scores.EFFICIENCY, Episode_Scores.RECALL, Episode_Scores.NEGATIVE_RECALL]

# from evaluation.evalutils:
#   save_raw_turn_scores()
#   save_raw_episode_scores()
#   save_raw_scores()
#   build_df_episode_scores()
#   build_df_turn_scores()
#   load_interactions(game_name)
#   load_scores(game_name, results)

def load_episode_scores(results_path):
    # Get all episode scores as a pandas dataframe
    scores = utils.load_scores(path=args.results_path)
    df_episode_scores = utils.build_df_episode_scores(scores)

    # Create the PLAYED variable, inferring it from ABORTED
    df_played = score_amount_played(df_episode_scores)
    
    # We need ignore_index=True to reset the indices (otherwise we have duplicates)
    df = pd.concat([df_episode_scores, df_played], ignore_index=True)

    # dropping all rows not concerning codenames
    df = df[df['game'] == GAME_NAME].drop(columns=['game'])
    df = df.set_index(['model', 'experiment', 'episode', 'metric'])
    df = df['value'].unstack()
    df = df.mask(df[METRIC_ABORTED] == True)
    return df

def score_models(args):
    df_episode_scores = load_episode_scores(args.results_path)
    save_table(df_episode_scores, args.results_path, "raw results")

    # create and save main benchmark table
    df = make_clem_table(df_episode_scores)
    save_table(df, args.results_path, "results")

    # create and save codenames tables
    df_metrics, df_requests, df_flags = make_codenames_table(df_episode_scores)
    save_table(df_metrics, args.results_path, "codenames-specific results")
    save_table(df_requests, args.results_path, "codenames-requests")
    save_table(df_flags, args.results_path, "codenames-flags")

def score_experiments(args):
    episode_df = load_episode_scores(args.results_path)

def score_amount_played(df_episode_scores):
    if METRIC_PLAYED in df_episode_scores['metric'].unique():
        raise PlayedScoreError("Computed scores should not contain METRIC_PLAYED.")
    aux = df_episode_scores[df_episode_scores["metric"] == METRIC_ABORTED].copy()
    aux["metric"] = METRIC_PLAYED
    aux["value"] = 1 - aux["value"]
    return aux

def make_clem_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create benchmark results as a table."""
    columns = [column for column in df.columns if column in utils.MAIN_METRICS]
    df_aux = df[columns]

    # compute mean benchscore and mean played (which is binary, so a proportion)
    df_mean = (df_aux.groupby(['model'])
                  .mean(numeric_only=True))
    df_mean[METRIC_PLAYED] *= 100
    df_mean = df_mean.round(2)
    df_mean.rename(columns={METRIC_PLAYED : f'% {METRIC_PLAYED}'}, inplace=True)
    df_mean = df_mean[sorted(df_mean.columns)]

    # compute the std of benchscore
    df_std_benchscore = df_aux[BENCH_SCORE]
    df_std_benchscore = (df_std_benchscore.groupby(['model'])
                    .std(numeric_only=True)
                    .round(2))
    df_mean.insert(len(df_mean.columns), f'{BENCH_SCORE} (std)', df_std_benchscore)

    # compute clemscores and add to df
    clemscore = ((df_mean['% Played'] / 100)
                 * df_mean[BENCH_SCORE])
    df_mean.insert(0, 'clemscore', clemscore.round(2))

    return df_mean

def make_codenames_table(df: pd.DataFrame) -> pd.DataFrame:
    print(df)
    df_aux = (df.groupby(['model'])
                  .mean(numeric_only=True)
                  .round(2))
    print(df_aux.columns)

    df_game_metrics = df_aux[GAME_METRICS]
    df_requests = df_aux[REQUEST_METRICS]
    df_flags = df_aux.filter(regex='Cluegiver|Guesser')
    return df_game_metrics, df_requests, df_flags
    # also put main clem scoring into this table as well?

def save_table(df, path, table_name):
    df.to_csv(Path(path) / f'{table_name}.csv')
    df.to_html(Path(path) / f'{table_name}.html')
    print(f'\n Saved results into {path}/{table_name}.csv and .html')

def error_evaluation(results_path):
    # load interaction files
    errors = {}
    error_causes = {}
    interactions = utils.load_interactions(GAME_NAME)
    # loop through interactions
    for key, interaction in interactions.items():
        game, experiment = interaction
        players = game["players"]
        players = f"{players['Player 1']}:{players['Player 2']}"
        if not players in errors:
            errors[players] = Counter()
        if not players in error_causes:
            error_causes[players] = []
        error_causes[players].append({}) 
        turns = game["turns"]
        for turn in turns:
            for event in turn:
                action = event["action"]
                match action["type"]:
                    case Turn_logs.VALIDATION_ERROR:
                        error_type = action["content"]["type"]
                        errors[players][error_type] += 1

                        error_cause = action["content"]["utterance"]
                        if error_type not in error_causes[players][-1]:
                            error_causes[players][-1][error_type] = []
                        error_causes[players][-1][error_type].append(error_cause)

    # save errors aggregated over models as a table
    error_df = pd.DataFrame.from_dict(errors)
    error_df = error_df.transpose().fillna(0)
    print(error_df)
    save_table(error_df, results_path, "errors")
    # error_df.to_csv(f"{results_path}/errors.csv")

    # save errors with double index player-episode, put utterances in there as well?
    with open(f"{results_path}/error_causes.json", 'w') as file:
        json.dump(error_causes, file)

def main(args):
    if args.command_name == "models":
        score_models(args)
    elif args.command_name == "experiments":
        score_experiments(args)
    elif args.command_name == "errors":
        error_evaluation(args.results_path)
    else:
        print("Usage: $: python3 evaluation/codenames_eval.py [models|experiments|errors]")

if __name__ == '__main__':
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers(dest="command_name")

    model_parser = sub_parsers.add_parser("models")
    model_parser.add_argument("-p", "--results_path", type=str, default='./results',
                              help="Path to the results folder containing model scores.")
    
    experiment_parser = sub_parsers.add_parser("experiments")
    experiment_parser.add_argument("-p", "--results_path", type=str, default='./results',
                              help="Path to the results folder containing experiment scores.")

    error_parser = sub_parsers.add_parser("errors")
    error_parser.add_argument("-p", "--results_path", type=str, default='./results',
                              help="Path to the results folder containing collected interaction errors.")

    args = parser.parse_args()
    main(args)