from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import numpy as np

import evaluation.evalutils as utils
from clemgame.metrics import *
from games.codenames.constants import *
from collections import Counter
import json, copy
from evaluation.bencheval import PlayedScoreError

REQUEST_METRICS = [METRIC_REQUEST_COUNT, METRIC_REQUEST_COUNT_PARSED, METRIC_REQUEST_COUNT_VIOLATED, METRIC_REQUEST_SUCCESS]
GAME_METRICS = [METRIC_ABORTED, METRIC_PLAYED, METRIC_SUCCESS, METRIC_LOSE, GAME_ENDED_THROUGH_ASSASSIN, NUMBER_OF_TURNS, 
                Episode_Scores.EFFICIENCY, Episode_Scores.RECALL, Episode_Scores.NEGATIVE_RECALL]

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
    scores = utils.load_scores(path=results_path)
    df_episode_scores = utils.build_df_episode_scores(scores)

    # Create the PLAYED variable, inferring it from ABORTED
    df_played = score_amount_played(df_episode_scores)
    
    # We need ignore_index=True to reset the indices (otherwise we have duplicates)
    df = pd.concat([df_episode_scores, df_played], ignore_index=True)

    # dropping all rows not concerning codenames
    df = df[df['game'] == GAME_NAME].drop(columns=['game'])
    df = df.set_index(['model', 'experiment', 'episode', 'metric'])
    df = df['value'].unstack()

    # setting values NaN or 0 if game was aborted
    keep_metrics = [METRIC_ABORTED, METRIC_PLAYED, METRIC_SUCCESS, METRIC_LOSE, VARIABLE, EXPERIMENT_NAME, GAME_ENDED_THROUGH_ASSASSIN, METRIC_REQUEST_COUNT, METRIC_REQUEST_COUNT_PARSED, METRIC_REQUEST_COUNT_VIOLATED, METRIC_REQUEST_SUCCESS]
    flag_columns = [column for column in df.columns if (column.startswith('Cluegiver') or column.startswith('Guesser'))]
    keep_metrics.extend(flag_columns)
    df.loc[df[METRIC_ABORTED] == True, [column for column in df.columns if column not in keep_metrics]] = np.nan

    # re-sorting the experiments by their number
    for index, row in df.iterrows():
        df.loc[index, 'order_number'] = int(index[1][0 : index[1].index('_')])
    df = df.reset_index().set_index(['model', 'order_number', 'experiment', 'episode'])
    df.sort_index(level = 1, inplace = True)
    return df

def score_models(args):
    df_episode_scores = load_episode_scores(args.results_path)
    save_table(df_episode_scores, args.results_path, "raw results")

    # create and save main benchmark table
    df = make_clem_table(df_episode_scores)
    save_table(df, args.results_path, "results")

    # create and save codenames tables
    df_metrics, df_requests, df_flags, df_average_turn_scores = make_codenames_tables(df_episode_scores)
    save_table(df_metrics, args.results_path, "codenames-specific results")
    save_table(df_requests, args.results_path, "codenames-requests")
    save_table(df_flags, args.results_path, "codenames-flags")
    save_table(df_average_turn_scores, args.results_path, "codenames-turn scores")

def score_experiments(args):
    episode_df = load_episode_scores(args.results_path)
    df_experiments_avg = (episode_df.groupby([VARIABLE, 'model', 'experiment name'], sort=False, dropna=False)
                  .mean())
    
    save_table(df_experiments_avg, f"{args.results_path}/experiment-results", "all")
    df_experiments = episode_df.reset_index().set_index(['model', 'experiment name'])
    for variable in df_experiments[VARIABLE].unique():
        experiment_df = df_experiments[df_experiments[VARIABLE] == variable].drop(columns=['order_number', 'experiment', 'experiment variable'])
        save_table(experiment_df, f"{args.results_path}/experiment-results", variable)

        df_metrics, df_requests, df_flags, df_average_turn_scores = make_codenames_tables(experiment_df)
        save_table(df_metrics, f"{args.results_path}/experiment-results", f"{variable}-results")
        save_table(df_requests, f"{args.results_path}/experiment-results", f"{variable}-requests")
        save_table(df_flags, f"{args.results_path}/experiment-results", f"{variable}-flags")
        save_table(df_average_turn_scores, f"{args.results_path}/experiment-results", f"{variable}-turn scores")

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
    df_mean = (df_aux.groupby(['model'], sort=False)
                  .mean(numeric_only=False))  #numeric_only=True
    df_mean = df_mean.apply(pd.to_numeric)
    
    df_mean[METRIC_PLAYED] *= 100
    df_mean = df_mean.round(2)
    df_mean.rename(columns={METRIC_PLAYED : f'% {METRIC_PLAYED}'}, inplace=True)
    df_mean = df_mean[sorted(df_mean.columns)]

    # compute the std of benchscore
    df_std_benchscore = df_aux[BENCH_SCORE]
    df_std_benchscore = (df_std_benchscore.groupby(['model'], sort=False)
                    .std(numeric_only=False)
                    .round(2))
    df_mean.insert(len(df_mean.columns), f'{BENCH_SCORE} (std)', df_std_benchscore)

    # compute clemscores and add to df
    clemscore = ((df_mean['% Played'] / 100)
                 * df_mean[BENCH_SCORE])
    df_mean.insert(0, 'clemscore', clemscore.values.astype(float).round(2))

    return df_mean

def make_codenames_tables(df: pd.DataFrame) -> pd.DataFrame:
    df_aux= df.drop(columns=['experiment variable', 'experiment name'], errors='ignore')
    
    df_game_metrics = df_aux[GAME_METRICS].groupby(['model'], sort=False).mean(numeric_only=False)
    df_game_metrics = df_game_metrics.apply(pd.to_numeric).round(2)
    
    df_requests = df_aux[REQUEST_METRICS].groupby(['model'], sort=False).sum(numeric_only=False)
    df_requests[METRIC_REQUEST_SUCCESS] = df_requests[METRIC_REQUEST_COUNT_PARSED] / df_requests[METRIC_REQUEST_COUNT]
    df_requests = df_requests.apply(pd.to_numeric)
    df_requests[METRIC_REQUEST_SUCCESS] = df_requests[METRIC_REQUEST_SUCCESS].round(2)

    df_flags = df_aux.filter(regex='^(?!Average)').filter(regex='Cluegiver|Guesser').groupby(['model'], sort=False).sum(numeric_only=False)
    df_average_turn_scores = df_aux.filter(regex='Average').groupby(['model'], sort=False).mean(numeric_only=False)
    df_average_turn_scores = df_average_turn_scores.apply(pd.to_numeric).round(2)
    return df_game_metrics, df_requests, df_flags, df_average_turn_scores

def save_table(df, path, table_name):
    Path(path).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(path) / f'{table_name}.csv')
    # df.to_html(Path(path) / f'{table_name}.html')
    print(f'\n Saved results into {path}/{table_name}.csv')

def error_evaluation(results_path):
    # load interaction files
    errors = {}
    error_causes = {}
    interactions = utils.load_interactions(GAME_NAME, results_path)
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
    error_df = error_df.apply(pd.to_numeric, downcast = 'integer')
    print(error_df)
    save_table(error_df, results_path, "errors")

    # save errors with double index player-episode, put utterances in there as well?
    with open(f"{results_path}/error_causes.json", 'w') as file:
        json.dump(error_causes, file)

def main(args):
    if args.mode == "models":
        score_models(args)
    elif args.mode == "experiments":
        score_experiments(args)
    elif args.mode == "errors":
        error_evaluation(args.results_path)
    elif args.mode == "all":
        score_models(args)
        score_experiments(args)
        error_evaluation(args.results_path)
    else:
        print("Usage: $: python3 evaluation/codenames_eval.py [-m <models|experiments|errors|all>] [-r <results_path>]")

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("-m", '--mode', type=str, default="all", help="Mode, either one of models, experiments, errors, or all.")
    parser.add_argument("-r", "--results_path", type=str, default='./results',
                        help="Path to the results folder containing scores and interactions.")

    args = parser.parse_args()
    main(args)