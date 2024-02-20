from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

import evaluation.evalutils as utils
import clemgame.metrics as clemmetrics
from games.codenames.constants import *
from collections import Counter
import json

# from evaluation.evalutils:
#   save_raw_turn_scores()
#   save_raw_episode_scores()
#   save_raw_scores()
#   build_df_episode_scores()
#   build_df_turn_scores()
#   load_interactions(game_name)
#   load_scores(game_name, results)

def score_evaluation(args):
    # Get all episode scores as a pandas dataframe
    scores = utils.load_scores(path=args.results_path)
    df_episode_scores = utils.build_df_episode_scores(scores)

    # Create the PLAYED variable, inferring it from ABORTED
    aux = score_amount_played(df_episode_scores)
    
    # We need ignore_index=True to reset the indices (otherwise we have duplicates)
    df_episode_scores = pd.concat([df_episode_scores, aux], ignore_index=True)

    # create and save main benchmark table
    df = make_clem_table(df_episode_scores)
    save_table(df, args.results_path, "results")

    # create and save codenames tables
    df_metrics, df_requests = make_codenames_table(df_episode_scores)
    save_table(df_metrics, args.results_path, "codenames-specific results")
    save_table(df_requests, args.results_path, "codenames-requests")

def score_amount_played(df_episode_scores):
    if clemmetrics.METRIC_PLAYED in df_episode_scores['metric'].unique():
        raise PlayedScoreError("Computed scores should not contain METRIC_PLAYED.")
    aux = df_episode_scores[df_episode_scores["metric"] == clemmetrics.METRIC_ABORTED].copy()
    aux["metric"] = clemmetrics.METRIC_PLAYED
    aux["value"] = 1 - aux["value"]
    return aux

def make_clem_table(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: remove double metrics, as I only am interested in my own game, not everyone elses game
    """Create benchmark results as a table."""
    df_aux = df[df['metric'].isin(utils.MAIN_METRICS)]

    # compute mean benchscore and mean played (which is binary, so a proportion)
    df_a = (df_aux.groupby(['game', 'model', 'metric'])
                  .mean(numeric_only=True)
                  .reset_index())
    df_a.loc[df_a.metric == clemmetrics.METRIC_PLAYED, 'value'] *= 100
    df_a = df_a.round(2)
    df_a['metric'].replace(
        {clemmetrics.METRIC_PLAYED: '% '+clemmetrics.METRIC_PLAYED},
        inplace=True)

    # compute the std of benchscore
    df_aux_b = df_aux[df_aux.metric == clemmetrics.BENCH_SCORE]
    df_b = (df_aux_b.groupby(['game', 'model', 'metric'])
                    .std(numeric_only=True)
                    .reset_index()
                    .round(2))
    df_b['metric'].replace(
        {clemmetrics.BENCH_SCORE: clemmetrics.BENCH_SCORE+' (std)'},
        inplace=True)

    # compute the macro-average main score over games, per model
    df_all = (df_a.groupby(['model', 'metric'])
                  .mean(numeric_only=True)
                  .reset_index()
                  .round(2))
    # add columns for standard format in concatenation below
    df_all['game'] = 'all'
    df_all['metric'] = 'Average ' + df_all['metric']

    # merge all data and make it one model per row
    df_full = pd.concat([df_a, df_b, df_all], axis=0, ignore_index=True)
    # sort just so all metrics are close to each other in a game column
    df_full.sort_values(by=['game', 'metric'], inplace=True)
    # rename according to paper
    df_full['metric'] = df_full['metric'].str.replace(clemmetrics.BENCH_SCORE, 'Quality Score')
    df_full = df_full.pivot(columns=['game', 'metric'], index=['model'])
    df_full = df_full.droplevel(0, axis=1)

    # compute clemscores and add to df
    clemscore = ((df_full[('all', 'Average % Played')] / 100)
                 * df_full[('all', 'Average Quality Score')])
    clemscore = clemscore.round(2).to_frame(name=('-', 'clemscore'))
    df_results = pd.concat([clemscore, df_full], axis=1)

    # flatten header
    df_results.index.name = None
    df_results.columns = df_results.columns.to_flat_index() 
    df_results.columns = [', '.join(x) for x in df_results.columns]

    return df_results

def make_codenames_table(df: pd.DataFrame) -> pd.DataFrame:
    df_aux = df[df['game'] == GAME_NAME]
    print(df_aux)
    # compute mean benchscore and mean played (which is binary, so a proportion)
    df_aux = (df.groupby(['model', 'metric'])
                  .mean(numeric_only=True)
                  .reset_index()
                  .round(2))
    df_aux = df_aux.pivot(columns=['metric'], index=['model'])
    df_aux = df_aux.droplevel(0, axis=1)

    df_game_metrics = df_aux[['Aborted', 'Played', 'Success', 'Lose', 'game end', 'number of turns', 'efficiency', 'avg target f1', 'team words revealed/all team words', 'other words not revealed/all other words',]]
    df_requests = df_aux[['Request Count', 'Parsed Request Count', 'Violated Request Count', 'Request Success Ratio', 'ignore false targets or guesses', 'ignore rambling']]
    # TODO: use game_end Enum somehow to one-hot-encode game ends, mean does not tell anything!
    return df_game_metrics, df_requests
    
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
                    case Turn_logs.CLUEGIVER_INVALID_FORMAT | Turn_logs.GUESSER_INVALID_FORMAT:
                        # TODO: only log invalid format the same for both players
                        error_type = action["content"]["type"]
                        errors[players][error_type] += 1

                        error_cause = action["content"]["utterance"]
                        if error_type not in error_causes[players][-1]:
                            error_causes[players][-1][error_type] = []
                        error_causes[players][-1][error_type].append(error_cause)
                        # TODO: log error causes in an extra file

    # save errors aggregated over models as a table
    error_df = pd.DataFrame.from_dict(errors)
    error_df = error_df.transpose().fillna(0)
    print(error_df)
    error_df.to_csv(f"{results_path}/errors.csv")

    # save errors with double index player-episode, put utterances in there as well?
    with open(f"{results_path}/error_causes.json", 'w') as file:
        json.dump(error_causes, file)


def main(args):
    if args.command_name == "scores":
        score_evaluation(args)
    elif args.command_name == "errors":
        error_evaluation(args.results_path)
    else:
        print("Usage: $: python3 evaluation/codenames_eval.py [scores|errors]")

if __name__ == '__main__':
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers(dest="command_name")

    score_parser = sub_parsers.add_parser("scores")
    score_parser.add_argument("-p", "--results_path", type=str, default='./results',
                              help="Path to the results folder containing scores.")

    error_parser = sub_parsers.add_parser("errors")
    error_parser.add_argument("-p", "--results_path", type=str, default='./results',
                              help="Path to the results folder containing interactions.")

    args = parser.parse_args()
    main(args)