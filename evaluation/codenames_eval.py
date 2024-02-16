from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

import evaluation.evalutils as utils
import clemgame.metrics as clemmetrics

TABLE_NAME = 'results'

# from evaluation.evalutils:
#   save_raw_turn_scores()
#   save_raw_episode_scores()
#   save_raw_scores()
#   build_df_episode_scores()
#   build_df_turn_scores()
#   load_interactions(game_name)
#   load_scores(game_name, results)

def score_evaluation(args):
    # load scores
    # do some data science magic

    # Get all episode scores as a pandas dataframe
    scores = utils.load_scores(path=args.results_path)
    df_episode_scores = utils.build_df_episode_scores(scores)

    # Create the PLAYED variable, inferring it from ABORTED
    if clemmetrics.METRIC_PLAYED in df_episode_scores['metric'].unique():
        raise PlayedScoreError("Computed scores should not contain METRIC_PLAYED.")
    aux = df_episode_scores[df_episode_scores["metric"] == clemmetrics.METRIC_ABORTED].copy()
    aux["metric"] = clemmetrics.METRIC_PLAYED
    aux["value"] = 1 - aux["value"]
    # We need ignore_index=True to reset the indices (otherwise we have duplicates)
    df_episode_scores = pd.concat([df_episode_scores, aux], ignore_index=True)

    save_clem_table(df_episode_scores, args.results_path)

def save_clem_table(df: pd.DataFrame, path: str) -> None:
    # WHY DOES THE FUNCTION SAVING A TABLE CALCULATE STUFF? SRSLY
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

    # save table
    df_results.to_csv(Path(path) / f'{TABLE_NAME}.csv')
    df_results.to_html(Path(path) / f'{TABLE_NAME}.html')
    print(f'\n Saved results into {path}/{TABLE_NAME}.csv and .html')

def error_evaluation():
    # load interaction files
    # loop through interactions
    # keep track per model, which error messages occurred
    #   for that put error names and error messages into ValidationError and log both when logging _to_self
    pass

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