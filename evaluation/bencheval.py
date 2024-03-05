"""
Clembench Evaluation

This script produces the main table with benchmark results, for all models
and games in the given results directory structure.

"""
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

import evaluation.evalutils as utils
import clemgame.metrics as clemmetrics

TABLE_NAME = 'results'


class PlayedScoreError(Exception):
    """clemmetrics.METRIC_PLAYED found in scores.
    
    This metric is computed locally, as the complement of 
    clemmetrics.METRIC_ABORTED. Games should not compute it, otherwise there
    would be duplicates in the dataframe. This is in the documentation.
    NOTE: This could instead be verified silently and only computed
    for games that do not have it.
    """
    pass


def save_clem_table(df: pd.DataFrame, path: str) -> None:
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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-p", "--results_path",
                        type=str,
                        default='./results',
                        help="Path to the results folder containing scores.")
    args = parser.parse_args()

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

    # save raw scores
    df_episode_scores.to_csv(Path(args.results_path) / f'raw.csv')
    print(f'\n Saved raw scores into {args.results_path}/raw.csv')

    # save main table
    save_clem_table(df_episode_scores, args.results_path)
