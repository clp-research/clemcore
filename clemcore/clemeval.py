"""
Clembench Evaluation

This script produces the main table with benchmark results, for all models
and games in the given results directory structure.

"""
import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import clemcore.clemgame.metrics as clemmetrics

TABLE_NAME = 'results'

# metrics that go in the main results table
MAIN_METRICS = [clemmetrics.METRIC_PLAYED, clemmetrics.BENCH_SCORE]


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

    # extract only relevant metrics
    df = df[df['metric'].isin(MAIN_METRICS)]

    # make sure all values are actually numeric (temporarily surpressing SettingwithCopyWarning)
    with pd.option_context('mode.chained_assignment', None):
        df['value'] = pd.to_numeric(df['value'])

    # compute mean benchscore and mean played (which is binary, so a proportion)
    df_a = (df.groupby(['game', 'model', 'metric'])
            .mean(numeric_only=True)
            .reset_index())
    df_a.loc[df_a.metric == clemmetrics.METRIC_PLAYED, 'value'] *= 100
    df_a = df_a.round(2)
    df_a['metric'].replace(
        {clemmetrics.METRIC_PLAYED: '% ' + clemmetrics.METRIC_PLAYED},
        inplace=True)

    # compute the std of benchscore
    df = df[df.metric == clemmetrics.BENCH_SCORE]
    df_b = (df.groupby(['game', 'model', 'metric'])
            .std(numeric_only=True)
            .reset_index()
            .round(2))
    df_b['metric'].replace(
        {clemmetrics.BENCH_SCORE: clemmetrics.BENCH_SCORE + ' (std)'},
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


def name_as_tuple(name: dict) -> tuple:
    """Turn the file path name into a tuple."""
    return (name['game'], name['model'], name['experiment'], name['episode'])


def load_json(path: Path) -> dict:
    """Load a json file."""
    with open(path, 'r') as file:
        data = json.load(file)
    return data


def parse_directory_name(name: Path) -> dict:
    """Extract information from the directory name structure."""

    splits = str(name).split(os.sep)
    model, game, experiment, episode, _ = splits[-5:]
    return {'game': game,
            'model': model,
            'experiment': experiment,
            'episode': episode}


def load_scores(path: str) -> dict:
    """Get all turn and episodes scores and return them in a dictionary."""
    # https://stackoverflow.com/a/18394205
    score_files = list(Path(path).rglob("*scores.json"))
    print(f'Loading {len(score_files)} JSON files.')
    scores = {}
    for path in tqdm(score_files, desc="Loading scores"):
        naming = name_as_tuple(parse_directory_name(path))
        if naming not in scores:
            data = load_json(path)
            scores[naming] = {}
            scores[naming]['turns'] = data['turn scores']
            scores[naming]['episodes'] = data['episode scores']
        else:
            print(f'Repeated file {naming}!')
    print(f'Retrieved {len(scores)} JSON files with scores.')
    return scores


def build_df_episode_scores(scores: dict) -> pd.DataFrame:
    """Create dataframe with all episode scores."""
    cols = ['game', 'model', 'experiment', 'episode', 'metric', 'value']
    df_episode_scores = pd.DataFrame(columns=cols)
    desc = "Build episode scores dataframe"
    for name, data in tqdm(scores.items(), desc=desc):
        (game, model, experiment, episode) = name
        for metric_name, metric_value in data['episodes'].items():
            new_row = [game, model, experiment, episode,
                       metric_name, metric_value]
            df_episode_scores.loc[len(df_episode_scores)] = new_row
    return df_episode_scores


def perform_evaluation(results_path: str):
    # Get all episode scores as a pandas dataframe
    scores = load_scores(path=results_path)
    df_episode_scores = build_df_episode_scores(scores)

    # Create the PLAYED variable, inferring it from ABORTED
    if clemmetrics.METRIC_PLAYED in df_episode_scores['metric'].unique():
        raise PlayedScoreError("Computed scores should not contain METRIC_PLAYED.")
    aux = df_episode_scores[df_episode_scores["metric"] == clemmetrics.METRIC_ABORTED].copy()
    aux["metric"] = clemmetrics.METRIC_PLAYED
    aux["value"] = 1 - aux["value"]
    # We need ignore_index=True to reset the indices (otherwise we have duplicates)
    df_episode_scores = pd.concat([df_episode_scores, aux], ignore_index=True)

    # save raw scores
    df_episode_scores.to_csv(Path(results_path) / f'raw.csv')
    print(f'\n Saved raw scores into {results_path}/raw.csv')

    # save main table
    save_clem_table(df_episode_scores, results_path)
