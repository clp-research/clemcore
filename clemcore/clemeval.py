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


def save_clem_table(df: pd.DataFrame, path: str, show_std: bool = False, sort_by: str = "model_name") -> pd.DataFrame | None:
    """Create benchmark results as a table.
    Args:
        df: Episode scores dataframe.
        path: Directory path to save the results files.
        show_std: If True, include standard deviation columns for Quality Score. Default: False.
        sort_by: Sort rows by 'model_name' (alphabetical) or 'clemscore' (descending). Default: 'model_name'.
    """

    # extract only relevant metrics
    df = df[df['metric'].isin(MAIN_METRICS)]

    # make sure all values are actually numeric
    df = df.copy()
    df['value'] = pd.to_numeric(df['value'])

    # compute mean benchscore and mean played (which is binary, so a proportion)
    df_a = (df.groupby(['game', 'model', 'metric'])
            .mean(numeric_only=True)
            .reset_index())
    df_a.loc[df_a.metric == clemmetrics.METRIC_PLAYED, 'value'] *= 100
    df_a = df_a.round(2)
    df_a['metric'] = df_a['metric'].replace(
        {clemmetrics.METRIC_PLAYED: '% ' + clemmetrics.METRIC_PLAYED})

    # compute the macro-average main score over games, per model
    df_all = (df_a.groupby(['model', 'metric'])
              .mean(numeric_only=True)
              .reset_index()
              .round(2))
    # add columns for standard format in concatenation below
    df_all['game'] = 'all'
    df_all['metric'] = 'Average ' + df_all['metric']

    parts = [df_a, df_all]

    if show_std:
        # compute the std of benchscore
        df_std = df[df.metric == clemmetrics.BENCH_SCORE]
        df_b = (df_std.groupby(['game', 'model', 'metric'])
                .std(numeric_only=True)
                .reset_index()
                .round(2))
        df_b['metric'] = df_b['metric'].replace(
            {clemmetrics.BENCH_SCORE: clemmetrics.BENCH_SCORE + ' (std)'})
        parts.insert(1, df_b)

    # merge all data and make it one model per row
    df_full = pd.concat(parts, axis=0, ignore_index=True)
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

    # sort rows
    if sort_by == "clemscore":
        df_results.sort_values(by='-, clemscore', ascending=False, inplace=True)
    else:
        df_results.sort_index(inplace=True)

    # save table
    df_results.to_csv(Path(path) / f'{TABLE_NAME}.csv')
    df_results.to_html(Path(path) / f'{TABLE_NAME}.html')
    print(f'\n Saved results into {path}/{TABLE_NAME}.csv and .html')
    return df_results

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


def load_scores(path: str, model_selector: str = None, game_selector: str = None) -> dict:
    """Get all turn and episodes scores and return them in a dictionary.
    Args:
        path: Root directory to search for score files.
        model_selector: Optional substring to filter score files by model name.
        game_selector: Optional substring to filter score files by game name.
    """
    # https://stackoverflow.com/a/18394205
    score_files = list(Path(path).rglob("*scores.json"))
    if model_selector is not None:
        score_files = [f for f in score_files if model_selector in str(f)]
    if game_selector is not None:
        score_files = [f for f in score_files if game_selector in Path(f).parts]
    print(f'Loading {len(score_files)} JSON files.')
    scores = {}
    for path in tqdm(score_files, desc="Loading scores"):
        naming = name_as_tuple(parse_directory_name(path))
        if naming not in scores:
            data = load_json(path)
            scores[naming] = {}
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


def perform_evaluation(results_path: str, return_dataframe: bool = False,
                       show_std: bool = False, sort_by: str = "model_name",
                       model_selector: str = None, game_selector: str = None) -> pd.DataFrame | None:
    """Run evaluation and save results table.
    Args:
        results_path: Root directory containing score files and where results are saved.
        return_dataframe: If True, return the results dataframe.
        show_std: If True, include standard deviation columns.
        sort_by: Sort rows by 'model_name' or 'clemscore'.
        model_selector: Optional substring to restrict evaluation to a specific model.
            When set (with or without game_selector), loads only matching scores from
            disk, merges them into the existing raw.csv (replacing matching rows), and
            recomputes the full results table.
        game_selector: Optional substring to restrict evaluation to a specific game.
            Can be combined with model_selector for finer-grained updates.
    """
    raw_csv_path = Path(results_path) / 'raw.csv'
    incremental = (model_selector is not None or game_selector is not None) and raw_csv_path.exists()

    if incremental:
        # Incremental update: load existing raw data, drop rows matching the selectors,
        # load fresh scores for the selection, then recompute the full table.
        df_existing = pd.read_csv(raw_csv_path, index_col=0)
        mask = pd.Series(True, index=df_existing.index)
        if model_selector is not None:
            mask &= df_existing['model'].str.contains(model_selector, na=False)
        if game_selector is not None:
            mask &= df_existing['game'] == game_selector
        df_existing = df_existing[~mask]
        scores = load_scores(path=results_path, model_selector=model_selector, game_selector=game_selector)
        df_new = build_df_episode_scores(scores)
        if clemmetrics.METRIC_PLAYED in df_new['metric'].unique():
            raise PlayedScoreError("Computed scores should not contain METRIC_PLAYED.")
        aux = df_new[df_new["metric"] == clemmetrics.METRIC_ABORTED].copy()
        aux["metric"] = clemmetrics.METRIC_PLAYED
        aux["value"] = 1 - aux["value"]
        df_new = pd.concat([df_new, aux], ignore_index=True)
        df_episode_scores = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        # Full evaluation: load all scores from disk.
        scores = load_scores(path=results_path, model_selector=model_selector, game_selector=game_selector)
        df_episode_scores = build_df_episode_scores(scores)
        if clemmetrics.METRIC_PLAYED in df_episode_scores['metric'].unique():
            raise PlayedScoreError("Computed scores should not contain METRIC_PLAYED.")
        aux = df_episode_scores[df_episode_scores["metric"] == clemmetrics.METRIC_ABORTED].copy()
        aux["metric"] = clemmetrics.METRIC_PLAYED
        aux["value"] = 1 - aux["value"]
        df_episode_scores = pd.concat([df_episode_scores, aux], ignore_index=True)

    # save raw scores
    df_episode_scores.to_csv(raw_csv_path)
    print(f'\n Saved raw scores into {results_path}/raw.csv')

    # save main table
    df_episode_scores = save_clem_table(df_episode_scores, results_path, show_std=show_std, sort_by=sort_by)
    if return_dataframe:
        return df_episode_scores
