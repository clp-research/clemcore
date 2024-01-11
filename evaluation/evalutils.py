"""
Auxiliary functions for plots and tables.
"""

import os
from pathlib import Path

import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import clemgame.metrics as clemmetrics

EVAL_DIR = 'results_eval'
RESULTS_DIR = './results'
SEP = '---'
FLOAT_FORMAT = "%.2f"

# metrics that go in the main results table
MAIN_METRICS = [clemmetrics.METRIC_PLAYED, clemmetrics.BENCH_SCORE]

# metrics that are percentual
PERC_METRICS = [
    clemmetrics.METRIC_SUCCESS,
    clemmetrics.METRIC_ABORTED,
    clemmetrics.METRIC_LOSE,
    clemmetrics.METRIC_REQUEST_SUCCESS,
    ]

# metrics that all games log
COMMON_METRICS = [
    clemmetrics.BENCH_SCORE,
    clemmetrics.METRIC_PLAYED,
    clemmetrics.METRIC_ABORTED,
    clemmetrics.METRIC_LOSE,
    clemmetrics.METRIC_SUCCESS,
    clemmetrics.METRIC_REQUEST_COUNT,
    clemmetrics.METRIC_REQUEST_COUNT_PARSED,
    clemmetrics.METRIC_REQUEST_COUNT_VIOLATED,
    clemmetrics.METRIC_REQUEST_SUCCESS,
    ]

GAMEPLAY_METRICS = [
    clemmetrics.METRIC_SUCCESS,
    clemmetrics.METRIC_ABORTED,
    clemmetrics.METRIC_LOSE
    ]

# order of the rows in the main table, to be used as a key in pandas
ROW_ORDER = ['lm--lm', 'ko--ko', 'flc--flc', 'ost--ost', 'vcn--vcn',
             'cl--cl',  '3--3', '3.5--3.5', '3.5--4', '4--3.5', '4--4',
             clemmetrics.METRIC_PLAYED, clemmetrics.BENCH_SCORE]

# order of the columns in the main table
COLUMN_ORDER = ['all', 'taboo', 'wordle', 'wordle_withclue',
                'wordle_withcritic', 'imagegame', 'referencegame',
                'privateshared']

# shorter names for the models
short_names = {
    "t0.0": "",
    "claude-v1.3-": "cl",
    "gpt-3.5-turbo-": "3.5",
    "gpt-4-": "4",
    "text-davinci-003-": "3",
    "luminous-supreme-": "lm",
    "koala-13b-": "ko",
    "falcon-40b-": "flc",
    "oasst-12b-": "ost",
    "vicuna-13b-": "vcn"
}

# short names for the scatterplot
plot_annotations = {
    '4--4': '4',
    '3--3': '3',
    'lm--lm': 'lm',
    'cl--cl': 'cl',
    '3.5--3.5': '3.5',
    '4--3.5': '4/3.5',
    '3.5--4': '3.5/4',
    'ko--ko': 'ko',
    'flc--flc': 'flc',
    'ost--ost': 'ost',
    'vcn--vcn': 'vcn'
    }

metric_lims = {
    clemmetrics.BENCH_SCORE: (-2, 102),
    clemmetrics.METRIC_PLAYED: (-0.05, 1.05),
    clemmetrics.METRIC_ABORTED: (-0.05, 1.05),
    clemmetrics.METRIC_LOSE: (-0.05, 1.05),
    clemmetrics.METRIC_SUCCESS: (-0.05, 1.05),
    clemmetrics.METRIC_REQUEST_COUNT: (-2, None),
    clemmetrics.METRIC_REQUEST_COUNT_PARSED: (-2, None),
    clemmetrics.METRIC_REQUEST_COUNT_VIOLATED: (-2, None),
    clemmetrics.METRIC_REQUEST_SUCCESS: (-0.05, 1.05),
}


def savefig(name: str) -> None:
    """Save a plt figure."""
    sns.despine(left=False, right=False, top=False, bottom=False)
    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight')
    plt.close()


def parse_directory_name(name: str) -> dict:
    """Extract information from the directory name structure."""

    splits = str(name).split('/')
    model, game, experiment, episode, _ = splits[-5:]
    return {'game': game,
            'model': model,
            'experiment': experiment,
            'episode': episode}


def name_as_tuple(name: str) -> tuple:
    """Turn the file path name into a tuple."""
    return (name['game'], name['model'], name['experiment'], name['episode'])


def load_json(path: str) -> dict:
    """Load a json file."""
    with open(path, 'r') as file:
        data = json.load(file)
    return data


def load_scores(game_name: str = None, path: str = RESULTS_DIR) -> dict:
    """Get all turn and episodes scores and return them in a dictionary."""
    # https://stackoverflow.com/a/18394205
    score_files = list(Path(path).rglob("*scores.json"))
    print(f'Loading {len(score_files)} JSON files.')
    scores = {}
    for path in tqdm(score_files, desc="Loading scores"):
        if game_name:
            if game_name not in str(path):
                continue
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


def load_interactions(game_name: str = None) -> dict:
    """Get all interaction records and return them in a dictionary."""
    # https://stackoverflow.com/a/18394205
    interaction_files = list(Path(RESULTS_DIR).rglob("*interactions.json"))
    print(f'Loading {len(interaction_files)} JSON files.')
    interactions = {}
    for path in tqdm(interaction_files, desc="Loading interactions"):
        if game_name:
            if game_name not in str(path):
                continue
        naming = name_as_tuple(parse_directory_name(path))
        if naming not in interactions:
            data = load_json(path)
            instance = load_json(str(path).replace('interactions.json', 'instance.json'))
            interactions[naming] = (data, instance)
        else:
            print(f'Repeated file {naming}!')
    print(f'Retrieved {len(interactions)} JSON files with interactions.')
    return interactions


def create_eval_subdirs(path: str) -> None:
    """Create folders for turn and episode scores, for plots and tables."""
    eval_path = path / EVAL_DIR
    path_turn = eval_path / 'turn-level'
    Path(path_turn).mkdir(parents=True, exist_ok=True)
    Path(f'{path_turn}/plots').mkdir(parents=True, exist_ok=True)
    Path(f'{path_turn}/tables').mkdir(parents=True, exist_ok=True)

    path_epi = eval_path / 'episode-level'
    Path(path_epi).mkdir(parents=True, exist_ok=True)
    Path(f'{path_epi}/plots').mkdir(parents=True, exist_ok=True)
    Path(f'{path_epi}/tables').mkdir(parents=True, exist_ok=True)


def create_eval_tree(levels: list) -> None:
    """Create eval directory with same structure as the results."""
    bencheval_path = Path(f'./{EVAL_DIR}/')
    create_eval_subdirs(bencheval_path)
    for game, model, experiment, episode in levels:
        game_path = bencheval_path / game
        create_eval_subdirs(game_path)
        model_path = game_path / model
        create_eval_subdirs(model_path)
        exp_path = model_path / experiment
        create_eval_subdirs(exp_path)
        episode_path = exp_path / episode
        create_eval_subdirs(episode_path)


def build_df_turn_scores(scores: dict) -> pd.DataFrame:
    """Create dataframe with all turn scores."""
    cols = ['game', 'model', 'experiment', 'episode', 'turn', 'metric', 'value']
    df_turn_scores = pd.DataFrame(columns=cols)
    for name, data in tqdm(scores.items(), desc="Build turn scores dataframe"):
        (game, model, experiment, episode) = name
        for turn, turn_data in data['turns'].items():
            for metric_name, metric_value in turn_data.items():
                new_row = [game, model, experiment, episode, turn,
                           metric_name, metric_value]
                df_turn_scores.loc[len(df_turn_scores)] = new_row
    return df_turn_scores


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


def filter_df_by_key(df: pd.DataFrame, value_dict: dict) -> pd.DataFrame:
    """Return a dataframe with only the desired values."""
    df_filtered = df
    for key, value in value_dict.items():
        df_filtered = df_filtered[(df_filtered[key] == value)]
    return df_filtered


def save_raw_scores(df_turn_scores: pd.DataFrame,
                    df_episode_scores: pd.DataFrame,
                    scores: dict) -> None:
    """Create .csv files with all the scores"""
    name = create_file_name('', 'turn', 'tables', 'scores_raw', 'csv')
    df_turn_scores.to_csv(name)
    name = create_file_name('', 'episode', 'tables', 'scores_raw', 'csv')
    df_episode_scores.to_csv(name)
    save_raw_episode_scores(scores.keys(), df_episode_scores)
    save_raw_turn_scores(scores.keys(), df_turn_scores)
    print('Saved raw scores into .csv files.')


def save_raw_episode_scores(keys: list, df_scores: pd.DataFrame) -> None:
    """Create csv files with episode scores for each level."""
    desc = "Saving raw episode scores"
    for game, model, experiment, episode in tqdm(keys, desc=desc):
        # all results
        filter_dic = {'game': game, 'model': model,
                      'experiment': experiment, 'episode': episode}
        df_aux = filter_df_by_key(df_scores, filter_dic)
        df_aux = df_aux[['metric', 'value']]
        prefix = f'{EVAL_DIR}/{game}/{model}/{experiment}/{episode}/{EVAL_DIR}'
        name = f'{prefix}/episode-level/tables/scores_raw.csv'
        df_aux.to_csv(name)

        # by experiment
        filter_dic = {'game': game, 'model': model, 'experiment': experiment}
        df_aux = filter_df_by_key(df_scores, filter_dic)
        df_aux = df_aux.pivot(index='episode',
                              columns=['metric'],
                              values='value')
        prefix = f'{EVAL_DIR}/{game}/{model}/{experiment}/{EVAL_DIR}'
        name = f'{prefix}/episode-level/tables/scores_raw.csv'
        df_aux.to_csv(name)

        # by model
        filter_dic = {'game': game, 'model': model}
        df_aux = filter_df_by_key(df_scores, filter_dic)
        df_aux = df_aux.pivot(index=['episode'],
                              columns=['experiment', 'metric'],
                              values='value')
        prefix = f'{EVAL_DIR}/{game}/{model}/{EVAL_DIR}/episode-level'
        name = f'{prefix}/tables/scores_raw.csv'
        df_aux.to_csv(name)

        # by game
        filter_dic = {'game': game}
        df_aux = filter_df_by_key(df_scores, filter_dic)
        df_aux = df_aux.pivot(index=['model', 'episode'],
                              columns=['experiment', 'metric'],
                              values='value') 
        prefix = f'{EVAL_DIR}/{game}/{EVAL_DIR}'
        name = f'{prefix}/episode-level/tables/scores_raw.csv'
        df_aux.to_csv(name)


def save_raw_turn_scores(keys: list, df_scores: pd.DataFrame) -> None:
    """Create csv files with turn scores for each level."""
    desc = "Saving raw turn scores"
    for game, model, experiment, episode in tqdm(keys, desc=desc):
        # all results
        filter_dic = {'game': game, 'model': model,
                      'experiment': experiment, 'episode': episode}
        df_aux = filter_df_by_key(df_scores, filter_dic)
        df_aux = df_aux.pivot(index='turn', columns=['metric'], values='value')
        prefix = f'{EVAL_DIR}/{game}/{model}/{experiment}/{episode}/{EVAL_DIR}'
        name = f'{prefix}/turn-level/tables/scores_raw.csv'
        df_aux.to_csv(name)

        # by experiment
        filter_dic = {'game': game, 'model': model, 'experiment': experiment}
        df_aux = filter_df_by_key(df_scores, filter_dic)
        df_aux = df_aux.pivot(index=['episode', 'turn'],
                              columns=['metric'],
                              values='value')
        prefix = f'{EVAL_DIR}/{game}/{model}/{experiment}/{EVAL_DIR}'
        name = f'{prefix}/turn-level/tables/scores_raw.csv'
        df_aux.to_csv(name)

        # by model
        filter_dic = {'game': game, 'model': model}
        df_aux = filter_df_by_key(df_scores, filter_dic)
        df_aux = df_aux.pivot(index=['episode', 'turn'],
                              columns=['experiment', 'metric'],
                              values='value') 
        prefix = f'{EVAL_DIR}/{game}/{model}/{EVAL_DIR}'
        name = f'{prefix}/turn-level/tables/scores_raw.csv'
        df_aux.to_csv()

        # by game
        filter_dic = {'game': game}
        df_aux = filter_df_by_key(df_scores, filter_dic)
        df_aux = df_aux.pivot(index=['model', 'episode', 'turn'],
                              columns=['experiment', 'metric'],
                              values='value') 
        prefix = f'{EVAL_DIR}/{game}/{EVAL_DIR}'
        name = f'{prefix}/turn-level/tables/scores_raw.csv'
        df_aux.to_csv(name)


def create_file_name(subfolders: str, level: str, kind: str,
                     ending: str, extension: str) -> str:
    """Create file name according to the results structure."""
    if subfolders != '':
        return (f"{EVAL_DIR}/{subfolders.replace(' | ', '/')}/{EVAL_DIR}/"
                f"{level}-level/{kind}/{subfolders.replace(' | ', SEP)}"
                f"{ending}.{extension}")
    return f"{EVAL_DIR}/{EVAL_DIR}/{level}-level/{kind}/{ending}.{extension}"


def get_metrics_in_zero_one(df: pd.DataFrame) -> list:
    """Return metrics whose values are in the interval [0, 1]."""
    metrics_in_zero_one = []
    for metric, metric_df in df.groupby('metric'):
        if metric_df['value'].min() >= 0.0 and metric_df['value'].max() <= 1.0:
            metrics_in_zero_one.append(metric)
    return metrics_in_zero_one


def filter_metrics_in_zero_one(df: pd.DataFrame,
                               zero_one_scores: list) -> tuple:
    """Split dataframe into one with score in [0, 1] and the rest."""
    df_01 = df[df.metric.isin(zero_one_scores)]
    df_other = df[~df.metric.isin(zero_one_scores)]
    return df_01, df_other


def get_metric_lims(metric_name, zeroone_metrics):
    """Get the ylim according to the metric."""
    if metric_name in zeroone_metrics:
        return (-0.05, 1.05)
    if metric_name in metric_lims:
        return metric_lims[metric_name]
    return None, None
