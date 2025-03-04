"""
Functions that create tables with results.
"""

import numpy as np
import pandas as pd

import clemcore.clemeval.evalutils as utils
import clemcore.clemgame.metrics as clemmetrics


def save_multiple_formats(options, df):
    """Save a dataframe in .csv, .html and .tex."""
    csv_name = utils.create_file_name(*options, 'csv')
    df.to_csv(csv_name)
    html_name = utils.create_file_name(*options, 'html')
    df.to_html(buf=html_name)
    tex_name = utils.create_file_name(*options, 'tex')
    df.to_latex(buf=tex_name, float_format=utils.FLOAT_FORMAT, na_rep='n/a')


def build_dispersion_table(catcolumns, df):
    """Group by categories and build table with dispersion statistics."""
    mean = (df.groupby(catcolumns)['value']
              .mean(numeric_only=True)
              .rename('mean')
              .to_frame())
    median = (df.groupby(catcolumns)['value']
                .median(numeric_only=True)
                .rename('median')
                .to_frame())
    var = (df.groupby(catcolumns)['value']
             .var(numeric_only=True)
             .rename('var')
             .to_frame())
    std = (df.groupby(catcolumns)['value']
             .std(numeric_only=True)
             .rename('std')
             .to_frame())
    minimum = (df.groupby(catcolumns)['value']
                 .min(numeric_only=True)
                 .rename('min')
                 .to_frame())
    maximum = (df.groupby(catcolumns)['value']
                 .max(numeric_only=True)
                 .rename('max')
                 .to_frame())
    skew = (df.groupby(catcolumns)['value']
              .skew(numeric_only=True)
              .rename('skew')
              .to_frame())
    df = pd.concat([mean, std, var, median, maximum, minimum, skew], axis=1)
    return df


def make_stats_table(df):
    """Create table with dispersion statistics over all episodes."""
    catcolumns = ['game', 'model', 'metric']
    df_stats = build_dispersion_table(catcolumns, df)
    options = ['', 'episode', 'tables', 'bench-stats']
    save_multiple_formats(options, df_stats)
    return df


def save_detailed_table(df):
    """Create table with dispersion statistics grouped by experiments."""
    catcolumns = ['game', 'model', 'experiment', 'metric']
    df_stats = build_dispersion_table(catcolumns, df)
    options = ['', 'episode', 'tables', 'detailed-bench-stats']
    save_multiple_formats(options, df_stats)


def save_paper_table(df):
    """Create table with % played and main score."""
    df_aux = df[df['metric'].isin(utils.MAIN_METRICS)]
    categories = ['game', 'model', 'metric']
    # mean over all experiments
    df_mean = (df_aux.groupby(categories)
                     .mean(numeric_only=True)
                     .rename({'value': 'mean'}, axis=1)
                     .reset_index())
    df_mean.loc[df_mean.metric == clemmetrics.METRIC_PLAYED, 'mean'] *= 100
    df_mean = df_mean.round(2)
    # standard deviation over all experiments
    df_std = (df_aux.groupby(categories)
                    .std(numeric_only=True)
                    .rename({'value': 'std'}, axis=1)
                    .reset_index()
                    .round(2))
    df_std.loc[df_std.metric == clemmetrics.METRIC_PLAYED, 'std'] = '-'

    # average across all activities (i.e. mean by row)
    df_mean = df_mean.pivot(columns=['game'], index=['model', 'metric'])
    df_mean['all'] = df_mean.mean(numeric_only=True, axis=1)
    df_std = df_std.pivot(columns=['game'], index=['model', 'metric'])

    # double check the order
    assert all(df_mean.index == df_std.index)
    pairs = zip(df_mean.columns[:-1], df_std.columns)
    assert all(mean_col[1] == std_col[1] for mean_col, std_col in pairs)

    # merge both, putting std in parenthesis
    df_aux = df_mean['mean'].astype(str) + ' (' + df_std['std'].astype(str)+')'
    df_paper = pd.concat([df_mean['all'].round(2), df_aux], axis=1)
    # put in desired row and column order for the paper
    # from: https://stackoverflow.com/questions/23482668/sorting-by-a-custom-list-in-pandas
    df_paper.sort_values(
        ['model', 'metric'],
        ascending=[True, True],
        key=lambda column: column.map(lambda e: utils.ROW_ORDER.index(e)),
        inplace=True
        )
    df_paper = df_paper[utils.COLUMN_ORDER]
    options = ['', 'episode', 'tables', 'bench-paper-table']
    save_multiple_formats(options, df_paper)
    return df_paper


def save_clem_score_table(df_paper: pd.DataFrame) -> None:
    """Create a table with the clem score for each model."""
    df_aux = (df_paper['all'].to_frame()
                             .reset_index()
                             .pivot(index=['model'], columns=['metric']))
    df_aux['clemscore'] = ((df_aux[('all', 'Played')] / 100)
                           * df_aux[('all', 'Main Score')])
    # put in desired row and column order for the paper
    # from: https://stackoverflow.com/questions/23482668/sorting-by-a-custom-list-in-pandas
    df_aux = df_aux['clemscore'].to_frame().reset_index()
    df_aux = df_aux.sort_values('model',
                                key=lambda column: column.map(lambda e: utils.ROW_ORDER.index(e)))
    options = ['', 'episode', 'tables', 'clem-score-table']
    save_multiple_formats(options, df_aux)
    return df_aux


def make_overview_by_game(df: pd.DataFrame) -> None:
    """Create one table by game with all metrics by experiment and model."""
    for game, game_df in df.groupby('game'):
        results_df = (game_df.groupby(['model', 'experiment', 'metric'])
                             .mean(numeric_only=True)
                             .reset_index()
                             .pivot(index=['model', 'experiment'],
                                    columns=['metric']))
        results_df.columns = results_df.columns.droplevel()
        results_df.columns.name = None
        results_df.index.names = [None, None]

        results_df['Aborted'] *= 100
        results_df['Played'] *= 100
        results_df['Lose'] *= 100
        results_df['Success'] *= 100

        aux_array = results_df['Success'] + results_df['Lose']
        np.testing.assert_array_almost_equal(aux_array, results_df['Played'],
                                             decimal=8)

        # add number of run episodes for each experiment/model
        # we use Played but it could be any other metric name
        # as long it gets logged (even if only a nan) for all games
        # that actually got played; we only care for the count
        aux_counts = (game_df[game_df.metric == 'Played']
                      .groupby(['model', 'experiment', 'metric'])
                      .count()
                      .rename(columns={'episode': 'n'})
                      .reset_index()
                      .drop(['metric', 'game', 'value'], axis=1)
                      .set_index(['model', 'experiment']))
        
        assert all(aux_counts.index == results_df.index)
        results_df = pd.concat([aux_counts, results_df], axis=1)

        options = [game, 'episode', 'tables', f'{game}-overview-table']
        save_multiple_formats(options, results_df)


def make_detailed_overview_by_game(df: pd.DataFrame) -> None:
    """Create one table by game with all metrics by experiment and model."""
    for game, game_df in df.groupby('game'):
        results_df = (game_df.drop('game', axis=1)
                             .sort_values(by=['metric', 'episode'])
                             .pivot(index=['model', 'experiment'],
                                    columns=['metric', 'episode']))
        results_df.columns = results_df.columns.droplevel()
        results_df.columns.name = None
        results_df.index.names = [None, None]

        options = [game, 'episode', 'tables', f'{game}-detailed-table']
        save_multiple_formats(options, results_df)
