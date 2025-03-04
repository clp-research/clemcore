"""
Functions that create evaluation plots.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon

import clemcore.clemeval.evalutils as utils
import clemcore.clemgame.metrics as clemmetrics

ABORTED = clemmetrics.METRIC_ABORTED

COLORS = ['darkorange', 'teal', 'firebrick', 'purple', 'darkgoldenrod',
          'steelblue', 'darkgreen']

STACK_COLORS = ['darkolivegreen', 'indianred', 'gray']


# ------------------------ Evaluation of the Benchmark ------------------------
# Overview plots
def plot_escore_benchmark(df: pd.DataFrame, ending: str) -> None:
    "Create benchmark overview with subplots for all games."
    n_games = len(df.game.unique())
    n_models = len(df.model.unique())

    fig, all_axes = plt.subplots(n_games, 1, figsize=(15, n_games * 5))
    axs = all_axes.flatten()

    for n, (game, df_group) in enumerate(df.groupby('game')):
        g = sns.barplot(data=df_group,
                        x='metric',
                        y='value',
                        hue='model',
                        hue_order=utils.ROW_ORDER[:-2],
                        errwidth=0.8,
                        ax=axs[n])
        axs[n].set_title(game, loc='left', fontsize=22)
        if ending == '_01':
            axs[n].set_ylim(-0.05, 1.05)
        labels = [(item.get_text()
                       .replace(' ', '\n')
                       .replace('_', '\n')
                       .replace('-', '\n')) for item in axs[n].get_xticklabels()]
        axs[n].set_xticklabels(labels)
        axs[n].set_xlabel('')
        axs[n].legend(loc='upper right',
                      bbox_to_anchor=(1, 1.3),
                      ncol=n_models / 2)

    plt.suptitle('Benchmark Overview', fontsize=22, y=1)
    fig.tight_layout()
    name = f'overview{ending}'
    path = utils.create_file_name('', 'episode', 'plots', name, 'pdf')
    utils.savefig(path)


def plot_stacked_micro_bar(df, df_clem):
    """Create plot with % aborted, success, lose horizontal bars.

    Ordered by clemscore. Micro average (i.e. mean over all episodes per model)
    """
    df_aux = df[df.metric.isin(utils.GAMEPLAY_METRICS)]
    df_aux = (df_aux.pivot(index=['game', 'model', 'experiment', 'episode'],
                           columns='metric',
                           values='value')
                    .reset_index()
                    .drop(columns=['game', 'experiment', 'episode'])
                    .groupby('model')
                    .sum()
                    .sort_values(axis=1, by='metric', ascending=False))
    percs = 100 * df_aux.div(df_aux.sum(axis=1), axis=0)
    order = df_clem.sort_values(by='clemscore').model
    percs = percs.reindex(order)

    percs.plot(kind='barh',
               stacked=True,
               figsize=(5, 5),
               colormap=ListedColormap(STACK_COLORS))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncols=3)
    plt.xlabel('% of all episodes')
    plt.xlim(-1, 101)
    plt.ylabel('')
    name = 'succes-lose-aborted_micro-avr'
    path = utils.create_file_name('', 'episode', 'plots', name, 'pdf')
    utils.savefig(path)


def plot_stacked_macro_bar(df, df_clem):
    """Create plot with % aborted, success, lose horizontal bars.

    Ordered by clemscore. Macro average (i.e. mean per game, then mean per
    model, as in the clemscore).
    """

    df_aux = df[df.metric.isin(utils.GAMEPLAY_METRICS)]
    df_aux = 100 * (df_aux.groupby(['model', 'game', 'metric'])
                          .mean(numeric_only=True)
                          .reset_index()
                          .groupby(['model', 'metric'])
                          .mean(numeric_only=True))
    df_aux = (df_aux.reset_index()
                    .pivot(columns='metric', index=['model']))
    order = df_clem.sort_values(by='clemscore').model
    df_aux = df_aux.reindex(order)
    df_aux.columns = df_aux.columns.droplevel()
    df_aux = df_aux[sorted(utils.GAMEPLAY_METRICS)[::-1]]

    df_aux.plot(kind='barh',
                stacked=True,
                figsize=(5, 5),
                colormap=ListedColormap(STACK_COLORS))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncols=3)
    plt.xlabel('macro average % episodes')
    plt.xlim(-1, 101)
    plt.ylabel('')
    name = 'succes-lose-aborted_macro-avr'
    path = utils.create_file_name('', 'episode', 'plots', name, 'pdf')
    utils.savefig(path)


def plot_paper_scatter(df_paper: pd.DataFrame) -> None:
    """Create scatter plot with % played vs. quality score."""
    dots = (df_paper['all'].to_frame()
                           .reset_index()
                           .pivot(index=['model'], columns=['metric']))
    dots.columns = dots.columns.to_flat_index()
    dots.index.name = None
    # use 0 for NaN
    dots.fillna(0, inplace=True)

    g = sns.scatterplot(dots,
                        x=('all', clemmetrics.METRIC_PLAYED),
                        y=('all', clemmetrics.BENCH_SCORE),
                        hue=dots.index,
                        s=100)
    sns.move_legend(g, loc='center right', bbox_to_anchor=(1.3, 0.8))
    for idx, row in dots.iterrows():
        if idx not in ('4--4', '3.5--4', 'flc--flc', 'ost--ost'):
            g.annotate(utils.plot_annotations[idx],
                       (row[('all', clemmetrics.METRIC_PLAYED)] - 2,
                        row[('all', clemmetrics.BENCH_SCORE)] - 3),
                       ha='right')
        else:
            g.annotate(utils.plot_annotations[idx],
                       (row[('all', clemmetrics.METRIC_PLAYED)] + 2,
                        row[('all', clemmetrics.BENCH_SCORE)] - 3),
                       ha='left')
    plt.xlim(-10, 110)
    plt.ylim(-10, 110)
    plt.xlabel('% played')
    g.xaxis.set_label_coords(0.9, 0.06)
    plt.ylabel('quality score')
    g.yaxis.set_label_coords(0.06, 0.85)
    plt.grid(alpha=0.5)
    plt.title('clembench Overview')

    name = 'played-quality-paper'
    path = utils.create_file_name('', 'episode', 'plots', name, 'pdf')
    utils.savefig(path)


def plot_clem_score(df_clem):
    """Create barplot with the clem score for each mode."""
    fig = plt.figure(figsize=(7, 5))
    sns.barplot(df_clem, x='model', y='clemscore', color='slategray')
    plt.ylim(-5, 105)
    plt.grid(alpha=0.5)
    name = 'clemscore'
    path = utils.create_file_name('', 'episode', 'plots', name, 'pdf')
    utils.savefig(path)


def ccw_sort(p):
    """Put the nodes in clockwise order."""
    # from https://stackoverflow.com/a/44143444 by user ImportanceOfBeingEarnest
    p = np.array(p)
    mean = np.mean(p, axis=0)
    d = p - mean
    s = np.arctan2(d[:, 0], d[:, 1])
    return p[np.argsort(s), :]


def plot_polygons(df):
    """Plot polygons for each game."""
    fig, ax_list = plt.subplots(3, 4, figsize=(9, 6), sharey=True, sharex=True)
    axs = ax_list.flatten()

    for n, (model, model_df) in enumerate(df.groupby('model')):
        rows = model_df.metric.isin(utils.MAIN_METRICS)
        df_aux = model_df[rows]
        df_aux = (df_aux.pivot(index=['game', 'experiment', 'episode'],
                               columns='metric',
                               values='value')
                        .reset_index())
        df_aux = df_aux.drop(['episode'], axis=1)

        # create the x and y coordinates for each game
        dots = []
        for game, game_df in df_aux.groupby('game'):
            overall_means = (game_df.mean(numeric_only=True)
                                    .fillna(0))
            # replace missing score by 0 when all aborted
            played = overall_means[clemmetrics.METRIC_PLAYED] * 100
            score = overall_means[clemmetrics.BENCH_SCORE]
            dots.append((game, played, score))
        labels, played, scores = zip(*dots)
        # put them in a good order for the polygon
        edges = ccw_sort(list(zip(played, scores)))

        # create the polygon and draw it
        polygon = Polygon(edges, facecolor='lightgray')
        axs[n].add_patch(polygon)

        legend = True if n == 10 else False
        g = sns.scatterplot(x=played,
                            y=scores,
                            hue=labels,
                            style=labels,
                            hue_order=utils.COLUMN_ORDER[1:],
                            style_order=utils.COLUMN_ORDER[1:],
                            s=80,
                            ax=axs[n],
                            legend=legend)
        axs[n].set_xlim(-5, 105)
        axs[n].set_ylim(-5, 105)
        axs[n].set_ylabel('avr. quality')
        axs[n].set_xlabel('% played')
        axs[n].set_title(model, y=0.84)

    fig.legend(loc='lower right', bbox_to_anchor=(0.98, 0.08))
    axs[10].legend().set_visible(False)
    fig.delaxes(axs[11])
    plt.tight_layout()
    path = utils.create_file_name('', 'episode', 'plots', 'polygons', 'pdf')
    utils.savefig(path)


def plot_lines(df):
    """Plot lineplot comparing models across experiments."""
    aux_df = (df[df.metric == clemmetrics.BENCH_SCORE]
              .groupby(['game', 'model', 'experiment'])
              .mean(numeric_only=True)
              .reset_index())
    g = sns.catplot(aux_df,
                    row='game',
                    row_order=utils.COLUMN_ORDER[1:],
                    hue='model',
                    sharey=False,
                    sharex=False,
                    x='experiment',
                    y='value',
                    kind='point',
                    aspect=2.5)
    g.set(ylim=(-5, 105))
    sns.move_legend(g, loc="upper center", ncols=len(aux_df.model.unique()),
                    bbox_to_anchor=(0.5, 1.03))
    plt.tight_layout()
    path = utils.create_file_name('', 'episode', 'plots', 'lines', 'pdf')
    utils.savefig(path)


def plot_escores_game(act_df, game):
    """Plot bar plots for a game, episode score by experiment and metric."""
    g = sns.catplot(act_df,
                    row='metric',
                    col='experiment',
                    col_order=sorted(act_df.experiment.unique()),
                    x='model',
                    order=utils.ROW_ORDER[:-2],
                    y='value',
                    # we had to set sharex to false to make sure that
                    # xticklabels are shown
                    # add redundant hue to ensure colors are the same
                    hue='model',
                    dodge=False,
                    kind='bar',
                    sharey=False,
                    sharex=False,
                    aspect=1.4)
    plt.suptitle(f'Overview of Episode Scores: {game}', y=1.)
    path = utils.create_file_name(game, 'episode', 'plots', '_overview', 'pdf')
    utils.savefig(path)


def plot_escores_line_game(act_df, game):
    """Plot lineplot for a game, across experiments."""
    act_df = (act_df.groupby(['model', 'experiment', 'metric'])
                    .mean(numeric_only=True)
                    .reset_index())
    g = sns.catplot(act_df,
                    row='metric',
                    hue='model',
                    sharey=False,
                    sharex=False,
                    x='experiment',
                    order=sorted(act_df.experiment.unique()),
                    y='value',
                    kind='point',
                    aspect=1.5)
    sns.move_legend(g, loc="upper center", ncols=len(act_df.model.unique()),
                    bbox_to_anchor=(0.5, 1.03))
    plt.suptitle(f'Overview of Episode Scores: {game}', y=1.)
    name = '_overview-lines'
    path = utils.create_file_name(game, 'episode', 'plots', name, 'pdf')
    utils.savefig(path)


def plot_escores_game_metric(metric_df, game, metric, lims):
    """Plot bar plots for a game and metric, episode score by experiment."""
    g = sns.catplot(metric_df,
                    col='experiment',
                    col_order=sorted(metric_df.experiment.unique()),
                    x='model',
                    order=utils.ROW_ORDER[:-2],
                    y='value',
                    # we had to set sharex to false to make sure that
                    # xticklabels are shown
                    # add redundant hue to ensure colors are the same
                    hue='model',
                    hue_order=utils.ROW_ORDER[:-2],
                    dodge=False,
                    kind='bar',
                    sharey=False,
                    sharex=False,
                    aspect=1.4)
    g.set(ylim=lims)
    plt.suptitle(f'Overview of Episode Scores: {game} | {metric}', y=1.1)
    name = f'_overview_{metric}'
    path = utils.create_file_name(game, 'episode', 'plots', name, 'pdf')
    utils.savefig(path)


def plot_escores_line_game_metric(metric_df, game, metric, lims):
    """Plot lineplot for a game, across experiments."""
    g = sns.catplot(metric_df,
                    hue='model',
                    hue_order=utils.ROW_ORDER[:-2],
                    sharey=False,
                    sharex=False,
                    x='experiment',
                    order=sorted(metric_df.experiment.unique()),
                    y='value',
                    kind='point',
                    errorbar=None,
                    aspect=1.5)
    g.set(ylim=lims)
    sns.move_legend(g, loc="center right", bbox_to_anchor=(1.2, 0.5))
    plt.suptitle(f'Overview of Episode Scores: {game}', y=1.1)
    name = f'_overview-lines_{metric}'
    path = utils.create_file_name(game, 'episode', 'plots', name, 'pdf')
    utils.savefig(path)


def plot_tscores_game(act_df, game):
    """Plot line plots for each game, turn score by experiment and metric."""
    g = sns.FacetGrid(act_df,
                      row='metric',
                      col='experiment',
                      col_order=sorted(act_df.experiment.unique()),
                      hue='model',
                      hue_order=utils.ROW_ORDER[:-2],
                      sharey=False,
                      sharex=False,
                      legend_out=True,
                      aspect=1.6)
    g.map(sns.lineplot, 'turn', 'value')
    g.add_legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.suptitle(f'Overview of Turn Scores: {game}', y=1.1)
    path = utils.create_file_name(game, 'turn', 'plots', '_overview', 'pdf')
    utils.savefig(path)


def plot_tscores_game_metric(metric_df, game, metric, lims):
    """Plot line plots for each game and metric, turn score by experiment."""
    g = sns.FacetGrid(metric_df,
                      col='experiment',
                      col_order=sorted(metric_df.experiment.unique()),
                      hue='model',
                      hue_order=utils.ROW_ORDER[:-2],
                      sharey=False,
                      sharex=False,
                      legend_out=True,
                      aspect=1.4)
    g.map(sns.lineplot, 'turn', 'value')
    g.add_legend(loc="center left", bbox_to_anchor=(1, 0.5))
    g.set(ylim=lims)
    plt.suptitle(f'Overview of Turn Scores: {game} | {metric}', y=1.1)
    name = f'_overview_{metric}'
    path = utils.create_file_name(game, 'turn', 'plots', name, 'pdf')
    utils.savefig(path)
