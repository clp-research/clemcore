"""
Clembench Evaluation

This script produces plots and tables based on all results.

Levels Set-Up: How results are grouped

| config    | play              |
|-----------|-------------------|
| instance  | episode           |
| dataset   | experiment        |
| game  |                   |
| benchmark | benchmark results |

Additional Dimensions

- metrics: turn-level scores, episode-level scores, dispersion stats
- player(s):  model_1, model_2
- language: TBD

The plots and tables are stored according to the records structure, in an
additional directory called results_eval containing two directories
(episode-level and turn-level), each with two directories (plots and tables).
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics
from tqdm import tqdm

import evaluation.evalutils as utils
import evaluation.plotting as plotting
import evaluation.makingtables as tables
import clemgame.metrics as clemmetrics

sns.set(font='Futura', style="white")

parser = argparse.ArgumentParser()
parser.add_argument('--no_plots', action='store_true',
                    help='Do not generate plots.')
args = parser.parse_args()

if args.no_plots:
    print('Only tables will be created, all plots skipped!')

scores = utils.load_scores()
utils.create_eval_tree(scores.keys())

df_turn_scores = utils.build_df_turn_scores(scores)
df_episode_scores = utils.build_df_episode_scores(scores)

# Create the PLAYED variable
aux = df_episode_scores[df_episode_scores["metric"] == "Aborted"].copy()
aux["metric"] = clemmetrics.METRIC_PLAYED
aux["value"] = 1 - aux["value"]
# We need ignore_index=True to reset the indices (otherwise we have duplicates)
df_episode_scores = pd.concat([df_episode_scores, aux], ignore_index=True)

GAMES = df_turn_scores['game'].unique().tolist()
MODELS = df_turn_scores['model'].unique().tolist()
EXPERIMENTS = list(set([x[:3] for x in scores]))
EPISODES = scores.keys()
ZERO_ONE_EPISODE_SCORES = utils.get_metrics_in_zero_one(df_episode_scores)
ZERO_ONE_TURN_SCORES = utils.get_metrics_in_zero_one(df_turn_scores)

# Save tables with raw scores
utils.save_raw_scores(df_turn_scores, df_episode_scores, scores)

for key, value in utils.short_names.items():
    df_turn_scores['model'] = df_turn_scores['model'].str.replace(key, value)
    df_episode_scores['model'] = df_episode_scores['model'].str.replace(key, value)

# ------------------------ Evaluation of the Benchmark ------------------------
# 
# Main tables with overall results. Visualising results for all games,
# all models, all experiments, all episodes.
print('\n Evaluating the whole benchmark...')

# ---------------------- Benchmark: Espiode-Level Scores ----------------------
#
# (1) Tables summarising all results, aggregated and split.
# (2) Plots summarising all results, aggregated and split.
#       (a) only scores in the interval [0,1]
#       (b) only scores not in the interval [0,1]
#
print('\t Generating tables and plots for episode-level scores...')

# Paper table
df_paper = tables.save_paper_table(df_episode_scores)

# Clem score table
df_clem = tables.save_clem_score_table(df_paper)

# Games vs. models with episode scores dispersion metrics across 
# all episodes and experiments
bench_table = tables.make_stats_table(df_episode_scores)

# Games vs. models with episode scores dispersion metrics across
# all episodes, split by experiments:
tables.save_detailed_table(df_episode_scores)

# Plots
if not args.no_plots:
    df_01, df_other = utils.filter_metrics_in_zero_one(df_episode_scores,
                                                       ZERO_ONE_EPISODE_SCORES)
    if not df_01.empty:
        plotting.plot_escore_benchmark(df_01, '_in01')       # (2a)
    if not df_other.empty:
        plotting.plot_escore_benchmark(df_other, '_other')   # (2b)

    # Stacked bar plots with success, lose and aborted
    # micro average
    plotting.plot_stacked_micro_bar(df_episode_scores, df_clem)
    # macro_average
    plotting.plot_stacked_macro_bar(df_episode_scores, df_clem)

    # polygons
    plotting.plot_polygons(df_episode_scores)

    # scatter plots with (% played, quality score) for each model
    # we generate for the benchmark, for each game and for each experiment
    plotting.plot_paper_scatter(df_paper)

    # lineplots with quality score for each model across experiments
    plotting.plot_lines(df_episode_scores)

    # barplots with clem score for each model 
    plotting.plot_clem_score(df_clem)

# ----------------------- Benchmark: Turn-Level Scores ------------------------
#
# - N/A

# -------------------- Evaluation of Games (by model) --------------------
#
# Visualising results for each game separately. It contains multiple 
# experiments, which contain multiple related episodes.
print('\n Evaluating games:')

# One table for each game aggregated by experiment
tables.make_overview_by_game(df_episode_scores)

# One detailed table for each game
tables.make_detailed_overview_by_game(df_episode_scores)

# Plots
if not args.no_plots:
    for game in tqdm(GAMES, desc="Generating game-specific plots"):
        act_df = df_episode_scores[df_episode_scores.game == game]
        # overview of all episode scores
        plotting.plot_escores_game(act_df, game)
        plotting.plot_escores_line_game(act_df, game)
        # one plot for each metric
        for metric, metric_df in act_df.groupby('metric'):
            lims = utils.get_metric_lims(metric, ZERO_ONE_EPISODE_SCORES)
            plotting.plot_escores_game_metric(metric_df, game, metric, lims)
            plotting.plot_escores_line_game_metric(metric_df, game, metric, lims)
        # overview of turn scores
        # there should not be nans, removing them here for now
        act_df = df_turn_scores[df_turn_scores.game == game].dropna()
        # overview of all turn scores
        plotting.plot_tscores_game(act_df, game)
        # one plot for each metric
        for metric, metric_df in act_df.groupby('metric'):
            lims = utils.get_metric_lims(metric, ZERO_ONE_TURN_SCORES)
            plotting.plot_tscores_game_metric(metric_df, game, metric, lims)
