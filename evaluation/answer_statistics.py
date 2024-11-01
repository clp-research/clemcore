# Author: Sandra Neuhäußer
# Python 3.9.19
# Linux OpenSUSE Leap 15.4

"""
Calculate interesting answer statistics about the referencegame
using excel overviews created by create_excel_overview.py
"""

import os
import glob
import subprocess
from ast import literal_eval
import json
import matplotlib.pyplot as plt
import argparse

import pandas as pd
import numpy as np
import re
from tqdm import tqdm

from multiling_eval_utils import short_names


def get_arguments(parser):
    parser.add_argument("results_path", help="Path where the result files are stored by language")
    parser.add_argument("-o", "--output_path", help="Path to output directory. Default: results_path + 'multiling_eval/referencegame/answer_statistics'", default="")

    parser.add_argument("-s", "--calc_statistics", action="store_true", help="If given, calculates statistics for all langs and creates output tables in 'output_path'. Option 'create_excel_overviews' must have been executed or must be given.")

    parser.add_argument("-e", "--create_excel_overviews", action="store_true", help="If given, creates excel overviews for all languages. Uses 'create_excel_overview.py'. Creates output in each model directory.")

    parser.add_argument("-i", "--show_p2_invalid", action="store_true", help="If given, creates csv files to each model dir with the invalid answers player B gave. Only works together with option 'calc_statistics'.")

    parser.add_argument("-f", "--multiling_results_file", help="Overgive path to csv that contains df with main results for all model-lang-combinations. If given, it is combined with the df results_consistent.csv. Option 'calc_statistics' must have been executed or must be given.")

    # This option is for debugging. create_tables is automatically done when calling calc_statistics.
    parser.add_argument("-t", "--create_tables", action="store_true", help="(For debugging) If given, creates output tables in 'output_path'. Reads in 'statistics.json'. Option 'calc_statistics' must have been executed before.")

    return parser.parse_args()


def verify_args(args):
    if not os.path.exists(args.results_path):
        raise FileNotFoundError(f"results_path '{args.results_path}' does not exist.")

    if args.calc_statistics or args.create_tables or args.multiling_results_file:
        if not args.output_path:
            args.output_path = os.path.join(args.results_path, "multiling_eval/referencegame/answer_statistics")
            if not os.path.exists(args.output_path):
                os.makedirs(args.output_path)

    if args.create_tables and not args.calc_statistics:
        statistics_path = os.path.join(args.output_path, "statistics.json")
        if not os.path.exists(statistics_path):
            raise FileNotFoundError(f"'{statistics_path}' not found. Execute option 'calc_statistics' first.")

    if args.multiling_results_file:
        if not os.path.exists(args.multiling_results_file):
            raise FileNotFoundError(f"'{args.multiling_results_file}' not found.")
        if not (args.calc_statistics or args.create_tables):
            results_consistent_path = os.path.join(args.output_path, "results_consistent.csv")
            if not os.path.exists(results_consistent_path):
                raise FileNotFoundError(f"'{results_consistent_path}' not found. Execute option 'calc_statistics' first.")


def get_meaning(string, options):
    """
    :param options: nested list -> [["first", "1"], ["second", ...]...]
    """
    if not isinstance(string, str):
        return string
    for opt in options:
        if re.match("|".join(opt), string, flags=re.IGNORECASE):
            assert opt[1].isdigit()
            return opt[1]
    return string


def calc_statistics(results_path, show_p2_invalid: bool):
    """
    Calculate statistics for all models in results_path.

    :param results_path: Path with model dirs.
    :param show_p2_invalid: If true, save csv files with player B's invalid answers to results_path/model.
    """
    assert os.path.isdir(results_path)

    model_dirs = glob.glob(f"{results_path}/*/")
    statistics = {}
    for model_dir in model_dirs:
        model_name = model_dir.split("/")[-2]
        model_shortname = short_names[re.sub(r'-t0.0--.+', "", model_name)]

        try:
            overview = pd.read_excel(os.path.join(model_dir, f"{model_name}.xlsx"), dtype=object)
        except FileNotFoundError:
            raise FileNotFoundError("Execute 'calc_statistics_for_all_langs()' or 'create_excel_overview.py' to receive the missing file.")
        p1_expressions = overview["Player 1 Parsed Expression"].dropna()  # contains parsed expressions without tag. Or 'invalid generated expression'
        p1_output = overview["Player 1 Text"].dropna()                    # contains whole answer
        p2_answers = overview["Player 2 Parsed Answer"]                   # contains parsed answer without tag. Or 'Invalid generated choice'. Or Nan when aborted at player A.
        # invalid answers of p2
        p2_answers_invalid = overview["Player 2 Text"].loc[overview["Player 2 Parsed Answer"] == "Invalid generated choice"].dropna().value_counts()

        # convert p2 literal answers to their underlying meaning
        p2_options = tuple(overview["Ground Truth"][:3].apply(literal_eval))
        p2_answers = p2_answers.apply(get_meaning,
                                    options=p2_options)

        assert len(p1_expressions) == 180
        assert len(p1_expressions) == ((len(p2_answers) + 3) / 2) == len(p1_output)

        n_episodes = len(p1_expressions)
        n_triplets = int(n_episodes / 3)


        # Statistics player A

        # Unique output ratio: unique expressions / unique grids
        unique_output = p1_output.unique()
        # higher than 1 indicates the model generates different expressions for same grids
        # lower than 1 indicates the model generates same expressions for different grids
        p1_unique_output_count = len(unique_output)
        p1_unique_output_ratio = len(unique_output) / n_triplets

        # Consistency (same output for same grid)
        n_consistent = 0
        for triplet in np.reshape(p1_output, newshape=(n_triplets, 3)):
            if triplet[0] == triplet[1] == triplet[2]:
                n_consistent += 1
        p1_consistency = n_consistent / n_triplets

        # Average expression length in words (filtering out invalid)
        p1_average_expression_length = p1_expressions.loc[
            p1_expressions != "Invalid generated expression"
            ].str.split(" ").str.len().mean()
        p1_average_expression_length = None if pd.isnull(p1_average_expression_length) else p1_average_expression_length  # json does not support nan

        # Average output length in words (including invalid output)
        p1_average_output_length = p1_output.str.split(" ").str.len().mean()

        statistics[model_shortname] = {
            "p1": {
                "Unique Output Count": p1_unique_output_count,
                "Unique Output Ratio": p1_unique_output_ratio,
                "Output Consistency": p1_consistency,
                "Avg. Output Length": p1_average_output_length,
                "Avg. Expression Length": p1_average_expression_length
            }
        }

        # Statistics player B

        # Save invalid answers to csv file
        if show_p2_invalid:
            p2_answers_invalid.to_csv(os.path.join(model_dir, "player2_invalid_answers.csv"))

        if p2_answers.dropna().loc[p2_answers != "Invalid generated choice"].empty:
            # leave player B statistics empty when there are only nan or invalid answers
            statistics[model_shortname].update({
                "p2": {
                    "Choices Distribution": {},
                    "Grid Consistency": None,
                    "Position Consistency": None,
                    "Invalid Answers": {"Number Unique": p2_answers_invalid.size, "Frequent Answers": p2_answers_invalid.head(5).to_dict()},
                },
                "% Played Consistent": 0.0,
                "Quality Score Consistent": None,
            })
            continue

        # Distribution of choices (first/second/third)
        p2_choices_dist = p2_answers.dropna().loc[
            p2_answers != "Invalid generated choice"  # ignore invalid and non-existant content
            ].value_counts().to_dict()

        # Position consistency (does the model pick same number for same grid?)
        n_consistent_pos = 0

        # Consistency in grid selection
        consistent_orders = [
            ("1", "2", "3"),  # target grid is always in this order
            ("2", "1", "2"),  # distracor grid 1 is always in this order
            ("3", "3", "1")   # distracor grid 2 is always in this order
        ]
        # number of consistent triplets
        n_consistent_grid = 0
        # number of correct and consistent triplets
        n_consistent_correct = 0
        # number of complete triplets (all three episodes non-aborted)
        n_complete_triplets = 0
        for triplet in np.reshape(p2_answers, newshape=((n_triplets*2)-1, 3)):
            if (pd.isnull(triplet).any()) or ("Invalid generated choice" in triplet):
                # skip triplets where answers are missing
                # every second triplet is empty because of the format of overview
                # some are empty because of abort at player 1
                # skip triplets where some of player B's answers were invalid
                continue
            if tuple(triplet) == consistent_orders[0]:
                n_consistent_correct += 1
            if tuple(triplet) in consistent_orders:
                n_consistent_grid += 1
            elif triplet[0] == triplet[1] == triplet[2]:
                n_consistent_pos += 1
            n_complete_triplets += 1

        # Consistent % Played: percentage of triplets where all three episodes were not aborted
        played_consistent = (n_complete_triplets / n_triplets) * 100

        try:
            p2_grid_consistency = n_consistent_grid / n_complete_triplets  # expectation by chance is 1/9
            p2_pos_consistency = n_consistent_pos / n_complete_triplets  # expectation by chance is 1/9
            # Consistent Quality Score: percentage of consistently correct triplets
            quality_score_consistent = (n_consistent_correct / n_complete_triplets) * 100
        except ZeroDivisionError:
            p2_grid_consistency = None
            p2_pos_consistency = None
            quality_score_consistent = None
        statistics[model_shortname].update({
                "p2": {
                    "Choices Distribution": p2_choices_dist,  # is dict
                    "Grid Consistency": p2_grid_consistency,
                    "Position Consistency": p2_pos_consistency,
                    "Invalid Answers": {"Number Unique": p2_answers_invalid.size, "Frequent Answers": p2_answers_invalid.head(5).to_dict()},
                },
                "% Played Consistent": played_consistent,
                "Quality Score Consistent": quality_score_consistent,
            })

    return dict(sorted(statistics.items()))


def calc_statistics_for_all_langs(results_path, show_p2_invalid: bool):
    """
    Calculate statistics for all languages in results_path.

    :param results_path: Path were language dirs are.
    :param show_p2_invalid: If true, save csv files with player B's invalid answers to results_path/lang/model.
    """
    lang_dirs = glob.glob(f"{results_path}/*/")
    statistics = {}
    for lang_dir in tqdm(lang_dirs, desc="Calculating statistics"):
        lang = lang_dir.split("/")[-2]
        # skip non-language directories
        if not (
            (len(lang) == 2) or (len(lang.split('_')[0]) == 2)
            ):  # machine translations have identifiers such as 'de_google'
            continue
        statistics[lang] = calc_statistics(lang_dir, show_p2_invalid=show_p2_invalid)
    return dict(sorted(statistics.items()))


def execute_create_exel_overview_for_all_langs(results_path):
    """
    :param results_path: Path were language dirs are.
    """
    lang_dirs = glob.glob(f"{results_path}/*/")
    for lang_dir in tqdm(lang_dirs, desc="Creating overviews"):
        subprocess.run(
            [f"python3 evaluation/create_excel_overview.py {lang_dir} -game_name referencegame"],
            shell=True,
            stdout = subprocess.DEVNULL
            )

def create_tables(statistics, path):
    """
    Create tables to compare statistics across languages.

    :param statistics: dict created by calc_statistics_for_all_langs().
    :param path: path where output is written.
    """
    # Player A

    # consistency df
    temp = {}
    for lang, models in statistics.items():
        temp[lang] = {}
        for model, players in models.items():
            temp[lang][(model, "Unique Output Count")] = players["p1"]["Unique Output Count"]
            temp[lang][(model, "Unique Output Ratio")] = players["p1"]["Unique Output Ratio"]
            temp[lang][(model, "Output Consistency")] = players["p1"]["Output Consistency"]
    df_p1_consistency = pd.DataFrame(temp)
    df_p1_consistency.columns = df_p1_consistency.columns.str.replace("google", "")
    df_p1_consistency = df_p1_consistency.round(2)
    df_p1_consistency.to_html(os.path.join(path, "p1_consistency.html"))
    df_p1_consistency.to_latex(os.path.join(path, "p1_consistency.tex"), float_format="%.2f")

    # answer length df
    temp = {}
    for lang, models in statistics.items():
        temp[lang] = {}
        for model, players in models.items():
            temp[lang][(model, "Avg. Output Length")] = players["p1"]["Avg. Output Length"]
            temp[lang][(model, "Avg. Expression Length")] = players["p1"]["Avg. Expression Length"]
    df_p1_words = pd.DataFrame(temp)
    df_p1_words.columns = df_p1_words.columns.str.replace("google", "")
    df_p1_words = df_p1_words.round(2)
    df_p1_words.to_html(os.path.join(path, "p1_word_count.html"))
    df_p1_words.to_latex(os.path.join(path, "p1_word_count.tex"), float_format="%.2f")

    # Player B
    temp = {}
    for lang, models in statistics.items():
        temp[lang] = {}
        for model, players in models.items():
            temp[lang][(model, "Grid Consistency")] = players["p2"]["Grid Consistency"]
            temp[lang][(model, "Position Consistency")] = players["p2"]["Position Consistency"]
    df_p2_cosistency = pd.DataFrame(temp)
    df_p2_cosistency.columns = df_p2_cosistency.columns.str.replace("google", "")
    df_p2_cosistency = df_p2_cosistency.round(2)
    df_p2_cosistency.to_html(os.path.join(path, "p2_consistency.html"))
    df_p2_cosistency.to_latex(os.path.join(path, "p2_consistency.tex"), float_format="%.2f")

    # answer distribution
    temp = {}
    for lang, models in statistics.items():
        temp[lang] = {}
        for model, players in models.items():
            try:
                temp[lang][(model, "first")] = players["p2"]["Choices Distribution"]["1"]
                temp[lang][(model, "second")] = players["p2"]["Choices Distribution"]["2"]
                temp[lang][(model, "third")] = players["p2"]["Choices Distribution"]["3"]
            except KeyError:  # keys might not exist when choices did not appear
                continue  # next model
    df_p2_answer_dist = pd.DataFrame(temp)
    df_p2_answer_dist.columns = df_p2_answer_dist.columns.str.replace("google", "")
    df_p2_answer_dist = df_p2_answer_dist.round(2)
    df_p2_answer_dist.to_html(os.path.join(path, "p2_choice_dist.html"))
    df_p2_answer_dist.to_latex(os.path.join(path, "p2_choice_dist.tex"), float_format="%.2f")
    for model in df_p2_answer_dist.index.levels[0]:
        df = df_p2_answer_dist.loc[model]
        df = df.T
        ax = df.plot(kind="bar")
        plt.xticks(rotation=0)
        fig = ax.get_figure()
        fig.savefig(os.path.join(path, f"p2_choice_dist_{model}.png"))
        plt.close()

    # Consistent Quality Score & Consistent Played
    temp = {"% Played Consistent": {},
            "Quality Score Consistent": {}}
    for lang, models in statistics.items():
        for model, scores in models.items():
            temp["% Played Consistent"][(lang, model)] = scores["% Played Consistent"]
            temp["Quality Score Consistent"][(lang, model)] = scores["Quality Score Consistent"]
    df_scores_consistent = pd.DataFrame(temp)
    df_scores_consistent.index = df_scores_consistent.index.set_names(["lang", "model"])
    df_scores_consistent.columns = df_scores_consistent.columns.str.replace("google", "")
    df_scores_consistent = df_scores_consistent.round(2)
    df_scores_consistent.to_csv(os.path.join(path, "results_consistent.csv"))
    df_scores_consistent.to_html(os.path.join(path, "results_consistent.html"))
    df_scores_consistent.to_latex(os.path.join(path, "results_consistent.tex"), float_format="%.2f")


def combine_results(file_results, file_results_consistent, path_out):
    """
    Creates one df with % Played, Quality Score, % Played Consistent and Quality Score Consistent.

    :param file_results: path to csv that contains df with results for all model-lang-combinations.
    :param file_results_consistent: path to csv that contains df with consistent scores.
    """
    df_results = pd.read_csv(file_results, index_col=[0, 1])
    df_results = df_results.sort_index(level=[0, 1], ascending=[True, True])
    df_results_consistent = pd.read_csv(file_results_consistent, index_col=[0, 1])
    df_results_consistent = df_results_consistent.sort_index(level=[0, 1], ascending=[True, True])

    df_out = pd.concat([df_results, df_results_consistent], axis=1)
    df_out.rename(columns={"clemscore (Played * Success)": "clemscore", "% Success (of Played)": "Quality Score"}, inplace=True)
    df_out.drop(columns="Aborted at Player 1 (of Aborted)", inplace=True)

    df_out.to_csv(os.path.join(path_out, "results_combined.csv"))
    df_out.to_html(os.path.join(path_out, "results_combined.html"))
    df_out.to_latex(os.path.join(path_out, "results_combined.tex"), float_format="%.2f")


if __name__ == "__main__":
    args = get_arguments(argparse.ArgumentParser())
    verify_args(args)

    if args.create_excel_overviews:
        execute_create_exel_overview_for_all_langs(args.results_path)

    if args.calc_statistics:
        statistics = calc_statistics_for_all_langs(args.results_path, show_p2_invalid=args.show_p2_invalid)
        with open(os.path.join(args.output_path, "statistics.json"), "w") as file:
            json.dump(statistics, file, indent=4)

    if args.calc_statistics or args.create_tables:
        with open(os.path.join(args.output_path, "statistics.json")) as file:
            statistics = json.load(file)
        create_tables(statistics, args.output_path)

    if args.multiling_results_file:
        combine_results(file_results=args.multiling_results_file,
                        file_results_consistent=os.path.join(args.output_path, "results_consistent.csv"),
                        path_out=args.output_path)
