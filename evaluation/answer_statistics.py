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

import pandas as pd
import numpy as np
import re
from tqdm import tqdm

from multiling_eval_utils import short_names


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


def calc_statistics(results_path):
    """
    Calculate statistics for all models in results_path.

    :param results_path: Path with model dirs.
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
        p2_answers = overview["Player 2 Parsed Answer"]                   # contains parsed answer without tag. Or 'invalid generated choice'. Or Nan when aborted at player A.
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

        if p2_answers.dropna().loc[p2_answers != "Invalid generated choice"].empty:
            # leave player B statistics empty when there are only nan or invalid answers
            statistics[model_shortname].update({
                "p2": {
                    "Choices Distribution": {},
                    "Grid Consistency": None,
                    "Position Consistency": None,
                }
            })
            continue

        # Distribution of choices (first/second/third)
        p2_choices_dist = p2_answers.dropna().loc[
            p2_answers != "Invalid generated choice"  # ignore invalid and non-existant content
            ].value_counts().to_dict()

        # Position consistency (does the model pick same number for same grid?)
        n_consistent_pos = 0

        # Consistency in grid selection
        consistant_orders = {
            ("1", "2", "3"),  # target grid is always in this order
            ("2", "1", "2"),  # distracor grid 1 is always in this order
            ("3", "3", "1")   # distracor grid 2 is always in this order
            }
        n_consistent_grid = 0
        n_complete_triplets = 0
        for triplet in np.reshape(p2_answers, newshape=((n_triplets*2)-1, 3)):
            if (pd.isnull(triplet).any()) or ("Invalid generated choice" in triplet):
                # skip triplets where answers are missing
                # every second triplet is empty because of the format of overview
                # some are empty because of abort at player 1
                continue
            if tuple(triplet) in consistant_orders:
                n_consistent_grid += 1
            elif triplet[0] == triplet[1] == triplet[2]:
                n_consistent_pos += 1
            n_complete_triplets += 1
        try:
            p2_grid_consistency = n_consistent_grid / n_complete_triplets  # expectation by chance is 1/9
            p2_pos_consistency = n_consistent_pos / n_complete_triplets  # expectation by chance is 1/9
        except ZeroDivisionError:
            p2_grid_consistency = None
            p2_pos_consistency = None
        statistics[model_shortname].update({
                "p2": {
                    "Choices Distribution": p2_choices_dist,  # is dict
                    "Grid Consistency": p2_grid_consistency,
                    "Position Consistency": p2_pos_consistency,
                }
            })

    return dict(sorted(statistics.items()))


def calc_statistics_for_all_langs(results_path):
    """
    Calculate statistics for all languages in results_path.

    :param results_path: Path were language dirs are.
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
        statistics[lang] = calc_statistics(lang_dir)
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




if __name__ == "__main__":
    results_path = "results/v1.5_multiling"
    # get excel overviews
    # execute_create_exel_overview_for_all_langs(results_path)

    # statistics for one language
    # statistics = calc_statistics(results_path + "/ru_google")

    if not os.path.exists(os.path.join(results_path, "referencegame_additional_statistics.json")):
        statistics = calc_statistics_for_all_langs(results_path)
        with open(os.path.join(results_path, "referencegame_additional_statistics.json"), "w") as file:
            json.dump(statistics, file, indent=4)
    else:
        with open(os.path.join(results_path, "referencegame_additional_statistics.json")) as file:
            statistics = json.load(file)

    create_tables(statistics, os.path.join(results_path, "multiling_eval/referencegame"))
