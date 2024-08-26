# tested with Linux OpenSUSE Leap 15.4
# Python 3.9.19

"""
Script for calculation of Kendalls Tau.

`pip install scipy` necessary.
"""
from scipy.stats import kendalltau

import pandas as pd
from typing import Union, Dict
import matplotlib.pyplot as plt


# source: https://en.wikipedia.org/wiki/Wikipedia:List_of_Wikipedias (accessed: 1 Jul 24)
wikipedia_articles = {
        "de": 2922481,
        "en": 6843684,
        "es": 1963407,
        "ru": 1986725,
        "te": 96780,
        "tk": 6864,
        "tr": 613039
    }

gpt4_report_ranking = {
        "de": 83.7,
        "en": 85.5,
        "es": 84,
        "ru": 82.7,
        "te": 62,
        "tr": 80
    }


def calc_kendalltau(a:Union[Dict, pd.Series], b:Union[Dict, pd.Series]):
    # filter out entries that contain nan values
    a_filtered = {k: v for k, v in a.items() if not pd.isnull(v)}
    b_filtered = {k: v for k, v in b.items() if not pd.isnull(v)}
    # filter out entries that don't occur in both dicts
    a_filtered = {k: v for k, v in a_filtered.items() if k in b_filtered}
    b_filtered = {k: v for k, v in b_filtered.items() if k in a_filtered}
    # ensure same order of both dicts (sorted by keys)
    a_filtered = dict(sorted(a_filtered.items()))
    b_filtered = dict(sorted(b_filtered.items()))
    return kendalltau(list(a_filtered.values()), list(b_filtered.values()))


if __name__ == "__main__":
    print(calc_kendalltau(wikipedia_articles, gpt4_report_ranking))
    tau, _ = calc_kendalltau(wikipedia_articles, gpt4_report_ranking)
    print(f"Kendall's Tau: {tau}")

    # using pandas as in evaluation/multiling_single_game_evaluation.py:
    df_scores = pd.DataFrame([wikipedia_articles, gpt4_report_ranking]).transpose()
    tau, _ = calc_kendalltau(df_scores[0], df_scores[1])
    print(f"Kendall's Tau: {tau}")
