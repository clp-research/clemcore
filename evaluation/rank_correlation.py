"""
Script for calculation of Kendalls Tau.
"""
from scipy.stats import kendalltau


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

def calc_kendalltau(a:dict, b:dict):
    # filter out entries that don't occur in both dicts
    a_filtered = {k: v for k, v in a.items() if k in b}
    b_filtered = {k: v for k, v in b.items() if k in a}
    # ensure same order of both dicts (sorted by keys)
    a_filtered = dict(sorted(a_filtered.items()))
    b_filtered = dict(sorted(b_filtered.items()))
    return kendalltau(list(a_filtered.values()), list(b_filtered.values()))


if __name__ == "__main__":
    print(calc_kendalltau(wikipedia_articles, gpt4_report_ranking))
