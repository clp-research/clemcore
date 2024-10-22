"""
Script to get a list of all episodes that differ between two languages in given metrics.
A raw.csv including all raw results of the languages being compared must exist in the overgiven results path (Can be created by multiling_single_game_evaluation.py).
"""

import os

import pandas as pd

from multiling_eval_utils import short_names


def get_differences(results_path, game, langs, compare_metrics):
    df_raw = pd.read_csv(f"{results_path}/raw.csv", index_col=0)

    df1 = (df_raw.loc[
        (df_raw["lang"]==langs[0]) \
            & (df_raw["game"]==game) \
            & (df_raw["metric"].isin(compare_metrics))
        ].drop("lang", axis=1)
        .reset_index(drop=True))

    df2 = (df_raw.loc[
        (df_raw["lang"]==langs[1]) \
            & (df_raw["game"]==game) \
            & (df_raw["metric"].isin(compare_metrics))
        ].drop("lang", axis=1)
        .reset_index(drop=True))

    assert len(df1) == len(df2)

    df_compare = df1.compare(df2, result_names=langs)  # is a multiindex dataframe
    df_compare.columns = df_compare.columns.droplevel(0)  # remove outer level of multiindex
    print(f"Found {len(df_compare)} differences comparing {langs} in the {game}. Compared metrics: {compare_metrics}")

    df_diff = df1[df1.index.isin(df_compare.index)]
    df_diff = df_diff.drop(["value"], axis=1)
    df_diff = pd.concat([df_diff, df_compare], axis=1)
    # use column values to create path to transcript
    link_html = '<a href={} target="_blank">transcript.html</a>'
    df_diff["link1"] = df_diff.apply(lambda x: link_html.format(os.path.join(
        langs[0], x["model"], game, x["experiment"],
        x["episode"], "transcript.html"
        )), axis=1)
    df_diff["link2"] = df_diff.apply(lambda x: link_html.format(os.path.join(
        langs[1], x["model"], game, x["experiment"],
        x["episode"], "transcript.html"
        )), axis=1)
    # use model short names
    df_diff["model"] = df_diff["model"].replace(to_replace=r'(.+)-t0.0--.+', value=r'\1', regex=True).replace(short_names)
    df_diff.to_html(os.path.join(results_path, f"diff_{langs[0]}_{langs[1]}.html"), escape=False)
    df_diff.drop(["link1", "link2"], axis=1).to_csv(os.path.join(results_path, f"diff_{langs[0]}_{langs[1]}.csv"))

    print(f"Saved html to {results_path}")


if __name__ == "__main__":
    results_path = "results/v1.5_multiling_liberal"
    game = "imagegame"
    langs = ("tr", "tr_google")
    compare_metrics = ["F1"]
    get_differences(results_path, game, langs, compare_metrics)
