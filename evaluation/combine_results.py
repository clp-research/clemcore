"""
Script to combine tables with invalid answers created by answer_statistics.py.
Takes files from the model level and creates new file on the language level.

Output dfs look like:
    modelname1              modelname2              ...
    Player 2 Text  count    Player 2 Text  count
    <str>          <int>    <str>          <int>
    ...

"""

import glob
from pathlib import Path

import pandas as pd


RESULTS_PATH = "results/v1.5_multiling"
ANSWERS_FILE = "player2_invalid_answers" # extension is added automatically


# on the language level
lang_dirs = glob.glob(f"{RESULTS_PATH}/*/")
for lang_dir in lang_dirs:
    lang = lang_dir.split("/")[-2]
    # skip non-language directories
    if not (
        (len(lang) == 2) or (len(lang.split('_')[0]) == 2)
        ):  # machine translations have identifiers such as 'de_google'
        continue

    # on the model level
    dfs = {}
    file_paths = glob.glob(f"{RESULTS_PATH}/{lang}/*/{ANSWERS_FILE}.csv")
    for path in file_paths:
        model = path.split("/")[-2].split("--")[0]
        df = pd.read_csv(path, dtype={"count": "int64"})
        dfs[model] = df
    # df with multiindex column
    df_lang = pd.concat(dfs.values(), axis=1, keys=dfs.keys())

    df_lang.to_csv(Path(lang_dir) / f"{ANSWERS_FILE}.csv", index=False, float_format="%.0f")
    df_lang.to_html(Path(lang_dir) / f"{ANSWERS_FILE}.html", index=False, na_rep="", float_format="%.0f")