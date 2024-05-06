import json
import os
import sys

sys.path.append('..')
import glob
from pathlib import Path
import pandas as pd


def create_overview_table(df: pd.DataFrame, categories: list) -> pd.DataFrame:
    """Create multilingual results as a table."""
    scored_df = df[(df.game == 'referencegame') & (df["metric"].isin(["Played", "Success", "Aborted at Player 1"]))]

    # refactor model names for readability
    scored_df = scored_df.replace(to_replace=r'(.+)-t0.0--.+', value=r'\1', regex=True)

    # compute mean metrics
    df_means = (scored_df.groupby(categories)
                .mean(numeric_only=True)
                .reset_index())
    # convert to percentages
    aux_ab_p1 = df_means.loc[df_means.metric == "Aborted at Player 1", 'value']
    aux_played = df_means.loc[df_means.metric == "Played", 'value']
    aux_aborted = (1-aux_played).to_list()
    df_means.loc[df_means.metric == "Aborted at Player 1", 'value'] = (aux_ab_p1/aux_aborted) * 100
    df_means.loc[df_means.metric == "Played", 'value'] *= 100
    df_means.loc[df_means.metric == "Success", 'value'] *= 100

    df_means = df_means.round(2)
    df_means['metric'].replace(
        {"Played": '% Played', "Success": '% Success (of Played)', "Aborted at Player 1": 'Aborted at Player 1 (of Aborted)'},
        inplace=True)

    # make metrics separate columns
    df_means = df_means.pivot(columns=categories[-1], index=categories[:-1])
    df_means = df_means.droplevel(0, axis=1)

    # compute clemscores and add to df
    clemscore = (df_means['% Played'] / 100) * df_means['% Success (of Played)']
    clemscore = clemscore.round(2).to_frame(name=('clemscore (Played * Success)'))
    df_results = pd.concat([clemscore, df_means], axis=1)
    df_results.reset_index(inplace=True)

    return df_results


def save_table(df, path: str, file: str):
    # save table
    # for adapting for a paper
    df.to_latex(Path(path) / f'{file}.tex', float_format="%.2f") # index = False
    # for easy checking in a browser
    df.to_html(Path(path) / f'{file}.html')# index = False
    print(f'\n Saved results into {path}/{file}.html and .tex')


if __name__ == '__main__':
    # TODO: make options command line argument
    compare = True # if true, same language results must be in '../results/v1.5_multiling_liberal'
    detailed = False
    result_path = '../results/v1.5_multiling'

    # collect all language specific results in one dataframe
    df_strict = None
    if compare:
        df_liberal = None
    result_dir = Path(result_path)
    lang_dirs = glob.glob(f"{result_dir}/*/") # the trailing / ensures that only directories are found
    for lang_dir in lang_dirs:
        lang = lang_dir.split("/")[-2]
        assert len(lang) == 2
        raw_file = os.path.join(lang_dir, 'raw.csv')
        assert Path(raw_file).is_file()
        lang_result = pd.read_csv(raw_file, index_col=0)
        lang_result.insert(0, 'lang', lang)
        df_strict = pd.concat([df_strict, lang_result], ignore_index=True)

        if compare:
            raw_file = raw_file.replace("v1.5_multiling", "v1.5_multiling_liberal")
            assert Path(raw_file).is_file()
            lang_result = pd.read_csv(raw_file, index_col=0)
            lang_result.insert(0, 'lang', lang)
            df_liberal = pd.concat([df_liberal, lang_result], ignore_index=True)

    if detailed:
        categories = ['lang', 'model', 'experiment', 'metric'] #detailed by experiment
    else:
        categories = ['lang', 'model', 'metric']

    overview_strict = create_overview_table(df_strict, categories)
    # sort models within language by clemscore
    sorted_df = overview_strict.sort_values(['lang','clemscore (Played * Success)'],ascending=[True,False])
    # extract model order by language for rank correaltion analysis
    model_orders = {}
    languages = sorted_df['lang'].unique()
    for lang in languages:
        models = sorted_df.loc[sorted_df.lang == lang, 'model']
        scores = sorted_df.loc[sorted_df.lang == lang, 'clemscore (Played * Success)']
        model_orders[lang] = zip(models.tolist(), scores.tolist())
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(model_orders, f, ensure_ascii=False)
    save_table(sorted_df.set_index(['lang', 'model']), result_path, 'results_multiling')

    if compare:
        overview_liberal = create_overview_table(df_liberal, categories)
        models = ["command-r-plus", "Llama-3-8b-chat-hf",
                  "Llama-3-70b-chat-hf"]
        selected_liberal = overview_liberal[(overview_liberal["model"].isin(models))].set_index(['lang', 'model'])
        selected_strict = overview_strict[(overview_strict["model"].isin(models))].set_index(['lang', 'model'])

        comparison = selected_strict.compare(selected_liberal, keep_shape=True, keep_equal=True, result_names=("strict", "liberal"))
        save_table(comparison, result_path, 'results_multiling_strict_liberal')
