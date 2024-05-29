import argparse
import json
import os
import sys

sys.path.append('..')
import glob
from pathlib import Path
import pandas as pd
from clemgame import metrics


def create_overview_table(df: pd.DataFrame, categories: list) -> pd.DataFrame:
    """
    Create multilingual results dataframe.
    :param df: the initial dataframe with all episode scores
    :param categories: list of columns over which to aggregate scores
    :return: the aggregated dataframe
    """

    relevant_metrics = [metrics.METRIC_PLAYED, metrics.BENCH_SCORE, "Aborted at Player 1"]
    # BENCH_SCORE for referencegame = success * 100
    scored_df = df[(df.game == 'referencegame') & (df["metric"].isin(relevant_metrics))]

    # refactor model names for readability
    scored_df = scored_df.replace(to_replace=r'(.+)-t0.0--.+', value=r'\1', regex=True)

    # compute mean metrics
    df_means = (scored_df.groupby(categories)
                .mean(numeric_only=True)
                .reset_index())

    # convert to percentages
    aux_ab_p1 = df_means.loc[df_means.metric == "Aborted at Player 1", 'value']
    aux_played = df_means.loc[df_means.metric == metrics.METRIC_PLAYED, 'value']
    aux_aborted = (1-aux_played).to_list()
    df_means.loc[df_means.metric == "Aborted at Player 1", 'value'] = (aux_ab_p1/aux_aborted) * 100

    df_means.loc[df_means.metric == metrics.METRIC_PLAYED, 'value'] *= 100
    # BENCH_SCORE is already success * 100

    df_means = df_means.round(2)

    # rename columns
    df_means['metric'].replace(
        {metrics.METRIC_PLAYED: '% Played', metrics.BENCH_SCORE: '% Success (of Played)', "Aborted at Player 1": 'Aborted at Player 1 (of Aborted)'},
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


def save_overview_tables_by_scores(df, categories, path, prefix):
     df_played = df[categories + ['% Played']]
     df_played = df_played.pivot(columns='lang', index="model")
     save_table(df_played, path, f"{prefix}_by_played")

     df_success = df[categories + ['% Success (of Played)']]
     df_success = df_success.pivot(columns='lang', index="model")
     save_table(df_success, path, f"{prefix}_by_success")

     df_clemscore = df[categories + ['clemscore (Played * Success)']]
     df_clemscore = df_clemscore.pivot(columns='lang', index="model")
     save_table(df_clemscore, path, f"{prefix}_by_clemscore")


def save_table(df, path: str, file: str):
    # save table
    df.to_csv(Path(path) / f'{file}.csv')
    # for adapting for a paper
    df.to_latex(Path(path) / f'{file}.tex', float_format="%.2f") # index = False
    # for easy checking in a browser
    df.to_html(Path(path) / f'{file}.html')
    print(f'\n Saved results into {path}/{file}.csv, .html and .tex')


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-p", "--results_path", type=str, default="../results/v1.5_multiling",
                            help="A relative or absolute path to the results root directory. Default: ../results/v1.5_multiling")
    arg_parser.add_argument("-d", "--detailed", type=bool, default=False,
                            help="Whether to create a detailed overview table by experiment. Default: False")
    arg_parser.add_argument("-c", "--compare", type=str, default="",
                            help="An optional relative or absolute path to another results root directory to which the results should be compared.")
    parser = arg_parser.parse_args()

    output_prefix = parser.results_path.split("/")[-1]

    # collect all language specific results in one dataframe
    df_lang = None
    if parser.compare:
        df_compare = None
    result_dir = Path(parser.results_path)
    lang_dirs = glob.glob(f"{result_dir}/*/") # the trailing / ensures that only directories are found
    for lang_dir in lang_dirs:
        lang = lang_dir.split("/")[-2]
        assert len(lang) == 2
        raw_file = os.path.join(lang_dir, 'raw.csv')
        assert Path(raw_file).is_file()
        lang_result = pd.read_csv(raw_file, index_col=0)
        lang_result.insert(0, 'lang', lang)
        df_lang = pd.concat([df_lang, lang_result], ignore_index=True)

        if parser.compare:
            raw_file = raw_file.replace(parser.results_path.split("/")[-1], parser.compare.split("/")[-1])
            assert Path(raw_file).is_file()
            lang_result = pd.read_csv(raw_file, index_col=0)
            lang_result.insert(0, 'lang', lang)
            df_compare = pd.concat([df_compare, lang_result], ignore_index=True)

    if parser.detailed:
        categories = ['lang', 'model', 'experiment', 'metric'] # detailed by experiment
        overview_detailed = create_overview_table(df_lang, categories)
        save_overview_tables_by_scores(overview_detailed, categories[:-1], parser.results_path, f'{output_prefix}_by_experiment')

    else:
        categories = ['lang', 'model', 'metric']
        overview_strict = create_overview_table(df_lang, categories)
        save_overview_tables_by_scores(overview_strict, categories[:-1], parser.results_path, output_prefix)

        # sort models within language by clemscore
        sorted_df = overview_strict.sort_values(['lang','clemscore (Played * Success)'],ascending=[True,False])
        # extract model order by language for rank correlation analysis
        model_orders = {}
        languages = sorted_df['lang'].unique()
        for lang in languages:
            models = sorted_df.loc[sorted_df.lang == lang, 'model']
            scores = sorted_df.loc[sorted_df.lang == lang, 'clemscore (Played * Success)']
            models_and_scores = list(zip(models.tolist(), scores.tolist()))
            model_orders[lang] = models_and_scores
        with open(f'{parser.results_path}/model_rankings_by_language.json', 'w', encoding='utf-8') as f:
            json.dump(model_orders, f, ensure_ascii=False)
        save_table(sorted_df.set_index(['lang', 'model']), parser.results_path, output_prefix)

    if parser.compare:
        overview_liberal = create_overview_table(df_compare, categories)
        # TODO: adapt comparison to new table format (model x score/lang)
        # get intersection of models
        #models = ["fsc-openchat-3.5-0106"] # "command-r-plus", "Llama-3-8b-chat-hf",
        #          "Llama-3-70b-chat-hf"]
        # compare % Played between strict and liberal
        #selected_strict = overview_strict.loc[categories + ['% Played']].pivot(columns='lang', index="model")
        #selected_liberal = overview_liberal.loc[categories + ['% Played']].pivot(columns='lang', index="model")
        #comparison = selected_strict.compare(selected_liberal, keep_shape=True, keep_equal=True, result_names=("strict", "liberal"))
        # compute delta and replace on df
        #delta = comparison['% Played']['liberal'] - comparison['% Played']['strict']
        #delta = delta.round(2).to_frame(name=('improvement of % Played in liberal mode'))
        #save_table(delta, result_path, 'results_delta_strict_liberal')
