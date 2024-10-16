# tested on Linux OpenSUSE Leap 15.4
# Python 3.9.19

import argparse
import json
import os
import sys

sys.path.append('..')
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import seaborn as sns

from clemgame import metrics, get_logger
from rank_correlation import calc_kendalltau, wikipedia_articles, gpt4_report_ranking
from multiling_eval_utils import short_names, gpt4_report_str, wiki_articles_str, mean_models_str, mean_langs_str

logger = get_logger(__name__)


def create_overview_table(df: pd.DataFrame, game: str, categories: list) -> pd.DataFrame:
    """
    Create multilingual results dataframe.
    :param df: the initial dataframe with all episode scores
    :param categories: list of columns over which to aggregate scores
    :return: the aggregated dataframe
    """

    relevant_metrics = [metrics.METRIC_PLAYED, metrics.BENCH_SCORE, "Aborted at Player 1"]
    # BENCH_SCORE for specified game = success * 100
    scored_df = df[(df.game == game) & (df["metric"].isin(relevant_metrics))]

    # refactor model names for readability
    scored_df = scored_df.replace(to_replace=r'(.+)-t0.0--.+', value=r'\1', regex=True)
    scored_df = scored_df.replace(short_names)

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
    clemscore = clemscore.fillna(0)  # set clemscore to 0 if no game is played
    clemscore = clemscore.round(2).to_frame(name=('clemscore (Played * Success)'))
    df_results = pd.concat([clemscore, df_means], axis=1)
    df_results.reset_index(inplace=True)

    return df_results

def create_overview_tables_by_scores(df, categories):
    pivot_cols = ['lang', 'experiment'] if 'experiment' in categories else 'lang'

    metrics = ['% Played', '% Success (of Played)', 'clemscore (Played * Success)']

    dfs_out = []
    for metric in metrics:
        df_score = df[categories + [metric]]
        df_score = df_score.pivot(columns=pivot_cols, index="model")
        df_score.loc[mean_models_str] = df_score.mean().round(2)  # row for mean of models
        if 'experiment' not in categories:
            df_score[metric, mean_langs_str] = df_score.mean(axis=1).round(2)  # column for mean over languages
        dfs_out.append(df_score)

    return dfs_out


def save_overview_tables_by_scores(df, categories, path, prefix):
    df_played, df_success, df_clemscore = create_overview_tables_by_scores(df, categories)
    save_table(df_played, path, f"{prefix}_by_played")
    save_table(df_success, path, f"{prefix}_by_success")
    save_table(df_clemscore, path, f"{prefix}_by_clemscore")


def save_table(df, path: str, file: str):
    # save table
    df.to_csv(Path(path) / f'{file}.csv')
    # for adapting for a paper
    df.to_latex(Path(path) / f'{file}.tex', float_format="%.2f") # index = False
    # for easy checking in a browser
    df.to_html(Path(path) / f'{file}.html')
    logger.info(f'\n Saved results into {path}/{file}.csv, .html and .tex')


def prepare_compare_models_dfs(*dfs):
    """Takes dfs computed by create_overview_tables_by_scores."""
    dfs_out = []
    for df in dfs:
        df.columns = df.columns.droplevel(0)
        df = df.transpose()
        df = df.drop(mean_langs_str)
        dfs_out.append(df)
    return dfs_out


def save_model_score_plot(df_score, path: str, file: str):
    """
    Creates and saves a plot showing the performance of each model (yachsis) for each language (xachsis).
    """
    ax = df_score.plot(style=".-")
    ax.set_xticks(range(len(df_score)))
    ax.set_xticklabels(df_score.index.str.replace("_google", "_"))
    ax.legend(title=None)
    plt.savefig(Path(path) / f'{file}.png')
    plt.close()
    logger.info(f'\n Saved plot into {path}/{file}.png')


def create_rank_correlation_df(df, *compare_rankings: pd.Series):
    """
    Build df showing correlation (kendall's tau) between each pair of columns in overgiven df.
    A ranking is created from the values in each column.
    If a language is missing in a column, the correlation with this column is calculated without the missing language.

    :param df: Each column contains scores. Language identifiers are in the index.
    :compare_rankings: Additional Series objects to be compared to each column in df.
    """
    df = pd.concat([df, *compare_rankings], axis=1)

    col_pairs = combinations(df, 2)  # combinations of column names

    # new df for correlation (kendalls tau) between two models
    df_corr = pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)
    for colname in df.columns:
        df_corr[colname][colname] = calc_kendalltau(df[colname], df[colname])[0]  # kendalltau is nan when all languages in a rank are in a tie
    for col1, col2 in col_pairs:
        tau, _ = calc_kendalltau(df[col1], df[col2])
        df_corr[col1][col2] = tau
        df_corr[col2][col1] = tau
    return df_corr.round(3)


def save_as_heatmap(df, path: str, file: str):
    ax = sns.heatmap(df, vmin=-1.0, vmax=1.0, annot=True, cmap="coolwarm")
    ax.figure.tight_layout()
    plt.xticks(rotation=20)
    fig = ax.get_figure()
    fig.savefig(Path(path) / f'{file}.png')
    plt.close()
    logger.info(f'\n Saved plot into {path}/{file}.png')


def prepare_external_language_rank(rank: dict, name: str, languages: list[str]):
    """
    :param rank: Values are language identifiers (iso-639-1).
    :param name: Will be the name of the resulting pd.Series.
    :param languages: The languages that have been evaluated.
    """
    # for each entry with key 'lang' insert an entry 'lang_suffix'
    rank_suf = {f"{key}_{machine_suffix}": value for key, value in rank.items()}
    rank.update(rank_suf)
    # remove language entries that are not languages
    rank = {key: value for key, value in rank.items() if key in languages}
    return pd.Series(rank, name=name)


def assert_log(condition: bool, message: str = ""):
    """Raise AssertionError if condition is false and log error."""
    try:
        assert condition, message
    except AssertionError as e:
        logger.warning(e)
        sys.exit()


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-g", "--game", type=str, help="The game that should be evaluated.")
    arg_parser.add_argument("-p", "--results_path", type=str, default="../results/v1.5_multiling",
                            help="A relative or absolute path to the results root directory. Default: ../results/v1.5_multiling")
    arg_parser.add_argument("-d", "--detailed", action="store_true",
                            help="Whether to create a detailed overview table by experiment. Default: False")
    arg_parser.add_argument("-c", "--compare", type=str, default="",
                            help="An optional relative or absolute path to another results root directory to which the results should be compared.")
    arg_parser.add_argument("-t", "--translation_type", type=str, default="human+google",
                            help="Specifies which translations are evaluated (human/machine). "
                                 "The string representing the machine translations should match the suffix chosen for the corresponding folders. "
                                 "E.g. '-t google' if you only want to evaluate folders ending with '_google'. "
                                 "Write '-t human' to only include folders without suffix. "
                                 "Use '+' to separate both types. Default: human+google.")
    arg_parser.add_argument("-cm", "--compare_models", nargs="+",
                            help="Compare the language rankings of the models. Default: [Llama-3-70B-Instruct, Mixtral-8x22B-Instruct-v0.1]",
                            default=["aya-23-35B",
                                     "Llama-3-70B-Instruct",
                                     "Llama-3-SauerkrautLM-70b-Instruct",
                                     "Meta-Llama-3.1-70B-Instruct",
                                     "Mixtral-8x22B-Instruct-v0.1",
                                     "Qwen1.5-72B-Chat"])
    arg_parser.add_argument("-s", "--save_merged_csv", action="store_true", help="Save a big csv file 'raw.csv' to the results_path that combines all raw.csv's from the lang directories. Is filtered as specified by option '-t'.")
    arg_parser.add_argument("-sq", "--save_merged_csv_and_quit", action="store_true", help="Save a big csv file as with option '-s' and quit directly afterwards.")

    parser = arg_parser.parse_args()

    output_prefix = parser.results_path.rstrip("/").split("/")[-1]

    human = "human" in parser.translation_type
    machine_suffix = parser.translation_type.replace("human", "").replace("+", "", 1)
    # check if not more than one translation suffix is specified
    assert_log("+" not in machine_suffix, "Not more than one suffix can be specified in --translation_type")

    # use model short names
    compare_models = [short_names[model] for model in parser.compare_models if model in short_names]

    # create subdirectories for evaluation output files
    assert_log(Path(parser.results_path).is_dir())
    out_dir = os.path.join(parser.results_path, f"multiling_eval/{parser.game}")
    out_dir = os.path.join(out_dir, parser.translation_type, ("detailed" if parser.detailed else ""))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # collect all language specific results in one dataframe
    df_lang = None
    if parser.compare:
        df_compare = None
    result_dir = Path(parser.results_path)
    lang_dirs = glob.glob(f"{result_dir}/*/") # the trailing / ensures that only directories are found
    for lang_dir in lang_dirs:
        lang = lang_dir.split("/")[-2]

        # skip non-language directories
        if not (
            (len(lang) == 2) or (len(lang.split('_')[0]) == 2)
            ):  # machine translations have identifiers such as 'de_google'
            continue

        # skip languages that are not specified in parser.translation_type
        if len(lang.split("_")) > 1:  # if lang has a suffix
            if lang.split("_")[1] != machine_suffix:  # and that suffix is not specified in translation_type
                continue
        elif not human:  # if lang has no suffix and 'human' is not specified in translation_type
            continue

        # check if game folders exist for all models
        model_dirs = glob.glob(f"{lang_dir}/*/")
        for model_dir in model_dirs:
            assert_log(os.path.exists(os.path.join(model_dir, parser.game)), f"Missing directory '{parser.game}/' in '{model_dir}'")

        raw_file = os.path.join(lang_dir, 'raw.csv')
        assert_log(Path(raw_file).is_file())
        lang_result = pd.read_csv(raw_file, index_col=0)
        lang_result.insert(0, 'lang', lang)
        df_lang = pd.concat([df_lang, lang_result], ignore_index=True)

        if parser.compare:
            raw_file = raw_file.replace(output_prefix, parser.compare.rstrip("/").split("/")[-1])
            assert_log(Path(raw_file).is_file())
            lang_result = pd.read_csv(raw_file, index_col=0)
            lang_result.insert(0, 'lang', lang)
            df_compare = pd.concat([df_compare, lang_result], ignore_index=True)

    assert_log(not df_lang.empty, f"no results found for {parser.game} {parser.translation_type}")

    # save combined raw results together in one file
    if parser.save_merged_csv or parser.save_merged_csv_and_quit:
        df_lang.to_csv(result_dir / f"{parser.translation_type}_raw.csv")
        print(f"Saved '{parser.translation_type}_raw.csv' to {result_dir}")
        if parser.save_merged_csv_and_quit:
            sys.exit()

    if parser.detailed:
        categories = ['lang', 'model', 'experiment', 'metric'] # detailed by experiment
        overview_detailed = create_overview_table(df_lang, parser.game, categories)
        save_overview_tables_by_scores(overview_detailed, categories[:-1], out_dir, f'{output_prefix}_{parser.game}_by_experiment')

    else:
        categories = ['lang', 'model', 'metric']
        overview_strict = create_overview_table(df_lang, parser.game, categories)
        save_overview_tables_by_scores(overview_strict, categories[:-1], out_dir, f'{output_prefix}_{parser.game}')

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
        with open(f'{out_dir}/model_rankings_by_language_{parser.game}.json', 'w', encoding='utf-8') as f:
            json.dump(model_orders, f, ensure_ascii=False)
        save_table(sorted_df.set_index(['lang', 'model']), out_dir, f'{output_prefix}_{parser.game}')


        # --Compare models--

        # check if all models played in all languages
        for model in compare_models:
            assert_log(model in sorted_df["model"].unique(), f"{model} has not been run")
            assert_log(sorted_df["model"].value_counts()[model] == len(languages), f"{model} has not benn run in all languages")

        # dfs with models as index and lang as columns
        df_played, df_success, df_clemscore = create_overview_tables_by_scores(overview_strict, categories[:-1])
        # prepare dfs to create plot
        df_played, df_success, df_clemscore = prepare_compare_models_dfs(df_played, df_success, df_clemscore)

        # visualise scores of models in the different languages
        save_model_score_plot(df_clemscore, out_dir, f'{output_prefix}_{parser.game}_models_clemscore')
        save_model_score_plot(df_played, out_dir, f'{output_prefix}_{parser.game}_models_played')
        save_model_score_plot(df_success, out_dir, f'{output_prefix}_{parser.game}_models_success')

        if human and not machine_suffix:
            # model rankings are compared to two external rankings (wikipedia_articles, gpt4_report_ranking)
            wikipedia_articles = prepare_external_language_rank(wikipedia_articles, name=wiki_articles_str, languages=languages)
            gpt4_report_ranking = prepare_external_language_rank(gpt4_report_ranking, name=gpt4_report_str, languages=languages)

            # df with language ranking correlation for each pair of models
            df_clemscore_corr = create_rank_correlation_df(df_clemscore, wikipedia_articles, gpt4_report_ranking)
            df_played_corr = create_rank_correlation_df(df_played, wikipedia_articles, gpt4_report_ranking)
            df_success_corr = create_rank_correlation_df(df_success, wikipedia_articles, gpt4_report_ranking)
        else:  # only compare human ranking with external rankings
            # df with language ranking correlation for each pair of models
            df_clemscore_corr = create_rank_correlation_df(df_clemscore)
            df_played_corr = create_rank_correlation_df(df_played)
            df_success_corr = create_rank_correlation_df(df_success)

        save_as_heatmap(df_clemscore_corr, out_dir, f'{output_prefix}_{parser.game}_correlation_clemscore')
        save_as_heatmap(df_played_corr, out_dir, f'{output_prefix}_{parser.game}_correlation_played')
        save_as_heatmap(df_success_corr, out_dir, f'{output_prefix}_{parser.game}_correlation_success')

        save_table(df_clemscore_corr, out_dir, f'{output_prefix}_{parser.game}_correlation_clemscore')
        save_table(df_played_corr, out_dir, f'{output_prefix}_{parser.game}_correlation_played')
        save_table(df_success_corr, out_dir, f'{output_prefix}_{parser.game}_correlation_success')


        if human and machine_suffix:
            # create table to compare human vs. machine (mean models)
            df_temp = pd.concat([df_clemscore[mean_models_str].rename(f"{metrics.BENCH_SCORE} (Ø models)"),
                            df_played[mean_models_str].rename(f"{metrics.METRIC_PLAYED} (Ø models)"),
                            df_success[mean_models_str].rename(f"{metrics.METRIC_SUCCESS} (Ø models)")],
                            axis=1)

            # create new column that specifies the translation type (human/machine)
            translator_series = pd.Series(df_temp.index.str.len() == 2).replace({True: "human", False: machine_suffix})
            translator_series.index = df_temp.index
            df_temp["translator"] = translator_series
            df_temp.index = df_temp.index.str.replace(f"_{machine_suffix}", "")

            df_human_machine = df_temp.pivot(columns=["translator"])

            save_table(df_human_machine, out_dir, f'{output_prefix}_{parser.game}_human_vs_machine')

            # create bar plot
            df_human_machine.columns.rename("metric", level=0, inplace=True)
            ax = df_human_machine.plot(kind="bar", color=["orange", "blue"])
            plt.savefig(Path(out_dir) / f'{output_prefix}_{parser.game}_human_vs_machine.png')
            plt.close()
            logger.info(f'\n Saved plot into {out_dir}/{output_prefix}_{parser.game}_human_vs_machine.png')



    if parser.compare:
        overview_liberal = create_overview_table(df_compare, parser.game, categories)
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
