import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


models_langs = {
    "aya-23": ["ar", "cs", "de", "el", "en",
              "es", "fa", "fr", "he", "hi",
              "id", "it", "ja", "ko", "nl",
              "pl", "pt", "ro", "ru", "tr",
              "uk", "vi", "zh"],
    "llama-3": ["en"],
    "llama-3-de": ["de", "en"],
    "llama-3.1": ["de", "en", "es", "fr", "hi",
                  "it", "pt", "th"],
    "mixtral": ["de", "en", "es", "fr", "it"],
    "qwen": ["en"]
}


if __name__ == "__main__":
    languages = sorted(set([lang for langs in models_langs.values() for lang in langs]))
    df = pd.DataFrame(index=languages, columns=models_langs.keys())

    for model, langs in models_langs.items():
        for lang in langs:
            df[model][lang] = "+"

    path = "results"
    file = "models-lang-support"
    df.to_latex(f'{path}/{file}.tex', float_format="%.2f", na_rep="")
    df.to_html(f'{path}/{file}.html', na_rep="")

    # save as heatmap
    df = ~df.isnull().T
    ax = sns.heatmap(df, cmap="Blues", square=True)
    ax.figure.tight_layout()
    fig = ax.get_figure()
    fig.savefig(f"{path}/{file}.png")
    plt.close()
