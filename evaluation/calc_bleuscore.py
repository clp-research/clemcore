# tested with Linux OpenSUSE Leap 15.4
# Python 3.9.19

"""
Calculate the bleu score for all translations (machine vs. human)
"""
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import pandas as pd

from clemgame import file_utils
from evaluation.multiling_single_game_evaluation import save_table


if __name__ == "__main__":
      games = ["referencegame", "imagegame"]
      languages = ["de", "es", "ru", "te", "tk", "tr"]
      machine_suffix = "google"

      bleu_scores = {}
      for game in games:
            bleu_scores[(game, "Player A")] = []
            bleu_scores[(game, "Player B")] = []
            for lang in languages:
                  human_p1 = file_utils.load_template(f"resources/initial_prompts/{lang}/player_a_prompt_header.template", game)
                  human_p2 = file_utils.load_template(f"resources/initial_prompts/{lang}/player_b_prompt_header.template", game)
                  machine_p1 = file_utils.load_template(f"resources/initial_prompts/{lang}_{machine_suffix}/player_a_prompt_header.template", game)
                  machine_p2 = file_utils.load_template(f"resources/initial_prompts/{lang}_{machine_suffix}/player_b_prompt_header.template", game)

                  human_p1 = word_tokenize(human_p1)
                  human_p2 = word_tokenize(human_p2)
                  machine_p1 = word_tokenize(machine_p1)
                  machine_p2 = word_tokenize(machine_p2)

                  # normalerweise wird BLEU-4 berechnet: maximal 4-Gramme. Über weights kann es angepasst werden.
                  bleu_scores[(game, "Player A")].append(sentence_bleu([human_p1], machine_p1))  # , weights=(1./6., 1./6., 1./6., 1./6., 1./6., 1./6.)
                  bleu_scores[(game, "Player B")].append(sentence_bleu([human_p2], machine_p2))  # , weights=(1./6., 1./6., 1./6., 1./6., 1./6., 1./6.)
      df = pd.DataFrame(bleu_scores, index=languages).round(2)
      df.columns.rename("Prompt", level=1, inplace=True)
      df.index.name = "Language"

      save_table(df, path="results", file=f"bleu_score_machine_vs_human")
      # save_table(df["imagegame"], path="results", file=f"imagegame_bleu_score_machine_vs_human")
      # save_table(df["referencegame"], path="results", file=f"referencegame_bleu_score_machine_vs_human")
      # print(df)
