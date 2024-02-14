"""
Generate instances for the taboo game.

Creates files in ./instances
"""
import random

from tqdm import tqdm

import requests

import nltk
from nltk.corpus import wordnet

import clemgame
from clemgame.clemgame import GameInstanceGenerator

nltk.download('wordnet', quiet=True)
EN_LEMMATIZER = nltk.stem.WordNetLemmatizer()

API_KEY = ""  # your key for the Merriam-Webster thesaurus
N_INSTANCES = 20  # how many different target words; zero means "all"
N_GUESSES = 3  # how many tries the guesser will have
N_REATED_WORDS = 3
LANGUAGE = "en"

logger = clemgame.get_logger(__name__)
GAME_NAME = "taboo"


def find_synonyms_remote(word, n, api_key):
    """ Choose n synonyms from all possible meanings """
    url = f"https://www.dictionaryapi.com/api/v3/references/thesaurus/json/{word}?key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            data = response.json()
            primate_sense = data[0]
            syns = primate_sense.get("meta", {}).get("syns", [])
            synonyms_flatten = set([syn for sense in syns for syn in sense])
            synonyms_flatten = list(synonyms_flatten)
            if len(synonyms_flatten) >= n:  # sub-sample
                selection = random.sample(synonyms_flatten, k=n)
            else:
                selection = synonyms_flatten
            return selection
        except Exception as e:
            print(e)
            return []
    else:
        print("Error:", response.status_code)
        return []


def find_synonyms(word, n):
    """ Choose n synonyms from all possible meanings """
    possible_synonyms_groups = wordnet.synonyms(word)
    synonyms_flatten = [synonym for synonym_group in possible_synonyms_groups for synonym in synonym_group]
    lemma = EN_LEMMATIZER.lemmatize(word)
    exclusive_synonyms = [synonym for synonym in synonyms_flatten
                          if lemma not in synonym and EN_LEMMATIZER.lemmatize(synonym) != lemma]
    selection = exclusive_synonyms
    if len(exclusive_synonyms) >= n:  # sub-sample
        selection = random.sample(exclusive_synonyms, k=n)
    return selection


class TabooGameInstanceGenerator(GameInstanceGenerator):

    def __init__(self):
        super().__init__(GAME_NAME)

    def on_generate(self):
        for difficulty in ["high", "medium", "low"]:

            # first choose target words based on the difficultly
            fp = f"resources/target_words/{LANGUAGE}/{difficulty}_freq_100"
            target_words = self.load_file(file_name=fp, file_ending=".txt").split('\n')
            if N_INSTANCES > 0:
                assert len(target_words) >= N_INSTANCES, \
                    f'Fewer words available ({len(target_words)}) than requested ({N_INSTANCES}).'
                target_words = random.sample(target_words, k=N_INSTANCES)

            # use the same target_words for the different player assignments
            experiment = self.add_experiment(f"{difficulty}_{LANGUAGE}")
            experiment["max_turns"] = N_GUESSES

            describer_prompt = self.load_template("resources/initial_prompts/initial_describer")
            guesser_prompt = self.load_template("resources/initial_prompts/initial_guesser")
            experiment["describer_initial_prompt"] = describer_prompt
            experiment["guesser_initial_prompt"] = guesser_prompt

            for game_id in tqdm(range(len(target_words))):
                target = target_words[game_id]

                game_instance = self.add_game_instance(experiment, game_id)
                game_instance["target_word"] = target
                game_instance["related_word"] = find_synonyms_remote(target, n=N_REATED_WORDS, api_key=API_KEY)

                if len(game_instance["related_word"]) < N_REATED_WORDS:
                    print(f"Found less than {N_REATED_WORDS} related words for: {target}")


if __name__ == '__main__':
    TabooGameInstanceGenerator().generate()
