import random
import requests

import nltk
from nltk.corpus import wordnet

nltk.download('wordnet', quiet=True)
EN_LEMMATIZER = nltk.stem.WordNetLemmatizer()

API_KEY = ""  # your key for the Merriam-Webster thesaurus


def find_synonyms_remote(word, n):
    """ Choose n synonyms from all possible meanings """
    url = f"https://www.dictionaryapi.com/api/v3/references/thesaurus/json/{word}?key={API_KEY}"
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
