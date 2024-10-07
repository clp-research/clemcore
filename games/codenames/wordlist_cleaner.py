import os, argparse, json
from pathlib import Path


GAME_PATH = "games/codenames/"

def clean_wordlist(wordlist_name, source, dest):
    print(f"Cleaning {wordlist_name}...")
    wordlist = load_wordlist(f"{GAME_PATH}{source}/{wordlist_name}")
    # check whether wordlist contains more than one hierarchical level (category names, frequency thresholds, etc.)
    if type(wordlist["words"]) == list:
        length_before = len(wordlist["words"])
        wordlist["words"] = clean(wordlist["words"])
        length_after = len(wordlist["words"])
    else:
        length_before = 0
        length_after = 0
        for attribute in wordlist["words"].keys():
            length_before += len(wordlist["words"][attribute])
            wordlist["words"][attribute] = clean(wordlist["words"][attribute])
            length_after += len(wordlist["words"][attribute])
    print(f"Removed {length_before - length_after} words, contains {length_after} words now.")
    save_wordlist(wordlist, f"{GAME_PATH}{dest}/{wordlist_name}")    

def clean(list_of_words):
    # all words in lower()
    list_of_words = [word.lower() for word in list_of_words]
    # no words with spaces or punctuation
    removed_words = [word for word in list_of_words if not word.isalpha()]
    if removed_words != []:
        print(f"Removing words {', '.join(removed_words)}") 
    list_of_words = [word for word in list_of_words if word.isalpha()]

    return list_of_words

def load_wordlist(filepath):
    with open(filepath) as file:
        wordlist = json.load(file)
    return wordlist

def save_wordlist(wordlist, filepath):
    Path(os.path.dirname(filepath)).mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as file:
        json.dump(wordlist, file)

def clean_all_wordlists(source="resources/wordlists", dest="resources/cleaned_wordlists"):
    wordlist_names = os.listdir(f"{GAME_PATH}{source}")
    print(wordlist_names)
    for wordlist_name in wordlist_names:
        clean_wordlist(wordlist_name, source, dest)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wordlist", help="Optional argument to only clean a specific wordlist.")
    parser.add_argument("-s", "--source", help="Optional argument to specify a source directory for the wordlist(s).")
    parser.add_argument("-d", "--dest", help="Optional argument to specify a destination directory for the cleaned wordlist(s).")
    args = parser.parse_args()
    arg_dict = vars(args)
    not_none_params = {k:v for k, v in arg_dict.items() if v is not None}
    if args.wordlist:
        clean_wordlist(args.wordlist, **not_none_params)
    else:
        clean_all_wordlists(**not_none_params)