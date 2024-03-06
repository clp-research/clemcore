import random
import requests

import zipfile

from clemgame import file_utils


class InstanceUtils:
    def __init__(self, experiment_config, game_name):
        self.experiment_config = experiment_config
        self.game_name = game_name
        self.common_config = file_utils.load_json(
            "../wordle/resources/common_config.json", self.game_name
        )

    def read_inital_prompt(self, use_clue, use_critic):
        file_name = self.common_config["system_definition_file_name"]
        system_definition = file_utils.load_file(
            f"../wordle/resources/{file_name}", self.game_name
        )

        guesser_prompt = []
        guesser_critic_prompt = []

        if use_critic:
            file_name = self.common_config["guess_prompt_with_critic_file_name"]
            guesser_prompt = file_utils.load_json(
                f"../wordle/resources/{file_name}", self.game_name
            )

            file_name = self.common_config["critic_prompt_file_name"]
            guesser_critic_prompt = file_utils.load_json(
                f"../wordle/resources/{file_name}", self.game_name
            )

        else:
            if use_clue:
                file_name = self.common_config["guess_prompt_with_clue_file_name"]
                guesser_prompt = file_utils.load_json(
                    f"../wordle/resources/{file_name}", self.game_name
                )
            else:
                file_name = self.common_config["guess_prompt_file_name"]
                guesser_prompt = file_utils.load_json(
                    f"../wordle/resources/{file_name}", self.game_name
                )

        return system_definition, guesser_prompt, guesser_critic_prompt
    
    def download_nytcrosswords(self):
        #Requires kaggle authentication for successfully downloading the file
        import os
        os.environ['KAGGLE_USERNAME'] = "<your-kaggle-user-name>"
        os.environ['KAGGLE_KEY'] = "<kaggle-api-key>"

        if os.environ['KAGGLE_USERNAME'] == "<your-kaggle-user-name>" or os.environ['KAGGLE_KEY'] == "<kaggle-api-key>":
            print("Please provide your kaggle credentials in the instance_utils.py file\n")
            return

        print("Downloading nytcrosswords...")
        fp = file_utils.file_path("resources/", "wordle")
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("darinhawley/new-york-times-crossword-clues-answers-19932021", path=fp)

        #Unzip the file 
        with zipfile.ZipFile(fp+"/new-york-times-crossword-clues-answers-19932021.zip","r") as zip_ref:
            zip_ref.extractall(fp)
        print("Stored the nytc crosswords clues file", fp)

    def download_allowed_words(self):
        print("Downloading wordle recognized words...")
        url = self.common_config["official_recognized_english_words_file_url"]
        r = requests.get(url, allow_redirects=True)
        fp = file_utils.file_path("resources/", "wordle")
        file_utils.store_file(r.content.decode("utf-8"), self.common_config["official_recognized_english_words_file"], fp)
        print("Stored the wordle recognized words file", fp)

    def read_file_contents(self, filename, file_ext="txt"):
        if file_ext == "csv":
            words_dict = {}        
            try:
                words_list = file_utils.load_csv(
                    f"../wordle/resources/{filename}", self.game_name
                )
            except FileNotFoundError:
                #File not available, downloading
                if filename == "nytcrosswords.csv":
                    self.download_nytcrosswords()
                    words_list = file_utils.load_csv(
                        f"../wordle/resources/{filename}", self.game_name
                    )

                else:
                    print(f"File {filename} not found")
                    return []

            if filename == "nytcrosswords.csv":
                for word in words_list:
                    words_dict[word[1].lower().strip()] = word[2].lower().strip()
                return words_dict
            else:
                return words_list

        elif file_ext == "txt":
            try:
                words = file_utils.load_file(
                    f"../wordle/resources/{filename}", self.game_name
                )
            except FileNotFoundError:
                if filename == "wordle_recognized_english_words.txt":
                    self.download_allowed_words()
                    words = file_utils.load_file(
                        f"../wordle/resources/{filename}", self.game_name
                    )             
                else:
                    print(f"File {filename} not found")
                    return []

            words = words.strip()
            if words:
                words_list = words.split("\n")
                words_list = [word.lower().strip() for word in words_list]
            else:
                words_list = []
            return words_list

    def categorize_target_words(self, unigram_freq_sorted_dict, clue_words_dict):
        easy_words = unigram_freq_sorted_dict[: int(len(unigram_freq_sorted_dict) / 3)]
        medium_words = unigram_freq_sorted_dict[
            int(len(unigram_freq_sorted_dict) / 3) : int(
                2 * len(unigram_freq_sorted_dict) / 3
            )
        ]
        hard_words = unigram_freq_sorted_dict[
            int(2 * len(unigram_freq_sorted_dict) / 3) :
        ]

        easy_words_list = [word[0] for word in easy_words]
        medium_words_list = [word[0] for word in medium_words]
        hard_words_list = [word[0] for word in hard_words]

        clue_words_keys = set(list(clue_words_dict.keys()))
        easy_words_list = set(easy_words_list).intersection(set(clue_words_keys))
        easy_words_list = list(easy_words_list)

        medium_words_list = set(medium_words_list).intersection(set(clue_words_keys))
        medium_words_list = list(medium_words_list)

        hard_words_list = set(hard_words_list).intersection(set(clue_words_keys))
        hard_words_list = list(hard_words_list)

        return easy_words_list, medium_words_list, hard_words_list

    def get_target_word_freq(self, word_list, freq_dict):
        data = {}
        for word in word_list:
            if word not in freq_dict:
                continue
            data[word] = freq_dict[word]
        return data

    def read_word_lists(self):
        english_words = []
        # officially recognized wordle words are downloaded from
        # https://github.com/3b1b/videos/blob/master/_2022/wordle/data/allowed_words.txt
        file_name = self.common_config["official_recognized_english_words_file"]
        english_words = self.read_file_contents(file_name)

        # wordle target words are downloaded from
        # https://github.com/3b1b/videos/blob/master/_2022/wordle/data/possible_words.txt
        # file_name = self.common_config["target_words_file_name"]
        # target_words = self.read_file_contents(file_name)

        # Uni-gram frequency data is downloaded from
        # https://www.kaggle.com/datasets/rtatman/english-word-frequency
        # file_name = self.common_config["unigram_freq_file_name"]
        # unigram_freq_dict = self.read_file_contents(file_name, file_ext="csv")

        # target_freq_dict = self.get_target_word_freq(target_words, unigram_freq_dict)
        # unigram_freq_sorted_dict = sorted(target_freq_dict.items(), key=lambda x: x[1])

        # Crosswords Clues are downloaded from
        # https://www.kaggle.com/datasets/darinhawley/new-york-times-crossword-clues-answers-19932021
        word_clues_dict = {}
        file_name = self.common_config["word_clues_file_name"]
        word_clues_dict = self.read_file_contents(file_name, file_ext="csv")

        # Currently the categorized words are read directly from the files
        #   without doing the categorization during instance generation
        # Check the file dump_categorized_words.py to see how the
        #   categorization is done
        # easy_words_list, medium_words_list, hard_words_list = self._categorize_target_words(unigram_freq_sorted_dict, word_clues_dict)
        easy_words_list = self.read_file_contents(
            self.common_config["easy_words_file_name"]
        )
        medium_words_list = self.read_file_contents(
            self.common_config["medium_words_file_name"]
        )
        hard_words_list = self.read_file_contents(
            self.common_config["hard_words_file_name"]
        )

        if not english_words or not word_clues_dict:
            print("Error in reading the word lists, check and download the relevant files")
            return "DATA_NOT_AVAILABLE"

        self.english_words = english_words
        # self.target_words = target_words
        self.word_clues_dict = word_clues_dict
        self.easy_words_list = easy_words_list
        self.medium_words_list = medium_words_list
        self.hard_words_list = hard_words_list

    def select_target_words(self):
        use_seed = self.common_config["seed_to_select_target_word"]
        number_of_target_words = self.common_config["number_of_target_words"]

        target_words_test_dict = {}

        data_to_read = self.read_word_lists()
        if data_to_read == "DATA_NOT_AVAILABLE":
            return target_words_test_dict

        if "high_frequency" in self.common_config["supported_word_difficulty"]:
            if self.easy_words_list:
                random.seed(use_seed)
                target_words_test_dict["high_frequency"] = random.choices(
                    self.easy_words_list, k=number_of_target_words["high_frequency"]
                )

        if "medium_frequency" in self.common_config["supported_word_difficulty"]:
            if self.medium_words_list:
                random.seed(use_seed)
                target_words_test_dict["medium_frequency"] = random.choices(
                    self.medium_words_list, k=number_of_target_words["medium_frequency"]
                )

        if "low_frequency" in self.common_config["supported_word_difficulty"]:
            if self.hard_words_list:
                random.seed(use_seed)
                target_words_test_dict["low_frequency"] = random.choices(
                    self.hard_words_list, k=number_of_target_words["low_frequency"]
                )

        return target_words_test_dict

    def update_experiment_dict(self, experiment):
        experiment["common_config"] = self.common_config
        experiment["english_words"] = self.english_words
        experiment["use_clue"] = self.experiment_config["use_clue"]
        experiment["use_critic"] = self.experiment_config["use_critic"]

        (
            system_definition,
            guesser_prompt,
            guesser_critic_prompt,
        ) = self.read_inital_prompt(experiment["use_clue"], experiment["use_critic"])
        experiment["system_definition"] = system_definition
        experiment["guesser_prompt"] = guesser_prompt
        experiment["guesser_critic_prompt"] = guesser_critic_prompt
        experiment["response_format_keywords"] = self.common_config["response_format_keywords"]

    def update_game_instance_dict(self, game_instance, word, difficulty):
        game_instance["target_word"] = word
        game_instance["target_word_clue"] = self.word_clues_dict[word]
        game_instance["target_word_difficulty"] = difficulty
