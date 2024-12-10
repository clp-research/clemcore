"""
Generate instances for the taboo game.

Creates files in ./instances
"""
import random

from tqdm import tqdm

import clemgame
from clemgame.clemgame import GameInstanceGenerator

from games.wordle.utils.instance_utils import InstanceUtils

LANGUAGE = "en"

logger = clemgame.get_logger(__name__)
GAME_NAME = "wordle"
SEED = "17"

class WordleGameInstanceGenerator(GameInstanceGenerator):

    def __init__(self, game_name):
        super().__init__(game_name)
        self.game_name = game_name

    def load_instances(self):
        return self.load_json("in/instances")
    
    def _setresponseformatkeywords(self, language):
        """
            To support a new language, update the file resources/langconfig.json
            Do not change the key names
            Only update their values with the text in the new language
        """
        # The relative path is used to force looking for the file in the wordle directory
        lang_keywords = self.load_json("../wordle/resources/langconfig.json")
        self.lang_keywords = lang_keywords[language]    

    def on_generate(self):
        self.experiment_config = self.load_json("resources/config.json")
        self._setresponseformatkeywords(LANGUAGE)

        self.instance_utils = InstanceUtils(self.experiment_config, self.game_name, LANGUAGE)

        target_words_test_dict = self.instance_utils.select_target_words(SEED)

        word_difficulty = list(target_words_test_dict.keys())

        for difficulty in word_difficulty:
            experiment_name = f'{difficulty}_words_{self.experiment_config["name"]}'
            experiment = self.add_experiment(experiment_name)
            self.instance_utils.update_experiment_dict(experiment, self.lang_keywords)

            for index, word in enumerate(target_words_test_dict[difficulty]):
                game_instance = self.add_game_instance(experiment, index + 1)
                self.instance_utils.update_game_instance_dict(
                    game_instance, word, difficulty
                )

if __name__ == "__main__":
    WordleGameInstanceGenerator(GAME_NAME).generate()                