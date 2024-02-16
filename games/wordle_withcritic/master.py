from typing import Dict, List

from backends import Model
from clemgame.clemgame import GameBenchmark, GameMaster
from games.wordle.master import WordleGameMaster

GAME_NAME = "wordle_withcritic"


class WordleWithClueAndCriticGameBenchmark(GameBenchmark):
    def __init__(self):
        super().__init__(GAME_NAME)

    def get_description(self):
        return "Wordle Game with a clue given to the guesser and a critic for the clue"

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return WordleGameMaster(self.name, experiment, player_models)
