from typing import Dict, List

from backends import Backend
from clemgame.clemgame import GameBenchmark, GameMaster
from games.wordle.master import WordleGameMaster

# this will resolve into subdirectories to find the instances
GAME_NAME = "wordle_withclue"


class WordleWithClueGameBenchmark(GameBenchmark):
    def __init__(self):
        super().__init__(GAME_NAME)

    def get_description(self):
        return "Wordle Game with a clue given to the guesser"

    def create_game_master(
        self, experiment: Dict, player_backend: List[Backend]
    ) -> GameMaster:
        return WordleGameMaster(self.name, experiment, player_backend)

    def is_single_player(self) -> bool:
        return True
