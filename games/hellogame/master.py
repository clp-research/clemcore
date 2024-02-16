import string
from typing import Dict, List

from backends import Model, CustomResponseModel
from clemgame.clemgame import GameMaster, GameBenchmark, Player, DialogueGameMaster
from clemgame import get_logger

logger = get_logger(__name__)

GAME_NAME = "hellogame"


class Greeted(Player):

    def __init__(self, name):
        super().__init__(CustomResponseModel())
        self.name = name

    def _custom_response(self, messages, turn_idx):
        return f"{self.name}: Hi, thanks for having me!"


class Greeter(Player):

    def __init__(self, model: Model):
        super().__init__(model)

    def _custom_response(self, messages, turn_idx):
        raise NotImplementedError("This should not be called, but the remote APIs.")


class HelloGame(DialogueGameMaster):
    """This class implements a greeting game in which player A
    is greeting another player with a target name.
    """

    def __init__(self, experiment: Dict, player_models: List[Model]):
        super().__init__(GAME_NAME, experiment, player_models)
        self.language: int = experiment["language"]  # fetch experiment parameters here
        self.turns = []
        self.required_words = ["welcome", "hello"]
        self.success = True

    def _on_setup(self, **game_instance):
        self.game_instance = game_instance  # fetch game parameters here

        # Create the players
        self.greeted = Greeted(game_instance["target_name"])
        self.greeter = Greeter(self.player_models[0])

        # Add the players: these will be logged to the records interactions.json
        # Note: During game play the players will be called in the order added here
        self.add_player(self.greeter)
        self.add_player(self.greeted)

        self.required_words.append(self.greeted.name.lower())

    def _on_before_game(self):
        # Do something before the game start e.g. add the initial prompts to the message list for the players
        self.add_user_message(self.greeter, self.game_instance["prompt"])

    def _does_game_proceed(self):
        # Determine if the game should proceed. This is also called once initially.
        if len(self.turns) == 0:
            return True
        return False

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        # Check responses for specific players
        if player == self.greeter:
            # Check rule: utterance starts with key word
            if not utterance.startswith("GREET:"):
                self.success = False
                return True
            # Check rule: required words are included
            utterance = utterance.lower()
            utterance = utterance.translate(str.maketrans("", "", string.punctuation))
            for required_word in self.required_words:
                if required_word not in utterance:
                    self.success = False
        return True

    def _on_after_turn(self, turn_idx: int):
        self.turns.append(self.success)

    def _after_add_player_response(self, player: Player, utterance: str):
        if player == self.greeter:
            self.add_user_message(self.greeted, utterance)

    def compute_scores(self) -> None:
        score = 0
        if self.success:
            score = 1
        self.log_episode_score('Accuracy', score)


class HelloGameBenchmark(GameBenchmark):

    def __init__(self):
        super().__init__(GAME_NAME)

    def get_description(self):
        return "Hello game between a greeter and a greeted player"

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return HelloGame(experiment, player_models)
