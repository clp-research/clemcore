from typing import Dict, Tuple, List, Union

import numpy as np

from backends import Model, CustomResponseModel
from clemgame.clemgame import GameMaster, GameBenchmark, Player, DialogueGameMaster, GameScorer
from clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, METRIC_REQUEST_COUNT, \
    METRIC_REQUEST_COUNT_VIOLATED, METRIC_REQUEST_COUNT_PARSED, METRIC_REQUEST_SUCCESS, BENCH_SCORE
from clemgame import get_logger
from clemgame import file_utils, string_utils

GAME_NAME = "askmissing"

logger = get_logger(__name__)


class Speaker(Player):

    def __init__(self, info, skipped):
        super().__init__(CustomResponseModel())
        self.info = info
        self.skipped = skipped

    def _custom_response(self, messages, turn_idx):
        return self.skipped

class Rememberer(Player):

    def __init__(self, model: Model):
        super().__init__(model)

    def _custom_response(self, messages, turn_idx):
        raise NotImplementedError("This should not be called, but the remote APIs.")

class AskMissing(DialogueGameMaster):
    """This class implements a greeting game in which player A
    is greeting another player with a target name.
    """

    def __init__(self, experiment: Dict, player_models: List[Model]):
        super().__init__(GAME_NAME, experiment, player_models)
        # self.max_turns: int = experiment["max_turns"]
        self.initial_prompt = experiment["initial_prompt"]
        self.language: int = experiment["language"]  # fetch experiment parameters here
        self.turns = []
        # self.success = True


    def _on_setup(self, **game_instance):
        self.game_instance = game_instance  # fetch game parameters here
        self.game_id = self.game_instance['game_id']
        # Create the players
        # print('question', self.question)
        self.speaker = Speaker(self.game_instance['info'], self.game_instance['skip'])
        self.rememberer = Rememberer(self.player_models[0])
        #if self.game_id == 0:
        self.current_prompt = self.initial_prompt + self.game_instance['info'] 
        #else:
        #self.current_prompt = self.game_instance['info'] 

        # print(self.initial_prompt)
        # print('\ncurrent prompt:', self.current_prompt)
        # Add the players: these will be logged to the records interactions.json
        # Note: During game play the players will be called in the order added here
        self.add_player(self.rememberer)
        self.add_player(self.speaker)


    def _on_before_game(self):
        # print('_on_before_game', self.initial_prompt)
        # Do something before the game start e.g. add the initial prompts to the message list for the players
        self.add_user_message(self.rememberer, self.current_prompt)

    def _does_game_proceed(self):
        # Determine if the game should proceed. This is also called once initially.
        if len(self.turns) <= 0:
            return True
        return False

    def get_num_turns(self):
        return str(len(self.turns))
    
    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        print(player, utterance)
        if player == self.rememberer:
            self.log_to_self('system_response'+self.get_num_turns(), utterance)
            self.log_to_self('label'+self.get_num_turns(), self.speaker.skipped)
        self.success = True
        return True

    def _on_after_turn(self, turn_idx: int):
        self.turns.append(self.success)

    def _after_add_player_response(self, player: Player, utterance: str):
        if player == self.speaker:
            self.add_user_message(self.rememberer, utterance)
        if player == self.rememberer:
            self.add_user_message(self.speaker, utterance)            

class AskMissingScorer(GameScorer):
    def __init__(self, experiment: Dict, game_instance: Dict):
        super().__init__(GAME_NAME, experiment, game_instance)

    def compute_scores(self, episode_interactions: Dict) -> None:
        total = 0
        correct = 0
        for turn_idx, turn in enumerate(episode_interactions["turns"]):
            guess = None
            gold = None
            for event in turn:
                action = event["action"]
                if action["type"] == "system_response0":
                    guess = action['content'].lower()
                    guess = guess.replace('occupation','work')
                if action["type"] == "label0":
                    gold = action['content'].lower()
            print(guess, gold)
            if gold is not None and guess is not None:
                total += 1
                if gold in guess:
                    correct+=1
                    self.log_turn_score(turn_idx, 'Accuracy', 1.0)
                else:
                    self.log_turn_score(turn_idx, 'Accuracy', 0.0)

        self.log_episode_score('Accuracy', correct/total)

class AskMissingGameBenchmark(GameBenchmark):

    def __init__(self):
        super().__init__(GAME_NAME)

    def get_description(self):
        return "The Speaker tells the Rememberer information about a person. The Rememberer needs to ask about a field that wasn't specified."

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return AskMissing(experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return AskMissingScorer(experiment, game_instance)
