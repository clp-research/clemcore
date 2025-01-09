from typing import Dict, Tuple, List, Union

import numpy as np

from backends import Model, CustomResponseModel
from clemgame.clemgame import GameMaster, GameBenchmark, Player, DialogueGameMaster, GameScorer
from clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, METRIC_REQUEST_COUNT, \
    METRIC_REQUEST_COUNT_VIOLATED, METRIC_REQUEST_COUNT_PARSED, METRIC_REQUEST_SUCCESS, BENCH_SCORE
from clemgame import get_logger
from clemgame import file_utils, string_utils

GAME_NAME = "memory_narrative_turns"

logger = get_logger(__name__)


class Speaker(Player):

    def __init__(self, question, answer):
        super().__init__(CustomResponseModel())
        self.answer = answer
        self.question = question

    def _custom_response(self, messages, turn_idx):
        return self.question

class Rememberer(Player):

    def __init__(self, model: Model):
        super().__init__(model)

    def _custom_response(self, messages, turn_idx):
        raise NotImplementedError("This should not be called, but the remote APIs.")

class MemoryNarrativeTurns(DialogueGameMaster):
    """This class implements a greeting game in which player A
    is greeting another player with a target name.
    """

    def __init__(self, experiment: Dict, player_models: List[Model]):
        super().__init__(GAME_NAME, experiment, player_models)
        # self.max_turns: int = experiment["max_turns"]
        self.initial_prompt = experiment["initial_prompt"]
        self.language: int = experiment["language"]  # fetch experiment parameters here
        self.question_pre_prompt = '$QUESTION$'
        self.turns = []
        # self.success = True


    def _on_setup(self, **game_instance):
        self.game_instance = game_instance  # fetch game parameters here
        self.game_id = self.game_instance['game_id']
        # Create the players
        self.question = self.question_pre_prompt.replace('$QUESTION$', self.game_instance['question'])
        self.speaker = Speaker(self.question, self.game_instance['answer'])
        self.rememberer = Rememberer(self.player_models[0])
        if self.game_id == 0:
            self.current_prompt = self.initial_prompt + self.game_instance['prompt'] 
        else:
            self.current_prompt = self.game_instance['prompt'] 

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
        if len(self.turns) <= 1:
            return True
        return False

    def get_num_turns(self):
        return str(len(self.turns))
    
    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        if player == self.rememberer:
            self.log_to_self('response'+self.get_num_turns(), utterance)
            self.log_to_self('answer'+self.get_num_turns(), self.speaker.answer)
        self.success = True
        return True

    def _on_after_turn(self, turn_idx: int):
        self.turns.append(self.success)

    def _after_add_player_response(self, player: Player, utterance: str):
        if player == self.speaker:
            self.add_user_message(self.rememberer, utterance)
        if player == self.rememberer:
            self.add_user_message(self.speaker, utterance)            

class MemoryNarrativeTurnsScorer(GameScorer):
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
                if action["type"] == "response1":
                    guess = action['content']
                    guess = guess.replace('.', '').replace('"', '').replace('*','').split(' ')[-1] # the guess is always the last word in the complete sentence
                if action["type"] == "answer1":
                    gold = action['content']
            if gold is not None and guess is not None:
                total += 1
                if guess == gold:
                    correct+=1
                    self.log_turn_score(turn_idx, 'Accuracy', 1.0)
                else:
                    self.log_turn_score(turn_idx, 'Accuracy', 0.0)

        self.log_episode_score('Accuracy', correct/total)

class MemoryNarrativeTurnsGameBenchmark(GameBenchmark):

    def __init__(self):
        super().__init__(GAME_NAME)

    def get_description(self):
        return "Memory game between two agents: the speaker tells the rememberer about person contact info (e.g., first and last names, hobbies) and then asks questions about what Rememberer remembers about a person."

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return MemoryNarrativeTurns(experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return MemoryNarrativeTurnsScorer(experiment, game_instance)

