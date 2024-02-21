from typing import List, Dict, Tuple

from backends import CustomResponseModel, Model, ModelSpec
from clemgame.clemgame import Player
from clemgame import get_logger


logger = get_logger(__name__)  # clem logging


class Human(Player):
    """This is the player that is entering messages via the slurk interface.
    The player's latest message is written to _latest_slurk_message when a
    message event is received by the bot. The replied variable keeps track of
    whether there has been a reply or not.
    """

    def __init__(self, model = CustomResponseModel(ModelSpec(model_name="_slurk_response"))):
        super().__init__(model)
        self._latest_slurk_message = None  # this will be set when a message event comes in form the slurk server
        self.replied = True

    def set_current_message(self, message):
        self._latest_slurk_message = message
        self.replied = False

    def get_current_message(self):
        return self._latest_slurk_message

    def _custom_response(self, messages, turn_idx):
        """Mock contribution from human user."""
        return 'I want to know about farming.'


class Answerer(Player):

    def __init__(self, model=CustomResponseModel(), max_turns=0):
        super().__init__(model)
        self.max_turns = max_turns
        self.current_contribution = None

    def _custom_response(self, messages, turn_idx):
        if turn_idx <= self.max_turns:
            return "I don't know anything about that."
        if turn_idx > self.max_turns:
            raise Exception("We should not be here...")


class ChatGame:
    """A game between a human user typing questions on slurk (Questioner)
    and a LM (Answerer)
    """

    def __init__(self, game_instance: Dict, player_models: Tuple[Model]):
        self.player_models = player_models
        self.game_id = game_instance['game_id']
        self.max_turns = game_instance['max_turns']
        self.current_turn: int = 1
        initial_prompt = game_instance['player_2_initial_prompt']

        self.questioner: Human = Human(player_models[0])
        self.answerer: Answerer = Answerer(player_models[1], max_turns=self.max_turns)
        self.messages: List = [{"role": "system", "content": initial_prompt}]

    def proceeds(self):
        return self.current_turn <= self.max_turns

    def answerer_turn(self):
        _messages, _response_type, utterance = \
            self.answerer(self.messages, self.current_turn)
        self.messages.append({"role": "assistant", "content": utterance})
        self.current_turn += 1
        self.questioner.replied = True
        self.answerer.current_contribution = utterance

    def questioner_turn(self):
        """Adds the utterance that was typed on slurk to the messages."""
        utterance = self.questioner.get_current_message()
        self.messages.append({"role": "user", "content": utterance})
