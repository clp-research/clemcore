"""
Basic structures for the players and the game QAs.
"""

import copy
import random
from typing import List, Dict, Any, Tuple

from backends import Model, CustomResponseModel
from clemgame.clemgame import Player
from clemgame.file_utils import load_json
from games.privateshared.constants import (REQUESTS_PATH, GAME_NAME, YES, NO, 
                                           ANSWER, ASIDE)


class Answerer(Player):
    """Implement the Answerer player, making API calls to get utterances."""
    def __init__(self, model: Model):
        super().__init__(model)

    def _custom_response(self, messages: Any, turn_idx: int) -> str:
        """Return a mock response with a tag and possibly a yes/no prefix."""
        r = random.random()
        # randomly decide whether to start with yes, no or nothing
        begin = ''
        if r < 0.33:
            begin = f'{NO}, '
        elif r < 0.66:
            begin = f'{YES}, '
        # randomly select an initial tag.
        tag = ANSWER if random.random() < 0.5 else ASIDE
        return f'{tag}{begin}placeholder for turn {turn_idx}.'


class Questioner(Player):
    """Programmatic realisation of the Questioner player."""
    def __init__(self,
                 exp_name: str,
                 max_turns: int,
                 question_order: List[str],
                 requests: Dict[str, int]
                 ):
        super().__init__(CustomResponseModel())
        request_strings = load_json(REQUESTS_PATH.format(exp_name), GAME_NAME)
        self.max_turns = max_turns
        self.question_order = question_order
        self.requests = requests
        self.request_strings = request_strings

    def _custom_response(self, messages: Any, turn_idx: int) -> str:
        """Return the request utterance for a given turn."""
        if turn_idx >= self.max_turns:
            raise IndexError('Maximum turns already reached!')
        question_type = self.question_order[turn_idx]
        request_idx = self.requests[question_type]
        return self.request_strings[question_type][request_idx]


class PrivateSharedGame:
    """Basic QA mechanism, to be called by the game master."""
    def __init__(self,
                 subtype: str,
                 request_order: List[str],
                 requests: Dict[str, int],
                 slots: Dict[str, str],
                 model: Model
                 ):
        self.slots = slots
        self.max_turns: int = len(self.slots)
        self.request_order = request_order
        self.answerer: Answerer = Answerer(model)
        self.questioner: Questioner = Questioner(
            subtype, self.max_turns, request_order, requests)
        self.messages: List = []
        self.current_turn: int = 0

    def proceeds(self) -> bool:
        """Check if the game can continue, i.e. not all slots are filled."""
        return self.current_turn < self.max_turns

    def initiate(self, initial_prompt: str) -> None:
        """Add initial prompt to the dialogue history."""
        self.messages.append({'role': 'user', 'content': initial_prompt})
        # append a "fake" turn to avoid adjacent user turns
        self.messages.append({'role': 'assistant', 'content': "Ok."})

    def questioner_turn(self, tag: str) -> str:
        """Append tagged next question to dialogue history and return it."""
        _, _, request = self.questioner(self.messages, self.current_turn)
        tagged_request = f"{tag}{request}"
        self.messages.append({'role': 'user', 'content': tagged_request})
        return tagged_request

    def answerer_turn(self) -> Tuple[Any, Any, str]:
        """
        Get response via API call, append it to dialogue history and return 
        manipulated prompt and response.
        """
        prompt, raw_answer, answer = self.answerer(self.messages,
                                                   self.current_turn)
        # make a copy to log a static state
        prompt = copy.deepcopy(prompt)
        self.messages.append({'role': 'assistant', 'content': answer})
        # increase the turn counter
        self.current_turn += 1
        return prompt, raw_answer, answer
