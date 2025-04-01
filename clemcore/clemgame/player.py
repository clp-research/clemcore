import abc
from copy import deepcopy
from datetime import datetime
from typing import List, Dict

from clemcore import backends
from clemcore.clemgame.recorder import GameRecorder


class Player(abc.ABC):
    """A participant of a game.

    A player can respond via a custom implementation, human input or a language model:

    - the programmatic players are called via the _custom_response() method
    - the human players are called via the _terminal_response() method
    - the backend players are called via the generate_response() method of the backend
    """

    def __init__(self, model: backends.Model, name: str = None, game_recorder: GameRecorder = None):
        """
        Args:
            model: The model used by this player.
            name: the player's name (optional). If not given, then automatically assigns a name like "Player 1 (Class)"
        """
        self._model = model
        self._game_recorder = game_recorder  # set by master
        self._name = name  # set by master
        self._messages: List[Dict] = []  # internal state
        self._prompt = None  # internal state
        self._response_object = None  # internal state

    def __deepcopy__(self, memo):
        _copy = type(self).__new__(self.__class__)
        memo[id(self)] = _copy
        for key, value in self.__dict__.items():
            if key not in ["_model", "_game_recorder"]:
                setattr(_copy, key, deepcopy(value, memo))
        _copy._model = self._model
        return _copy

    @property
    def game_recorder(self):
        return self._game_recorder

    @game_recorder.setter
    def game_recorder(self, game_recorder: GameRecorder):
        self._game_recorder = game_recorder

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def model(self):
        return self._model

    def get_description(self) -> str:
        """Get a description string for this Player instance.
        Returns:
            A string describing the player's class, model used and name given.
        """
        return f"{self.name} ({self.__class__.__name__}): {self.model}"

    def __log_send_context_event(self, context):
        assert self._game_recorder is not None, "Cannot log player event, because game_recorder has not been set"
        action = {'type': 'send message', 'content': context}
        self._game_recorder.log_event(from_='GM', to=self.name, action=action)

    def __log_response_received_event(self, response):
        assert self._game_recorder is not None, "Cannot log player event, because game_recorder has not been set"
        action = {'type': 'get message', 'content': response}
        _prompt, _response = self.get_last_call_info()  # log 'get message' event including backend/API call
        self._game_recorder.log_event(from_=self.name, to="GM", action=action, call=(_prompt, _response))

    def get_last_call_info(self):
        return self._prompt, self._response_object

    def __call__(self, context: Dict, memorize: bool = True) -> str:
        """
        Let the player respond (act verbally) to a given context.

        :param context: The context to which the player should respond.
        :param memorize: Whether the context and response are to be added to the player's message history.
        :return: the textual response
        """
        assert context["role"] == "user", f"The context must be given by the user role, but is {context['role']}"

        self.__log_send_context_event(context["content"])
        call_start = datetime.now()
        self._prompt, self._response_object, response_text = self.__call_model(context)  # new list
        call_duration = datetime.now() - call_start
        self.__log_response_received_event(response_text)

        self._response_object["clem_player"] = {
            "call_start": str(call_start),
            "call_duration": str(call_duration),
            "response": response_text,
            "model_name": self.model.get_name()
        }

        if memorize:
            self._messages.append(context)
            self._messages.append(dict(role="assistant", content=response_text))

        return response_text

    def __call_model(self, context: Dict):
        response_object = dict()
        prompt = context
        if isinstance(self.model, backends.CustomResponseModel):
            response_text = self._custom_response(context)
        elif isinstance(self.model, backends.HumanModel):
            response_text = self._terminal_response(context)
        else:
            prompt, response_object, response_text = self.model.generate_response(self._messages + [context])
        return prompt, response_object, response_text

    def _terminal_response(self, context: Dict) -> str:
        """Response for human interaction via terminal.
        Overwrite this method to customize human inputs (model_name: human, terminal).
        Args:
            context: The dialogue context to which the player should respond.
        Returns:
            The human response as text.
        """
        latest_response = "Nothing has been said yet."
        if context is not None:
            latest_response = context["content"]
        print(f"\n{latest_response}")
        user_input = input(f"Your response as {self.__class__.__name__}:\n")
        return user_input

    @abc.abstractmethod
    def _custom_response(self, context: Dict) -> str:
        """Response for programmatic Player interaction.

        Overwrite this method to implement programmatic behavior (model_name: mock, dry_run, programmatic, custom).
        Args:
            context: The dialogue context to which the player should respond.
        Returns:
            The programmatic response as text.
        """
        pass
