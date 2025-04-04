import abc
from copy import deepcopy
from datetime import datetime
from typing import List, Dict

from clemcore import backends
from clemcore.clemgame.recorder import GameRecorder, NoopGameRecorder


class Player(abc.ABC):
    """A participant of a game.

    A player can respond via a custom implementation, human input or a language model:

    - the programmatic players are called via the _custom_response() method
    - the human players are called via the _terminal_response() method
    - the backend players are called via the generate_response() method of the backend
    """

    def __init__(self, model: backends.Model, name: str = None,
                 game_recorder: GameRecorder = None, forget_extras: List[str] = None):
        """
        Args:
            model: The model used by this player.
            name: The player's name (optional). If not given, then automatically assigns a name like "Player 1 (Class)"
            game_recorder: The recorder for game interactions (optional). Default: NoopGameRecorder.
            forget_extras: A list of context entries (keys) to forget after response generation.
                           This is useful to not keep image extras in the player's message history,
                           but still to prompt the model with an image given in the context.
        """
        self._model = model
        self._name = name  # set by master
        self._game_recorder = game_recorder or NoopGameRecorder()  # set by master
        self._forget_extras = forget_extras or []  # set by game developer
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

    def __log_send_context_event(self, content: str, memorize=True):
        assert self._game_recorder is not None, "Cannot log player event, because game_recorder has not been set"
        _type = 'send message' if memorize else 'send message (forget)'
        action = {'type': _type, 'content': content}
        self._game_recorder.log_event(from_='GM', to=self.name, action=action)

    def __log_response_received_event(self, response, memorize=True):
        assert self._game_recorder is not None, "Cannot log player event, because game_recorder has not been set"
        _type = 'get message' if memorize else 'get message (forget)'
        action = {'type': _type, 'content': response}
        _prompt, _response = self.get_last_call_info()  # log 'get message' event including backend/API call
        self._game_recorder.log_event(from_=self.name, to="GM", action=action,
                                      call=(deepcopy(_prompt), deepcopy(_response)))

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

        self.__log_send_context_event(context["content"], memorize)
        call_start = datetime.now()
        self._prompt, self._response_object, response_text = self.__call_model(context)  # new list
        call_duration = datetime.now() - call_start
        self.__log_response_received_event(response_text, memorize)

        self._response_object["clem_player"] = {
            "call_start": str(call_start),
            "call_duration": str(call_duration),
            "response": response_text,
            "model_name": self.model.get_name()
        }

        if memorize:
            if self._forget_extras:
                # Copy context, so that original context given to the player is kept. This is, for example,
                # necessary to collect the original contexts in the rollout buffer for playpen training.
                context = deepcopy(context)
                for extra in self._forget_extras:
                    if extra in context:
                        del context[extra]
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
