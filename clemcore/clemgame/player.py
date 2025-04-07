import abc
from copy import deepcopy
from datetime import datetime
from typing import List, Dict, Union

from clemcore import backends
from clemcore.clemgame.recorder import GameRecorder, NoopGameRecorder


class Player(abc.ABC):
    """A participant of a game.

    A player can respond via a custom implementation, human input or a language model:

    - the programmatic players are called via the _custom_response() method
    - the human players are called via the _terminal_response() method
    - the backend players are called via the generate_response() method of the backend
    """

    def __init__(self, model: backends.Model, name: str = None, game_recorder: GameRecorder = None,
                 initial_prompt: Union[str, Dict] = None, forget_extras: List[str] = None):
        """
        Args:
            model: The model used by this player.
            name: The player's name (optional). If not given, then automatically assigns a name like "Player 1 (Class)"
            game_recorder: The recorder for game interactions (optional). Default: NoopGameRecorder.
            initial_prompt: The initial prompt given to the player (optional). Note that the initial prompt must be
                            set before the player is called the first time. If set, then on the first player call
                            the initial prompt will be added to the player's message history and logged as a
                            'send message' event without a response event. To properly log this make sure that a proper
                            game recorder is set. On each player call the initial prompt will be automatically merged
                            with the first memorized context given to the player (via two newlines) by the backend.
                            Alternatively, the initial prompt could be part of the first context given to the player.
            forget_extras: A list of context entries (keys) to forget after response generation.
                           This is useful to not keep image extras in the player's message history,
                           but still to prompt the model with an image given in the context.
        """
        self._model: backends.Model = model
        self._name: str = name  # set by master
        self._game_recorder = game_recorder or NoopGameRecorder()  # set by master
        self._is_initial_call: bool = True
        self._initial_prompt: Dict = None if initial_prompt is None else self.__validate_initial_prompt(initial_prompt)
        self._forget_extras: List[str] = forget_extras or []  # set by game developer
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
    def initial_prompt(self):
        return self._initial_prompt

    @initial_prompt.setter
    def initial_prompt(self, prompt: Union[str, Dict]):
        if prompt is None:
            self._initial_prompt = None  # allow to unset the initial prompt (again)
            return
        self._initial_prompt = self.__validate_initial_prompt(prompt)

    def __validate_initial_prompt(self, prompt: Union[str, Dict]) -> Dict:
        assert self._is_initial_call is True, "The initial prompt can only be set before the first player call"
        assert isinstance(prompt, (str, dict)), \
            f"The initial prompt must be a str or dict, but is {type(prompt)}"
        if isinstance(prompt, dict):
            assert "role" in prompt and prompt["role"] == "user", \
                "The initial prompt required a 'role' entry with value 'user'"
            return deepcopy(prompt)
        return dict(role="user", content=prompt)  # by default assume str

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

    def __log_send_context_event(self, content: str, label=None):
        assert self._game_recorder is not None, "Cannot log player event, because game_recorder has not been set"
        action = {'type': 'send message', 'content': content, 'label': label}
        self._game_recorder.log_event(from_='GM', to=self.name, action=action)

    def __log_response_received_event(self, response, label=None):
        assert self._game_recorder is not None, "Cannot log player event, because game_recorder has not been set"
        action = {'type': 'get message', 'content': response, 'label': label}
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
        memorized_initial_prompt = None
        if self._is_initial_call and self._initial_prompt is not None:
            assert len(self._messages) == 0, ("There must be no entry in the player's message history "
                                              "on the first call, when the initial prompt is set.")
            memorized_initial_prompt = deepcopy(self._initial_prompt)  # see explanation below
            self._messages.append(memorized_initial_prompt)  # merged with context in ensure_alternating_roles (backend)
            self.__log_send_context_event(memorized_initial_prompt["content"], label="initial prompt")

        self.__log_send_context_event(context["content"], label="context" if memorize else "forget")
        call_start = datetime.now()
        self._prompt, self._response_object, response_text = self.__call_model(context)
        call_duration = datetime.now() - call_start
        self.__log_response_received_event(response_text, label="response" if memorize else "forget")

        self._response_object["clem_player"] = {
            "call_start": str(call_start),
            "call_duration": str(call_duration),
            "response": response_text,
            "model_name": self.model.get_name()
        }

        # Copy context, so that original context given to the player is kept on forget extras. This is, for
        # example, necessary to collect the original contexts in the rollout buffer for playpen training.
        memorized_context = deepcopy(context)
        # forget must happen only after the model has been called with the extras
        # we forget extras here in any case, so that the prompt is also handled
        for extra in self._forget_extras:
            if extra in memorized_context:
                del memorized_context[extra]
            if memorized_initial_prompt is not None and extra in memorized_initial_prompt:
                del memorized_initial_prompt[extra]

        if memorize:
            self._messages.append(memorized_context)
            self._messages.append(dict(role="assistant", content=response_text))

        self._is_initial_call = False
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
