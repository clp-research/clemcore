import abc
from copy import deepcopy
from typing import List, Dict, Optional

from clemcore import backends
from clemcore.clemgame.events import GameEventSource


class Player(GameEventSource):
    """A participant in a dialogue-based game, capable of generating responses
    based on a given context. A Player may be human-controlled, programmatic, or model-backed.

    Players can interact in three main ways:

    - Programmatic players implement `_custom_response(context)`
    - Human players respond via `_terminal_response(context)`
    - Model-backed players delegate to the model's `generate_response()` method
    """

    def __init__(self,
                 model: backends.Model,
                 *,
                 name: str = None,
                 game_role: str = None,
                 forget_extras: List[str] = None
                 ):
        """
        Args:
            model: The model used by this player.
            name: The player's name (optional). If not given, then automatically assigns a name like "Player 1 (Class)"
            game_role: The player's game role (optional). If not given, then automatically resolves to the class name.
            forget_extras: A list of context entries (keys) to forget after response generation.
                           This is useful to not keep image extras in the player's message history,
                           but still to prompt the model with an image given in the context.
        """
        super().__init__()
        self._model: backends.Model = model
        self._name: str = name  # set by master
        self._game_role = game_role or self.__class__.__name__
        self._forget_extras: List[str] = forget_extras or []  # set by game developer
        self._messages: List[Dict] = []  # internal state
        self._last_context = None  # internal state

    def reset(self):
        """Reset the player to its initial state.

        Typically called at the end of an interaction round.
        By default, resets the underlying model if applicable.
        """
        self._model.reset()

    def __deepcopy__(self, memo):
        """Deepcopy override method.
        Deep copies Player class object, but keeps backend model and game recorder references intact.
        We don't want to multiply the recorders on each deep copy, but have a single set for each game.
        Args:
            memo: Dictionary of objects already copied during the current copying pass. (This is a deepcopy default.)
        """
        _copy = type(self).__new__(self.__class__)
        memo[id(self)] = _copy
        for key, value in self.__dict__.items():
            if key not in ["_model", "_loggers"]:
                setattr(_copy, key, deepcopy(value, memo))
        _copy._model = self._model
        return _copy

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def game_role(self):
        return self._game_role

    @property
    def model(self):
        return self._model

    @property
    def last_context(self):
        return self._last_context

    def get_description(self) -> str:
        """Returns a human-readable description of the player instance.

        Useful for debugging or display purposes.

        Returns:
            A string including the player's name, class, and model.
        """
        return f"{self.name} ({self.__class__.__name__}): {self.model}"

    def get_perspective(self):
        """Returns the player's current memory of the dialogue.

        This reflects the player's internal state (i.e., what it 'remembers').

        Returns:
            A list of message dictionaries representing the conversation history.
        """
        return self._messages

    def get_perspective_with(self, context: Dict, *, log_event=True, memorize=True) -> List[Dict]:
        """Builds the player's perspective on the conversation including a new context message.

        This method is used to prepare an input prompt (for batching or direct generation)
        without immediately invoking a model.

        Args:
            context: The newly received message (must have `role='user'`).
            log_event: Whether to log the message as a game event.
            memorize: Whether to persist the message in the player's internal memory.

        Returns:
            A list of messages including the player's memory plus the new context.
        """
        assert context["role"] == "user", f"The context must be given by the user role, but is {context['role']}"
        self._last_context = deepcopy(context)
        if log_event:
            action = {'type': 'send message', 'content': context["content"],
                      'label': "context" if memorize else "forget"}
            self.log_event(from_='GM', to=self.name, action=action)
        # Get return value already here, because we change the internal state on memorize
        perspective = self.get_perspective() + [context]
        if memorize:
            # Copy context, so that original context given to the player is kept on forget extras. This is, for
            # example, necessary to collect the original contexts in the rollout buffer for playpen training.
            memorized_context = deepcopy(context)
            for extra in self._forget_extras:
                if extra in memorized_context:
                    del memorized_context[extra]
            self._messages.append(memorized_context)
        return perspective

    def update_perspective_with(self, response: str, *, log_event=True, memorize=True, metadata: Optional[Dict] = None):
        """Updates the player's memory with the given response and optional metadata.

        This is used after the response is generated (either from a model or a human),
        allowing the Player to log and remember its own output.

        Args:
            response: The textual response generated by the player.
            log_event: Whether to log this response as a game event.
            memorize: Whether to add the response to the player's internal memory.
            metadata: Optional dictionary with info about the model call (e.g., prompt and raw response object).
        """
        if log_event:
            action = {'type': 'get message', 'content': response,
                      'label': "response" if memorize else "forget"}
            call_infos = None
            if metadata is not None:  # log 'get message' event including backend/API call
                call_infos = (deepcopy(metadata["prompt"]), deepcopy(metadata["response_object"]))
            self.log_event(from_=self.name, to="GM", action=action, call=call_infos)
        self.count_request()
        if memorize:
            self._messages.append(dict(role="assistant", content=response))

    def __call__(self, context: Dict, memorize: bool = True) -> str:
        """Generates a response to the given context message.

        This is the primary method for turning a user message into a player reply.
        It uses the appropriate backend (custom, human, or model) and optionally
        updates internal memory and logs the interaction.

        Args:
            context: A dictionary representing the latest user input (`role='user'`).
            memorize: Whether to store the context and response in memory.

        Returns:
            The textual response produced by the player.
        """
        perspective = self.get_perspective_with(context, memorize=memorize)
        if isinstance(self.model, backends.CustomResponseModel):
            response_text = self._custom_response(context)
            response_object = dict(clem_player={"response": response_text, "model_name": self.model.name})
            metadata = dict(prompt=context, response_object=response_object)
        elif isinstance(self.model, backends.HumanModel):
            response_text = self._terminal_response(context)
            response_object = dict(clem_player={"response": response_text, "model_name": self.model.name})
            metadata = dict(prompt=context, response_object=response_object)
        else:
            prompt, response_object, response_text = self.model.generate_response(perspective)
            metadata = dict(prompt=prompt, response_object=response_object)
            # TODO: add default ContextExceededError handling here or above
        self.update_perspective_with(response_text, memorize=memorize, metadata=metadata)
        return response_text

    def _terminal_response(self, context: Dict) -> str:
        """Prompts the user via terminal input for a response.

        This is used for human-in-the-loop players.

        Args:
            context: The message from the GM to respond to.

        Returns:
            The user's textual input.
        """
        latest_response = "Nothing has been said yet."
        if context is not None:
            latest_response = context["content"]
        print(f"\n{latest_response}")
        user_input = input(f"Your response as {self.__class__.__name__}:\n")
        return user_input

    @abc.abstractmethod
    def _custom_response(self, context: Dict) -> str:
        """Implements custom or programmatic player behavior.

        This method must be overridden in subclasses that simulate agent behavior
        without relying on human or model-based backends (model_name: mock, dry_run, programmatic, custom).

        Args:
            context: The context to which the player should respond.

        Returns:
            A string containing the player's response.
        """
        pass
