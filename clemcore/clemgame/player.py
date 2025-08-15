import abc
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict, Optional, Tuple

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
        _copy._loggers = []  # we don't want to copy loggers, but the list must be initialized
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

    def perceive_context(self, context: Dict, *, log_event=True, memorize=True) -> List[Dict]:
        """Processes a new context message and returns the player's updated perspective.

        This method allows the Player to receive and optionally memorize a new user message
        (e.g., from the GameMaster), without immediately generating a response.

        It is typically used when preparing input for batched or deferred model invocation.

        Args:
            context: A new user message to process (must have `role='user'`).
            log_event: Whether to log this context as a game event (e.g., for visualization or tracking).
            memorize: Whether to integrate the message into the player's internal memory.

        Returns:
            A list of messages representing the player's updated perspective on the conversation,
            including the newly received context.
        """
        assert context["role"] == "user", f"The context must be given by the user role, but is {context['role']}"
        self._last_context = deepcopy(context)
        if log_event:
            action = {'type': 'send message', 'content': context["content"],
                      'label': "context" if memorize else "forget"}
            if "image" in context:
                action["image"] = context["image"]
            self.log_event(from_='GM', to=self.name, action=action)
        # Get return value already here, because we might want to forget context extras on memorize
        updated_perspective = self.get_perspective() + [context]
        if memorize:
            # Copy context, so that original context given to the player is kept on forget extras. This is, for
            # example, necessary to collect the original contexts in the rollout buffer for playpen training.
            memorized_context = deepcopy(context)
            for extra in self._forget_extras:
                if extra in memorized_context:
                    del memorized_context[extra]
            self._messages.append(memorized_context)
        return updated_perspective

    def perceive_response(self, response: str, *, log_event=True, memorize=True, metadata: Optional[Dict] = None) -> \
            List[Dict]:
        """Processes a response generated by the player and updates internal state accordingly.

        This method should be called after a response has been produced (e.g., by a model or human),
        to optionally log the message and integrate it into the player's memory.

        Args:
            response: The textual response generated by the player.
            log_event: Whether to log this response as a game event (including call metadata, if available).
            memorize: Whether to add the response to the player's internal memory.
            metadata: Optional metadata containing call details, such as the input prompt and response object.

        Returns:
            A list of messages representing the player's updated perspective on the conversation,
            including the response.
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
        return self.get_perspective()

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
        perspective = self.perceive_context(context, memorize=memorize)
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
        self.perceive_response(response_text, memorize=memorize, metadata=metadata)
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

    @staticmethod
    def batch_response(players: List["Player"],
                       contexts: List[Dict],
                       *,
                       row_ids: Optional[List[int]] = None,
                       ) -> Dict[int, Tuple[Dict, str]]:
        """
        Generates batched responses for a list of players based on their individual models and contexts.

        This method processes each player's context to produce a prompt using the `perceive_context()` method.
        Players are grouped by the name of their associated model backend to allow batch processing via
        `generate_batch_response()`. Each batch of responses is then distributed back to the originating
        players using `perceive_response()`.

        Args:
            players (List[Player]): A list of `Player` instances that will generate responses.
            contexts (List[Dict]): A list of context dictionaries, one per player, used to generate prompts.
            row_ids (List[int]): A list of unique identifiers (e.g., session or row IDs), aligned with the
                                 `players` and `contexts` lists. These are used to map responses back to sessions.

        Returns:
            Dict[int, str]: A dictionary mapping each `row_id` to the corresponding context and its response text
                            generated by the appropriate model backend.

        Raises:
            AssertionError: If the lengths of `players`, `contexts`, and `row_ids` do not match.
            AssertionError: If a model returns a number of responses that doesn't match the number of prompts
                            it was given.
            AttributeError: If a model does not implement the required `generate_batch_response` method.

        Notes:
            - Models are grouped by name (not by instance) to avoid issues with unhashable model objects.
            - The order of inputs is preserved during batch generation to ensure correct mapping of responses
              to players and session IDs.
        """
        if row_ids is None:
            row_ids = list(range(len(players)))
        assert len(row_ids) == len(players), (
            f"`row_ids` and `players` must have the same length: {len(row_ids)} != {len(players)}"
        )
        assert len(players) == len(contexts), (
            f"`players` and `contexts` must have the same length: {len(players)} != {len(contexts)}"
        )
        # Assert all models implement `generate_batch_response` up front
        for player in players:
            assert hasattr(player.model, "generate_batch_response"), (
                f"Model '{player.model}' used by player '{player}' does not implement `generate_batch_response()`"
            )

        # Index models by name (since Model objects may not be hashable)
        model_by_name = {player.model.name: player.model for player in players}

        # Group inputs by model, tracking (row_id, player, perspective)
        input_batch_by_model: Dict[str, List[Tuple[int, Player, List[Dict], Dict]]] = defaultdict(list)
        for row_id, player, context in zip(row_ids, players, contexts):
            perspective = player.perceive_context(context)
            input_batch_by_model[player.model.name].append((row_id, player, perspective, context))

        # Collect responses per row_id
        context_response_by_row_id = {}
        for model_name, batched_inputs in input_batch_by_model.items():
            # Build input batch and track mapping back to session/player
            row_mapping: Dict[int, Tuple[int, Player, Dict]] = {}
            batched_perspectives: List[List[Dict]] = []
            for prompt_idx, (row_id, player, perspective, context) in enumerate(batched_inputs):
                row_mapping[prompt_idx] = (row_id, player, context)
                batched_perspectives.append(perspective)

            # Run batched generation (assumes order-preserving)
            model = model_by_name[model_name]
            if model.model_spec.is_programmatic():
                model.set_gen_arg("players", players)
            results = model.generate_batch_response(batched_perspectives)
            assert len(results) == len(batched_perspectives), (
                f"Model '{model_name}' returned {len(results)} responses, "
                f"but {len(batched_perspectives)} prompts were sent."
            )

            # Each result is assumed to be (prompt, response_object, response_text)
            for prompt_idx, (perspective, response_object, response_text) in enumerate(results):
                row_id, player, context = row_mapping[prompt_idx]
                context_response_by_row_id[row_id] = (context, response_text)
                metadata = dict(prompt=perspective, response_object=response_object)
                player.perceive_response(response_text, metadata=metadata)

        return context_response_by_row_id
