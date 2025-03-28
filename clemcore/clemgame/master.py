import abc
import collections
import logging
from copy import deepcopy
from datetime import datetime
from typing import List, Dict, Tuple, Any, Iterator

from clemcore import backends
from clemcore.clemgame import GameResourceLocator
from clemcore.clemgame.recorder import GameRecorder

module_logger = logging.getLogger(__name__)


class Player(abc.ABC):
    """A participant of a game.

    A player can respond via a custom implementation, human input or a language model:

    - the programmatic players are called via the _custom_response() method
    - the human players are called via the _terminal_response() method
    - the backend players are called via the generate_response() method of the backend
    """
    _num_players: int = 1

    def __init__(self, model: backends.Model, name: str = None):
        """
        Args:
            model: The model used by this player.
            name: the player's name (optional). If not given, then automatically assigns a name like "Player 1 (Class)"
        """
        self._model = model
        self._game_recorder = None
        self._messages: List[Dict] = []
        self._prompt = None
        self._response_object = None
        self._name = name
        if name is None:
            self._name: str = f"Player {Player._num_players} ({self.__class__.__name__})"
            Player._num_players += 1

    def __getstate__(self):
        """
        Get the attributes to be copied.
        :return: the attributes to be copied
        """
        state = self.__dict__.copy()
        del state["_model"]  # do not copy the model: we can use the same model instance for multiple players
        del state["_game_recorder"]  # the game recorder must be reset by the game master
        return dict(state=state, _model=self._model)

    def __setstate__(self, data):
        """
        Set the state of the deep copy.
        :param data: the values for attributes of the new instance
        """
        self.__dict__.update(data["state"])
        self._model = data["_model"]
        self._game_recorder = None

    @property
    def game_recorder(self):
        return self._game_recorder

    @game_recorder.setter
    def game_recorder(self, game_recorder: GameRecorder):
        self._game_recorder = game_recorder

    @property
    def name(self):
        return self._name

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
        self._prompt, self._response_object, response_text = self.__call_model(self._messages + [context])  # new list
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

    def __call_model(self, prompt: List[Dict]):
        response_object = dict()
        if isinstance(self.model, backends.CustomResponseModel):
            response_text = self._custom_response(prompt)
        elif isinstance(self.model, backends.HumanModel):
            response_text = self._terminal_response(prompt)
        else:
            prompt, response_object, response_text = self.model.generate_response(prompt)
        return prompt, response_object, response_text

    def _terminal_response(self, messages) -> str:
        """Response for human interaction via terminal.
        Overwrite this method to customize human inputs (model_name: human, terminal).
        Args:
            messages: A list of dicts that contain the history of the conversation.
        Returns:
            The human response as text.
        """
        latest_response = "Nothing has been said yet."
        if messages:
            latest_response = messages[-1]["content"]
        print(f"\n{latest_response}")
        user_input = input(f"Your response as {self.__class__.__name__}:\n")
        return user_input

    def _custom_response(self, messages) -> str:
        """Response for programmatic Player interaction.
        Overwrite this method to implement programmatic behavior (model_name: mock, dry_run, programmatic, custom).
        Args:
            messages: A list of dicts that contain the history of the conversation.
        Returns:
            The programmatic response as text.
        """
        raise NotImplementedError()


class GameMaster(GameResourceLocator):
    """Base class to contain game-specific functionality.

    A GameMaster (sub-)class

    - prepares a concrete game instance
    - plays an episode of a game instance
    - records a game episode
    - evaluates the game episode records
    - builds the interaction transcripts
    """

    def __init__(self, name: str, path: str, experiment: Dict, player_models: List[backends.Model] = None):
        """
        Args:
            name: The name of the game (as specified in game_registry).
            path: Path to the game (as specified in game_registry).
            experiment: The experiment (set of instances) to use.
            player_models: Player models to use for one or two players.
        """
        super().__init__(name, path)
        self.experiment: Dict = experiment
        self.player_models: List[backends.Model] = player_models
        self.game_recorder = GameRecorder(name, path)

    def log_players(self, players_dict):
        self.game_recorder.log_players(players_dict)

    def log_next_turn(self):
        self.game_recorder.log_next_turn()

    def log_event(self, from_, to, action):
        self.game_recorder.log_event(from_, to, action)

    def setup(self, **kwargs):
        """Load resources and prepare everything to play the game.
        Needs to log the players dictionary via self.log_players(players_dict).
        Called by the game's GameBenchmark run method for each game instance.
        Args:
            kwargs: Keyword arguments used to set up the GameMaster instance.
        """
        raise NotImplementedError()

    def play(self) -> None:
        """Play the game (multiple turns of a specific game instance)."""
        raise NotImplementedError()


class DialogueGameMaster(GameMaster):
    """Extended GameMaster, implementing turns as described in the clembench paper.
    Has most logging and gameplay procedures implemented, including convenient logging methods.
    """

    def __init__(self, name: str, path: str, experiment: dict, player_models: List[backends.Model]):
        """
        Args:
            name: The name of the game (as specified in game_registry).
            path: Path to the game (as specified in game_registry).
            experiment: The experiment (set of instances) to use.
            player_models: Player models to use for one or two players.
        """
        super().__init__(name, path, experiment, player_models)
        # the logging works with an internal mapping of "Player N" -> Player
        self.players_by_names: Dict[str, Player] = collections.OrderedDict()
        self.messages_by_names: Dict[str, List] = dict()
        self.current_turn: int = 0
        self.current_player: Player = None
        self.current_player_idx: int = 0
        self.info = {}

    def __setstate__(self, state):
        self.__dict__.update(state)
        for player in self.players_by_names.values():  # sync game recorders (not copied in Player)
            player.game_recorder = self.game_recorder

    def get_players(self) -> List[Player]:
        """Get a list of the players.
        Returns:
            List of Player instances in the order they are added.
        """
        return list(self.players_by_names.values())

    def add_player(self, player: Player):
        """Add a player to the game. The same player cannot be added twice.
        The player identity is determined by the player's name.

        Important: During gameplay, the players will be called in the same order as added to the game master!

        Args:
            player: The player to be added to the game. The player's name must be unique.
        """
        player.game_recorder = self.game_recorder  # player should record to the same interaction log
        if player.name in self.players_by_names:
            raise ValueError(f"Player names must be unique, "
                             f"but there is already a player registered with name '{player.name}'.")
        self.players_by_names[player.name] = player
        self.messages_by_names[player.name] = []

    def setup(self, **kwargs):
        """Load resources and prepare everything to play the game.
        Needs to log the players dictionary via self.log_players(players_dict).
        Intended to be left as-is by inheriting classes. Implement game-specific setup functionality in the _on_setup
        method.
        Called by the game's GameBenchmark run method for each game instance.
        Args:
            kwargs: Keyword arguments used to set up the GameMaster instance. This is usually a game instance object
                read from the game's instances.json.
        """
        self._on_setup(**kwargs)
        # log players
        players_descriptions = collections.OrderedDict(GM=f"Game master for {self.game_name}")
        for name, player in self.players_by_names.items():
            players_descriptions[name] = player.get_description()
        self.log_players(players_descriptions)
        self.current_player = self.get_players()[self.current_player_idx]
        # call game hooks
        self._on_before_game()
        self.__prepare_next_round()

    def _on_setup(self, **kwargs):
        """Method executed at the start of the default setup method.
        Template method: Must be implemented!
        Use add_player() here to add the players.
        Args:
            kwargs: Keyword arguments of the game instance. This is usually a game instance object
                read from the game's instances.json.
        """
        raise NotImplementedError()

    def get_game_state(self):
        return None

    def get_current_player(self) -> Player:
        return self.current_player

    def get_context_for(self, player) -> Dict:
        messages = self.messages_by_names[player.name]
        assert len(messages) > 0, f"Cannot get context, because there are no messages for {player.name}"
        return messages[-1]

    def step(self, response: str) -> Tuple[bool, Dict]:
        """
        Transitions the game state by applying the current player's response.

        :param response: The response (verbal action) of the current player.
        :return: done, info
        """
        # compute scores first, so that we are sure that the player's context
        # can still be retrieved (state has not changed yet)
        context = self.get_context_for(self.current_player)
        self.info["response_score"] = self.compute_response_score(response, context)
        self.info["response_feedback"] = self.get_response_feedback(response, context)
        self.info["episode_score"] = 0

        # todo: it seems we should change the order here: Parse should come first, and then validate.
        # While parse might throw a parsing (format error) validate would check solely for satisfied game rules.
        # Note: this would allow to cut off too long responses (during parse) and to only validate on the cut off piece.
        if self._validate_player_response(self.current_player, response):
            parsed_response = self.__parse_response(self.current_player, response)
            self.add_assistant_message(self.current_player, parsed_response)
            self._after_add_player_response(self.current_player, parsed_response)

        done = not self._does_game_proceed()
        if done:
            self._on_after_game()
            self.info["episode_score"] = self.compute_episode_score()
        elif self._should_pass_turn():
            self.current_player = self._next_player()
            if self.current_player_idx == 0:  # we cycled through the whole list
                self._on_after_turn(self.current_turn)
                self.current_turn += 1
                self.__prepare_next_round()

        info = deepcopy(self.info)
        self.info = {}  # reset info after each step
        return done, info

    def _next_player(self) -> Player:
        """
        Subclasses can overwrite this method to determine the next player after a player's turn has been passed.

        Default: The gamer master passes the turn to the next player in the player list (order as added).
        Starting again with the first player, when all players have had their turn(s).

        :return: the next (current) player
        """
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players_by_names)
        return self.get_players()[self.current_player_idx]

    def _start_next_round(self) -> bool:
        """
        Subclasses can overwrite this method to specify when a next round should start after a player's turn is passed.

        Default: Start next round when we cycled through the whole list i.e. it is again the first player's turn.

        :return: True, when to start a new round
        """
        return self.current_player_idx == 0

    def __prepare_next_round(self):
        self.log_next_turn()  # add record entry for player turns
        self._on_before_turn(self.current_turn)  # call hook
        module_logger.info(f"{self.game_name}: %s turn: %d", self.game_name, self.current_turn)

    def get_response_feedback(self, response: str, context: Dict):
        """
        Optional.
        :param response: The response (verbal action) of the current player.
        :param context: The context given to the current player to generate the response for.
        :return: a verbal feedback about the player's response given the context
        """
        return None

    def compute_response_score(self, response: str, context: Dict):
        """
        Mandatory.
        :param response: The response (verbal action) of the current player.
        :param context: The context given to the current player to generate the response for.
        :return: the performance score for a player's response given the context
        """
        return 0

    def compute_episode_score(self):
        """
        Mandatory.
        :return: the performance of the agent over the whole episode
        """
        return 0

    def play(self) -> None:
        """Main play loop method.
        This method is called to run the game for benchmarking.
        Intended to be left as-is by inheriting classes. Implement additional game functionality in the
        _on_before_game, _does_game_proceed, _on_before_turn, _should_reprompt, _on_before_reprompt, _on_after_turn and
        _on_after_game methods.
        """
        done = False
        while not done:
            context = self.get_context_for(self.current_player)
            response = self.current_player(context)
            done, _ = self.step(response)

    def _should_pass_turn(self):
        """
        Whether to pass the turn to the next player.
        """
        return True

    def _on_before_reprompt(self, player: Player):
        """Method executed before reprompt is passed to a Player.
        Hook
        Change the prompt to reprompt the player on e.g. an invalid response.
        Add the new prompt to the players message via self.add_user_message(player, new_prompt)
        Args:
            player: The Player instance that produced the invalid response.
        """
        pass

    def log_message_to(self, player: Player, message: str):
        """Logs a 'send message' action from GM to Player.
        This is a logging method, and will not add the message to the conversation history on its own!
        Args:
            player: The Player instance the message is targeted at.
            message: The message content sent to the Player instance.
        """
        action = {'type': 'send message', 'content': message}
        self.log_event("GM", player.name, action)

    def log_message_to_self(self, message: str):
        """Logs a 'metadata' action from GM to GM.
        This is a logging method, and will not add anything to the conversation history.
        Args:
            message: The message content logged as metadata.
        """
        action = {'type': 'metadata', 'content': message}
        self.log_event("GM", "GM", action)

    def log_to_self(self, type_: str, value: Any):
        """Logs an action of the passed type from GM to GM.
        This is a logging method, and will not add anything to the conversation history.
        Args:
            type_: The type of the action to be logged.
            value: The content value of the action to be logged. Must be JSON serializable.
        """
        action = {'type': type_, 'content': value}
        self.log_event("GM", "GM", action)

    def add_message(self, player: Player, utterance: str, role: str):
        """Adds a message to the conversation history.
        This method is used to iteratively create the conversation history, but will not log/record messages
        automatically.
        Args:
            player: The Player instance that produced the message. This is usually a model output, but can be the game's
                GM as well, if it directly adds messages to the conversation history. TODO: Check use
            utterance: The text content of the message to be added.
            role: The chat/instruct conversation role to use for this message. Either 'user' or 'assistant', or 'system'
                for models/templates that support it. This is important to properly apply chat templates. Some chat
                templates require that roles always alternate between messages!
        """
        message = {"role": role, "content": utterance}
        history = self.messages_by_names[player.name]
        history.append(message)

    def add_user_message(self, player: Player, utterance: str):
        """Adds a message with the 'user' role to the conversation history.
        This method is to be used for 'user' messages, usually the initial prompt and GM response messages. Used to
        iteratively create the conversation history, but will not log/record messages automatically.
        Args:
            player: The Player instance that produced the message. This is usually the game's GM, if it directly adds
                messages to the conversation history. TODO: Check use
            utterance: The text content of the message to be added.
        """
        self.add_message(player, utterance, role="user")

    def add_assistant_message(self, player: Player, utterance: str):
        """Adds a message with the 'assistant' role to the conversation history.
        This method is to be used for 'assistant' messages, usually model outputs. Used to iteratively create the
        conversation history, but will not log/record messages automatically.
        Args:
            player: The Player instance that produced the message.
            utterance: The text content of the message to be added.
        """
        self.add_message(player, utterance, role="assistant")

    def _after_add_player_response(self, player: Player, utterance: str):
        """Method executed after a player response has been validated and added to the conversation history.
        Hook: Modify this method for game-specific functionality.
        Add the utterance to other player's history, if necessary. To do this use the method
        add_user_message(other_player,utterance).
        Args:
            player: The Player instance that produced the response (or has been modified by the GM).
            utterance: The text content of the message that was added.
        """
        pass

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        """Decide if an utterance should be added to the conversation history.
        Hook: Modify this method for game-specific functionality.
        This is also the place to check for game end conditions.
        Args:
            player: The Player instance for which the response is added as "assistant" to the history.
            utterance: The text content of the message to be added.
        Returns:
            True, if the utterance is fine; False, if the response should not be added to the history.
        """
        return True

    def __parse_response(self, player: Player, utterance: str) -> str:
        """Parses a response and logs the message parsing result.
        Part of the validate-parse loop, not intended to be modified - modify _on_parse_response instead.
        Args:
            player: The Player instance that produced the response.
            utterance: The text content of the response.
        Returns:
            The response content, potentially modified by the _on_parse_response method.
        """
        _utterance, log_action = self._on_parse_response(player, utterance)
        if _utterance == utterance:
            return utterance
        if log_action:
            action = {'type': 'parse', 'content': _utterance}
            self.log_event(from_="GM", to="GM", action=action)
        return _utterance

    def _on_parse_response(self, player: Player, utterance: str) -> Tuple[str, bool]:
        """Decide if a response utterance should be modified and apply modifications.
        Hook: Modify this method for game-specific functionality.
        If no modifications are applied, this method must simply return a tuple of the utterance and True.
        When a modified utterance and a true value is returned, then a 'parse' event is logged.
        Args:
            player: The Player instance that produced the response. Intended to allow for individual handling of
                different players.
            utterance: The text content of the response.
        Returns:
            A tuple of the (modified) utterance, and a bool to determine if the parse action is to be logged (default:
            True).
        """
        return utterance, True

    def _on_before_turn(self, turn_idx: int):
        """Executed in play loop after turn advance and before proceed check and prompting.
        Hook: Modify this method for game-specific functionality.
        Args:
            turn_idx: The current turn index.
        """
        pass

    def _on_after_turn(self, turn_idx: int):
        """Executed in play loop after prompting.
        Hook: Modify this method for game-specific functionality.
        Args:
            turn_idx: The current turn index.
        """
        pass

    def _does_game_proceed(self) -> bool:
        """Check if game should proceed.
        Template method: Must be implemented!
        This method is used to determine if a game should continue or be stopped. Both successful completion of the game
        and game-ending failures should lead to this method returning False.
        Returns:
            A bool, True if game continues, False if game should stop.
        """
        raise NotImplementedError()

    def _on_before_game(self):
        """Executed once at the start, before entering the play loop.
        Hook: Modify this method for game-specific functionality.
        Adding the initial prompt to the dialogue history with this method is recommended.
        """
        pass

    def _on_after_game(self):
        """Executed once at the end, after exiting the play loop.
        Hook: Modify this method for game-specific functionality.
        This method is useful to process and log/record overall game results.
        """
        pass
