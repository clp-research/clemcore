import abc
import collections
import logging
from copy import deepcopy
from datetime import datetime
from typing import List, Dict, Tuple, Any, Iterator

from clemcore import backends
from clemcore.clemgame.recorder import GameRecorder

module_logger = logging.getLogger(__name__)


class Player(abc.ABC):
    """A participant of a game.

    A player can respond via a custom implementation, human input or a language model:

    - the programmatic players are called via the _custom_response() method
    - the human players are called via the _terminal_response() method
    - the backend players are called via the generate_response() method of the backend
    """

    def __init__(self, model: backends.Model, game_recorder: GameRecorder):
        """
        Args:
            model: A backends.Model instance to be used by this Player instance.
        """
        self.model = model
        self.game_recorder = game_recorder
        self.messages: List[Dict] = []
        self.descriptor: str = "<missing-description>"
        self.turns: int = 0
        self.prompt = None
        self.response_object = None
        module_logger.info("Player %s", self.get_description())

    def get_description(self) -> str:
        """Get a description string for this Player instance.
        Returns:
            A string describing this Player instance's class name and used model.
        """
        return f"{self.__class__.__name__}, {self.model}"

    def __log_send_context_event(self, context, is_reprompt: bool = False):
        action_type = 'send message' if not is_reprompt else 'send message (reprompt)'
        action = {'type': action_type, 'content': context}
        self.game_recorder.log_event(from_='GM', to=self.descriptor, action=action)

    def __log_response_received_event(self, response):
        # Player -> GM
        action = {'type': 'get message', 'content': response}
        # log 'get message' event including backend/API call:
        _prompt, _response = self.get_last_call_info()
        self.game_recorder.log_event(from_=self.descriptor, to="GM", action=action, call=(_prompt, _response))

    def get_last_call_info(self):
        return self.prompt, self.response_object

    def __call__(self, context: Dict) -> str:
        assert context["role"] != "assistant", "Last entry should not be assistant " \
                                               "b.c. this would be the role of the current player"
        last_message = self.messages[-1]
        is_reprompt = last_message == context
        if not is_reprompt:
            self.turns += 1
            self.messages.append(context)

        self.__log_send_context_event(context, is_reprompt)
        call_start = datetime.now()
        self.prompt, self.response_object, response_text = self.__call_model()
        call_duration = datetime.now() - call_start
        self.__log_response_received_event(response_text)

        self.response_object["clem_player"] = {
            "call_start": str(call_start),
            "call_duration": str(call_duration),
            "response": response_text,
            "model_name": self.model.get_name()
        }

        return response_text

    def __call_model(self):
        prompt = self.messages
        response_object = dict()
        if isinstance(self.model, backends.CustomResponseModel):
            response_text = self._custom_response(self.messages)
        elif isinstance(self.model, backends.HumanModel):
            response_text = self._terminal_response(self.messages)
        else:
            prompt, response_object, response_text = self.model.generate_response(self.messages)
        return prompt, response_object, response_text

    def _terminal_response(self, messages, turn_idx) -> str:
        """Response for human interaction via terminal.
        Overwrite this method to customize human inputs (model_name: human, terminal).
        Args:
            messages: A list of dicts that contain the history of the conversation.
            turn_idx: The index of the current turn.
        Returns:
            The human response as text.
        """
        latest_response = "Nothing has been said yet."
        if messages:
            latest_response = messages[-1]["content"]
        print(f"\n{latest_response}")
        user_input = input(f"Your response as {self.__class__.__name__} (turn: {turn_idx}):\n")
        return user_input

    def _custom_response(self, messages, turn_idx) -> str:
        """Response for programmatic Player interaction.
        Overwrite this method to implement programmatic behavior (model_name: mock, dry_run, programmatic, custom).
        Args:
            messages: A list of dicts that contain the history of the conversation.
            turn_idx: The index of the current turn.
        Returns:
            The programmatic response as text.
        """
        raise NotImplementedError()


class GameMaster(GameRecorder):
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
        self.player_iter: Iterator[Player] = iter([])
        self.current_turn: int = 0
        self.current_player: Player = None
        self.is_new_turn: bool = True
        self.is_new_game: bool = True
        self.info = {}

    def get_players(self) -> List[Player]:
        """Get a list of the players.
        Returns:
            List of Player instances in the order they are added.
        """
        return list(self.players_by_names.values())

    def add_player(self, player: Player):
        """Add a player to the game.
        Note: The players will be called in the same order as added!
        Args:
            player: The player to be added to the game.
        """
        idx = len(self.players_by_names)
        player.descriptor = f"Player {idx + 1}"
        self.players_by_names[player.descriptor] = player
        self.messages_by_names[player.descriptor] = []

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
        # log player ID and description dcit:
        self.log_players(players_descriptions)
        self.player_iter = iter(self.get_players())
        self.current_player = next(self.player_iter)

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

    def get_current_player(self):
        return self.current_player

    def get_context_for(self, player):
        messages = self.messages_by_names[player.descriptor]
        return messages[-1]

    def step(self, response: str, auto_reprompt: bool = False):
        if self.is_new_game:
            self.is_new_game = False
            self._on_before_game()
        if self.is_new_turn:
            self.is_new_turn = False
            self.log_next_turn()  # not sure if we want to do this always here (or add to _on_before_turn)
            self._on_before_turn(self.current_turn)
            module_logger.info(f"{self.game_name}: %s turn: %d", self.game_name, self.current_turn)

        self.__validate_parse_and_add_player_response(self.current_player, response)
        if auto_reprompt:
            while self._should_reprompt(self.current_player):
                self._on_before_reprompt(self.current_player)
                context = self.get_context_for(self.current_player)
                response = self.current_player(context, is_reprompt=True)
                self.__validate_parse_and_add_player_response(self.current_player, response)

        try:
            # check already here for next player to increment turn and to check end condition properly
            self.current_player = next(self.player_iter)
        except StopIteration:
            self._on_after_turn(self.current_turn)
            self.current_turn += 1
            self.player_iter = iter(self.get_players())
            self.current_player = next(self.player_iter)
            self.is_new_turn = True

        done = not self._does_game_proceed()
        self.info["turn_score"] = self.compute_turn_score()
        self.info["turn_feedback"] = self.get_turn_feedback()
        self.info["episode_score"] = 0
        if done:
            self._on_after_game()
            self.info["episode_score"] = self.compute_episode_score()
        return done, deepcopy(self.info)

    def get_turn_feedback(self):
        """
        Optional.
        :return: a verbal feedback about the agent's decision-making at a turn
        """
        return None

    @abc.abstractmethod
    def compute_turn_score(self):
        """
        Mandatory.
        :return: the performance score for an agent's turn
        """
        pass

    @abc.abstractmethod
    def compute_episode_score(self):
        """
        Mandatory.
        :return: the performance of the agent over the whole episode
        """
        pass

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
            done, _ = self.step(response, auto_reprompt=True)

    def _should_reprompt(self, player: Player):
        """Method to check if a Player should be re-prompted.
        This is intended to check for invalid responses.
        Args:
            player: The Player instance to re-prompt.
        """
        return False

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
        self.log_event("GM", player.descriptor, action)

    def log_message_to_self(self, message: str):
        """Logs a 'metadata' action from GM to GM.
        This is a logging method, and will not add anything to the conversation history.
        Args:
            message: The message content logged as metadata.
        """
        action = {'type': 'metadata', 'content': message}
        self.log_event("GM", "GM", action)

    def log_to_self(self, type_: str, value: str):
        """Logs an action of the passed type from GM to GM.
        This is a logging method, and will not add anything to the conversation history.
        Args:
            type_: The type of the action to be logged.
            value: The content value of the action to be logged.
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
        history = self.messages_by_names[player.descriptor]
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

    def __validate_parse_and_add_player_response(self, player: Player, utterance: str):
        """Checks player response validity, parses it and adds it to the conversation history.
        Part of the play loop, not intended to be modified - modify _validate_player_response, _on_parse_response and/or
        _after_add_player_response instead.
        Args:
            player: The Player instance that produced the response.
            utterance: The text content of the response.
        """
        # todo: it seems we should change the order here: Parse should come first, and then validate.
        # While parse might throw a parsing (format error) validate would check solely for satisfied game rules.
        # Note: this would allow to cut off too long responses (during parse) and to only validate on the cut off piece.
        if self._validate_player_response(player, utterance):
            utterance = self.__parse_response(player, utterance)
            self.add_assistant_message(player, utterance)
            self._after_add_player_response(player, utterance)

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
