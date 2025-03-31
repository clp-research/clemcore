import abc
import collections
import logging
from copy import deepcopy
from typing import List, Dict, Tuple, Any

from clemcore import backends
from clemcore.clemgame import GameResourceLocator
from clemcore.clemgame.player import Player
from clemcore.clemgame.recorder import NoopGameRecorder

module_logger = logging.getLogger(__name__)


class GameMaster(abc.ABC):
    """Base class to contain game-specific functionality.

    A GameMaster (sub-)class

    - prepares a concrete game instance
    - plays an episode of a game instance
    - records a game episode
    - evaluates the game episode records
    - builds the interaction transcripts
    """

    def __init__(self, name: str, path: str, experiment: Dict, player_models: List[backends.Model]):
        """
        Args:
            name: The name of the game (as specified in game_registry).
            path: Path to the game (as specified in game_registry).
            experiment: The parameter of the experiment, that is, parameters that are the same for all game instances.
            player_models: Player models to use for one or two players.
            game_recorder: Enables to log each interaction in the game.
                           Resulting records can be stored to an interactions.json.
        """
        self.game_name = name
        self.experiment: Dict = experiment
        self.player_models: List[backends.Model] = player_models
        self._game_recorder = NoopGameRecorder()
        self.game_resources = GameResourceLocator(name, path)  # could be obsolete, when all info is in the instances

    @property
    def game_recorder(self):
        return self._game_recorder

    @game_recorder.setter
    def game_recorder(self, game_recorder):
        self._game_recorder = game_recorder

    def load_json(self, file_name):
        return self.game_resources.load_json(file_name)

    def load_template(self, file_name):
        return self.game_resources.load_template(file_name)

    def log_key(self, key: str, value: Any):
        self._game_recorder.log_key(key, value)

    def log_players(self, players_dict):
        self._game_recorder.log_players(players_dict)

    def log_next_round(self):
        self._game_recorder.log_next_round()

    def log_event(self, from_, to, action):
        self._game_recorder.log_event(from_, to, action)

    def store_records(self, results_root, dialogue_pair_desc, game_record_dir):
        self._game_recorder.store_records(results_root, dialogue_pair_desc, game_record_dir)

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
        self.current_round: int = 0
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
        player.name = f"Player {len(self.players_by_names) + 1}"
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

        if self._should_pass_turn():
            self.current_player = self._next_player()
            if self._start_next_round():
                self._on_after_round()
                self.current_round += 1

        done = not self._does_game_proceed()
        if done:
            self._on_after_game()
            self.info["episode_score"] = self.compute_episode_score()
        elif self._start_next_round():
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
        self.log_next_round()  # add record entry for player turns
        self._on_before_round()  # call hook
        module_logger.info(f"{self.game_name}: %s turn: %d", self.game_name, self.current_round)

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

    def _on_before_round(self):
        """Executed in the play loop before a new round of gameplay starts.

        Hook: Modify this method for game-specific functionality.
        """
        pass

    def _on_after_round(self):
        """Executed in the play loop after a round of gameply finished i.e. _start_next_round() resolves to True.

        Hook: Modify this method for game-specific functionality.
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
