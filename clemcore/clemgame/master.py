import abc
import collections
import logging
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Tuple, Any, Union, final, Optional

from clemcore import backends
from clemcore.clemgame.environment import Action, GameEnvironment
from clemcore.clemgame.errors import ParseError, GameError
from clemcore.clemgame.registry import GameSpec
from clemcore.clemgame.player import Player
from clemcore.clemgame.recorder import NoopGameRecorder
from clemcore.clemgame.resources import GameResourceLocator
from clemcore.utils.string_utils import to_pretty_json

module_logger = logging.getLogger(__name__)


class GameMaster(abc.ABC):
    """Base class to contain game-specific functionality."""

    def __init__(self, game_spec: GameSpec, experiment: Dict, player_models: List[backends.Model]):
        """
        Args:
            game_spec: the game specifications for this game as given in the clemgame.json file
            experiment: The parameter of the experiment, that is, parameters that are the same for all game instances.
            player_models: Player models to use for one or two players.
        """
        self.game_spec = game_spec
        self.experiment: Dict = experiment
        # Automatic player expansion: When only a single model is given, then use this model given for each game role.
        if len(player_models) == 1 and game_spec.players > 1:
            player_models = [player_models[0]] * game_spec.players  # keeps original list untouched
        if len(player_models) != game_spec.players:
            raise ValueError(f"{game_spec.game_name} requires {game_spec.players} players, "
                             f"but {len(player_models)} were given: {[m.name for m in player_models]}")
        self.player_models: List[backends.Model] = player_models
        self._game_recorder = NoopGameRecorder()
        # Note: Using GameResourceLocator could be obsolete, when all necessary info is in the instances file.
        self.game_resources = GameResourceLocator(game_spec.game_name, game_spec.game_path)

    @property
    def game_recorder(self):
        return self._game_recorder

    @game_recorder.setter
    def game_recorder(self, game_recorder):
        self._game_recorder = game_recorder

    def load_json(self, file_path: Union[str, Path]):
        return self.game_resources.load_json(file_path)

    def load_template(self, file_path: Union[str, Path]):
        return self.game_resources.load_template(file_path)

    def log_to_self(self, type_: str, value: Any):
        """Logs an action of the passed type from GM to GM.
        This is a logging method, and will not add anything to the conversation history.
        Args:
            type_: The type of the action to be logged.
            value: The content value of the action to be logged. Must be JSON serializable.
        """
        self._game_recorder.log_event("GM", "GM", {"type": type_, "content": value})

    def log_key(self, key: str, value: Any):
        self._game_recorder.log_key(key, value)

    def log_player(self, player: Player):
        self._game_recorder.log_player(player.name, player.game_role, player.model.name)

    def log_next_round(self):
        self._game_recorder.log_next_round()

    def log_event(self, from_, to, action):
        self._game_recorder.log_event(from_, to, action)

    def store_records(self, results_root, dialogue_pair_desc, game_record_dir):
        self._game_recorder.store_records(results_root, dialogue_pair_desc, game_record_dir)

    @abc.abstractmethod
    def setup(self, **kwargs):
        """Load resources and prepare everything to play the game.
        Needs to log the player infos via self.log_player().
        Called by the game's GameBenchmark run method for each game instance.
        Args:
            kwargs: Keyword arguments used to set up the GameMaster instance.
        """
        pass

    @abc.abstractmethod
    def play(self) -> None:
        """Play the game (multiple turns of a specific game instance)."""
        pass


class DialogueGameMaster(GameMaster):
    """Extended GameMaster, implementing turns as described in the clembench paper.
    Has most logging and gameplay procedures implemented, including convenient logging methods.
    """

    def __init__(self, game_spec: GameSpec, experiment: dict, player_models: List[backends.Model]):
        """
        Args:
            name: The name of the game (as specified in game_registry).
            path: Path to the game (as specified in game_registry).
            experiment: The experiment (set of instances) to use.
            player_models: Player models to use for one or two players.
        """
        super().__init__(game_spec, experiment, player_models)
        # the logging works with an internal mapping of "Player N" -> Player
        self.players_by_names: Dict[str, Player] = collections.OrderedDict()
        self.context_for_player: Dict[str, Dict] = dict()  # context entries look like {"role":"user", "content": ...}
        self.current_round: int = 0
        self._current_player: Player = None
        self._current_player_idx: int = 0
        self.info = {}

    def __setstate__(self, state):
        self.__dict__.update(state)
        for player in self.players_by_names.values():  # sync game recorders (not copied in Player)
            player.game_recorder = self.game_recorder

    @property
    def game_state(self):
        return None

    @property
    def current_player(self) -> Player:
        return self._current_player

    @final
    def get_players(self) -> List[Player]:
        """Get a list of the players.
        Returns:
            List of Player instances in the order they are added.
        """
        return list(self.players_by_names.values())

    @final
    def add_player(self, player: Player, initial_prompt: Union[str, Dict] = None,
                   initial_context: Union[str, Dict] = None):
        """Add a player to the game. The same player cannot be added twice.
        The player identity is determined by the player's name.

        Important: During gameplay, the players will be called in the same order as added to the game master!

        Args:
            player: The player to be added to the game. The player's name must be unique.
            initial_prompt: The initial prompt given to the player (optional). See Player for more details.
            initial_context: A context to be immediately set for the player (optional). This is useful for initial
                            prompts that are supposed to be handled as the first context, for example, when adding
                            the other player's response to the prompt is not necessary, but the player is supposed
                            to directly react to the initial prompt. Alternatively, overwrite on_before_game() and
                            use set_context_for(player) to set the player context.
        """
        player.game_recorder = self.game_recorder  # player should record to the same interaction log
        player.initial_prompt = initial_prompt
        player.name = f"Player {len(self.players_by_names) + 1}"
        if player.name in self.players_by_names:
            raise ValueError(f"Player names must be unique, "
                             f"but there is already a player registered with name '{player.name}'.")
        self.players_by_names[player.name] = player
        self.log_player(player)
        if initial_context is not None:
            assert isinstance(initial_context, (str, dict)), \
                f"The initial context must be a str or dict, but is {type(initial_context)}"
            if isinstance(initial_context, dict):
                assert "content" in initial_context, "The initial context requires a content entry"
                extras = {k: v for k, v in initial_context.items() if k not in ["role", "content"]}
                self.set_context_for(player, initial_context["content"], **extras)
            else:
                self.set_context_for(player, initial_context)

    @final
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
        self._current_player = self.get_players()[self._current_player_idx]
        self._on_before_game()
        self._on_before_round()

    @abc.abstractmethod
    def _on_setup(self, **kwargs):
        """Method executed at the start of the default setup method.
        Template method: Must be implemented!
        Use add_player() here to add the players.
        Args:
            kwargs: Keyword arguments of the game instance. This is usually a game instance object
                read from the game's instances.json.
        """
        pass

    @final
    def set_context_for(self, player: Player, content: str, **extras):
        """
        Set the context for the specified Player. The player will be prompted with the context on its next turn.

        The context always has a 'role' and 'content' entry where the 'role' is always set to 'user'.
        Args:
            player: The player to set the context for.
            content: The text content to be added to the context.
            extras: Additional content to be merged into the context e.g. information about images
        """
        if player is None:
            return
        message = {"role": "user", "content": content}
        context = {**extras, **message}
        self.context_for_player[player.name] = context

    @final
    def get_context_for(self, player) -> Dict:
        assert player is not None, "Cannot get player context for 'None'"
        assert player.name in self.context_for_player, f"No context set for {player.name}"
        context = self.context_for_player[player.name]
        assert "role" in context, f"Player context must have a 'role' entry"
        assert context["role"] == "user", f"Role of player context must be 'user'"
        assert "content" in context, f"Player context must have a 'content' entry"
        return context

    @final
    def play(self) -> None:
        """
        Main play loop method. This method is called to run the game for benchmarking.
        """
        done = False
        while not done:
            context = self.get_context_for(self.current_player)
            response = self.current_player(context)
            done, _ = self.step(response)

    @final
    def step(self, response: str) -> Tuple[bool, Dict]:
        """
        Verifies the response and transitions the game by applying the current player's response for the turn.

        :param response: The response (verbal action) of the current player.
        :return: done, info
        """
        try:
            parsed_response = self._parse_response(self.current_player, response)  # throws ParseError
            self._advance_game(self.current_player, parsed_response)  # throws GameError
        except ParseError as error:
            self._game_recorder.count_request_violation()
            self._on_parse_error(error)
        except GameError as error:
            self._on_game_error(error)

        self.info["turn_score"] = self.compute_turn_score()
        self.info["turn_feedback"] = self.get_turn_feedback()

        # determine if the current player should pass the turn to the next player or get another turn:
        if self._should_pass_turn():  # True = move on to next player
            self._current_player = self._next_player()

        if self._start_next_round():
            self._on_after_round()
            self.current_round += 1  # already increment here b.c. _does_game_proceed might rely on it

        done = not self._does_game_proceed()
        if done:
            self._on_after_game()
            self.info["episode_score"] = self.compute_episode_score()
        elif self._start_next_round():  # prepare next round only when game has not ended yet
            self.__prepare_next_round()

        info = deepcopy(self.info)
        self.info = {}  # reset info after each step
        return done, info

    def _should_pass_turn(self):
        """
        Whether to pass the turn to the next player. Otherwise, the current player keeps playing based on the context
        set via set_player_context(player, content).
        As every response request entails a single turn, this should return False if the player is to be reprompted.
        """
        return True

    def _next_player(self) -> Player:
        """
        Subclasses can overwrite this method to determine the next player after a player's turn has been passed.

        Default: The gamer master passes the turn to the next player in the player list (order as added).
        Starting again with the first player, when all players have had their turn(s).

        :return: the next (current) player
        """
        self._current_player_idx = (self._current_player_idx + 1) % len(self.players_by_names)
        return self.get_players()[self._current_player_idx]

    def _start_next_round(self) -> bool:
        """
        Subclasses can overwrite this method to specify when a next round should start after a player's turn is passed.

        Default: Start next round when we cycled through the whole list i.e. it is again the first player's turn.

        :return: True, when to start a new round
        """
        return self._current_player_idx == 0

    def __prepare_next_round(self):
        self.log_next_round()  # add record entry for player turns
        self._on_before_round()

    def get_turn_feedback(self):
        """Optional textual feedback to be fed back to model (for playpen RL).
        :return: a verbal feedback about the player's response given the context
        """
        return None

    @abc.abstractmethod
    def compute_turn_score(self):
        """Score response based on last context (for playpen RL)
        :return: the performance score for a player's response given its last context
        """
        pass

    @abc.abstractmethod
    def compute_episode_score(self):
        """
        :return: the performance of the agent over the whole episode
        """
        pass

    @abc.abstractmethod
    def _advance_game(self, player: Player, parsed_response: str):
        """
        Method executed after a player response has been parsed and validated w.r.t to the communication protocol.

        Checks if a player response is applicable (w.r.t game state) and valid (w.r.t. game rules).

        Implements effects that an applicable player's response has on the game world, that is,
        advancing the game by using the player's response to update the game state.

        For example:
            - set the response as the context for the another player to respond to via set_context_for(other_player, response) and let _should_pass_turn() return True
            - set an adjusted context for the current player and give the current player an additional turn by letting _should_pass_turn() return False

        Args:
            player: The Player instance that produced the response (or has been modified by the GM).
            parsed_response: The response of the current player.
        """
        pass

    @abc.abstractmethod
    def _parse_response(self, player: Player, response: str) -> str:
        """Parse the response based on the communication protocol expected by the game master.
        For example, games might require the player to prefix every response with 'GUESS:'

        Args:
            player: The Player instance that produced the response. Intended to allow for individual handling of
                different players.
            response: The response of the current player.
        Returns:
            The parsed response
        Raises:
            ParseError: If the message format is incorrect or the message cannot be properly parsed by the game master.
        """
        pass

    @abc.abstractmethod
    def _does_game_proceed(self) -> bool:
        """Check if game should proceed.

        Mandatory override.

        This method is used to determine if a game should continue or be stopped. Both successful completion of the game
        and game-ending failures should lead to this method returning False.
        Returns:
            A bool, True if game continues, False if game should stop.
        """
        pass

    def _on_game_error(self, error: GameError):
        """
        Hook to implement consequences for game errors e.g. prepare re-prompting or set game state to failure.
        """
        pass

    def _on_parse_error(self, error: ParseError):
        """
        Hook to implement consequences for parsing errors e.g. prepare re-prompting or set game state to abort.
        """
        pass

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


class EnvGameMaster(GameMaster):
    """Extended GameMaster, integrating a GameEnvironment as self-contained object for state management."""

    def __init__(
            self,
            game_spec: GameSpec,
            experiment: dict,
            player_models: List[backends.Model],
            game_environment: GameEnvironment,
    ):
        """
        Args:
            name: The name of the game (as specified in game_registry).
            path: Path to the game (as specified in game_registry).
            experiment: The experiment (set of instances) to use.
            player_models: Player models to use for one or two players.
            game_environment: The environment that maintains the game state.
        """
        super().__init__(game_spec, experiment, player_models)
        self.game_environment = game_environment

        # set players
        self.players_by_names: Dict[str, Player] = collections.OrderedDict()

        self.current_player: Optional[Player] = None
        self.current_player_idx: int = 0

        self.current_round: int = 0

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
        self.log_player(player)

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
        if self.players_by_names:  # todo: why should this be empty here?
            self.current_player = self.get_players()[self.current_player_idx]

    @abc.abstractmethod
    def _on_setup(self, **kwargs):
        """Method executed at the start of the default setup method.
        Template method: Must be implemented!
        Use add_player() here to add the players.
        Args:
            kwargs: Keyword arguments of the game instance. This is usually a game instance object
                read from the game's instances.json.
        """
        raise NotImplementedError

    def get_environment_state(self):
        """Get the current game state from the environment."""
        return self.game_environment.state

    def get_current_player(self) -> Optional[Player]:
        return self.current_player

    def play(self) -> None:
        """
        Main play loop method. This method is called to run the game for benchmarking.
        This implementation uses the game environment for state management.
        """
        module_logger.debug(
            f"[_play] Starting game with current player: {self.current_player}"
        )
        if self.current_player is None:
            module_logger.warning("No current player set, ending game.")
            return

        while not self.game_environment.state["terminated"]:
            self._on_before_round()

            observation = self.game_environment.get_observation(self.current_player)
            module_logger.info(f"[_play] Player {self.current_player.name}")
            module_logger.info(f"[_play] Observation: \n{to_pretty_json(observation)}")

            response = self.current_player(observation)
            module_logger.info(f"[_play] Response: {response}")

            # TODO: now that we have _validate_action in the game_environment, do we still need this?
            if not self._validate_player_response(self.current_player, response):
                module_logger.warning(
                    f"[_play] Player {self.current_player.name} response is invalid"
                )
                terminated = self._should_terminate_on_invalid_response()
                if terminated:
                    self._on_after_game()
                    break

            action = self.parse_action_from_response(response)

            module_logger.debug(f"[_play] Action: {action}")
            self.game_environment.step(self.current_player, action)

            if self.game_environment.state["terminated"]:
                self._on_after_game()
                break

            if self._should_pass_turn():
                self.current_player = self._next_player()
                if self._start_next_round():
                    self._on_after_round()
                    self.current_round += 1
                    self.log_next_round()

    def _next_player(self) -> Player:
        """
        Subclasses can overwrite this method to determine the next player after a player's turn has been passed.

        Default: The gamer master passes the turn to the next player in the player list (order as added).
        Starting again with the first player, when all players have had their turn(s).

        :return: the next (current) player
        """
        players = self.get_players()
        if not players:
            raise ValueError("No players have been added to the game")

        self.current_player_idx = (self.current_player_idx + 1) % len(players)
        return players[self.current_player_idx]

    def _start_next_round(self) -> bool:
        """
        Subclasses can overwrite this method to specify when a next round should start after a player's turn is passed.

        Default: Start next round when we cycled through the whole list i.e. it is again the first player's turn.

        :return: True, when to start a new round
        """
        return self.current_player_idx == 0

    @abc.abstractmethod
    def compute_response_score(self, response: str, context: Dict):
        """
        Mandatory.
        :param response: The response of the current player.
        :param context: The context given to the current player to generate the response for.
        :return: the performance score for a player's response given the context
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_episode_score(self):
        """
        Mandatory.
        :return: the performance of the agent over the whole episode
        """
        raise NotImplementedError

    def _should_pass_turn(self):
        """
        Whether to pass the turn to the next player. Otherwise, the current player keeps playing
        based on the context set via set_player_context(player, content).
        """
        return True

    @abc.abstractmethod
    def _validate_player_response(self, player: Player, response: str) -> bool:
        """
        Decide if a player response is valid. An invalid response breaks the game rules and might end the game.

        Note: If the response is not valid, then _parse_response() and on_valid_player_response() will not be called.

        However, game developers can decide to give the player another turn by letting _should_pass_turn() return False.

        Args:
            player: The player that gave the response.
            response: The response of the current player.
        Returns:
            True, if the response is fine. Otherwise, False.
        """
        raise NotImplementedError

    def parse_action_from_response(self, response: str) -> Action:
        """Create an action from a player's response.

        Default: return action

        Args:
            response: The textual response from the player
            action_type: The type of action to create

        Returns:
            {"action_type": "verbal_response", "message": response}
        """
        return {"action_type": "verbal_response", "message": response}

    def _should_terminate_on_invalid_response(self) -> bool:
        """
        Decide if the game should terminate on an invalid response.

        Default: False
        """
        return False

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

    def _on_before_game(self):
        """Executed once at the start, before entering the play loop.

        Hook: Modify this method for game-specific functionality.
        """
        pass

    def _on_after_game(self):
        """Executed once at the end, after exiting the play loop.

        Hook: Modify this method for game-specific functionality.
        """
        # todo this is supposed to be a hook; users might accidentally overwrite which ignoes the call to _add_logs
        # I also think the DGM is already doing this now (to see an example; shouldn't diverge too much)
        self._add_logs_to_episode_scores()

    def _add_logs_to_episode_scores(self):
        """Executed once at the end, after exiting the play loop.

        Hook: Modify this method for game-specific functionality.

        This method is useful to process and log/record overall game results.
        """
        module_logger.info("[_on_after_game] Game completed, processing final state")

        final_state = self.game_environment.state

        module_logger.debug(f"Final game state: \n{to_pretty_json(final_state)}")

        for key, value in final_state.items():
            self.log_key(key, value)

        self.log_key("episode_score", self.compute_episode_score())

        module_logger.info(f"[_on_after_game] Game completed")
