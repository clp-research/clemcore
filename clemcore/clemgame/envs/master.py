import abc
import collections
import logging
from typing import List, Dict, Tuple, Optional

from clemcore import backends
from clemcore.clemgame.envs.environment import Action, GameEnvironment

from clemcore.clemgame.master import GameMaster
from clemcore.clemgame.metrics import (
    METRIC_ABORTED,
    METRIC_LOSE,
    METRIC_SUCCESS
)
from clemcore.clemgame.player import Player
from clemcore.clemgame.registry import GameSpec

module_logger = logging.getLogger(__name__)


class EnvGameMaster(GameMaster):
    """Extended GameMaster, integrating a GameEnvironment as self-contained object for state management."""

    def __init__(
            self,
            game_spec: GameSpec,
            experiment: dict,
            player_models: List[backends.Model],
            game_environment: Optional[GameEnvironment] = None,
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
        if game_environment is not None:
            self.game_environment = game_environment

        # set players
        self.players_by_names: Dict[str, Player] = collections.OrderedDict()

        self.current_player: Optional[Player] = None
        self.current_player_idx: int = 0

        self.current_round: int = 0

    def __setstate__(self, state):
        self.__dict__.update(state)
        for player in self.players_by_names.values():
            player.register_many(self._loggers)

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
        player.register_many(self._loggers)
        player.name = f"Player {len(self.players_by_names) + 1}"
        if player.name in self.players_by_names:
            raise ValueError(
                f"Player names must be unique, "
                f"but there is already a player registered with name '{player.name}'."
            )
        self.players_by_names[player.name] = player
        self.log_player(player.name, player.game_role, player.model.name)

        self.game_environment.add_player(player)

    def _next_player(self) -> Player:
        """
        Subclasses can overwrite this method to determine the next player after a player's turn has been passed.

        Default: The gamer master passes the turn to the next player in the player list (order as added).
        Starting again with the first player, when all players have had their turn(s).

        :return: the next (current) player
        """
        self.current_player_idx = (self.current_player_idx + 1) % len(
            self.players_by_names
        )
        return self.get_players()[self.current_player_idx]

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
        self._on_before_game()

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

    def observe(self) -> Tuple[Player, Dict]:
        """
        Returns the current player and their observation from the environment.
        """
        if self.current_player is None:
            raise RuntimeError("No current player set in EnvGameMaster.")
        observation = self.game_environment.get_observation(self.current_player)
        return self.current_player, observation

    def step(self, response: str) -> Tuple[bool, Dict]:
        """
        Applies the player's response as an action in the environment, advances the game, and returns (done, info).
        """
        info = {}  # mostly empty for now

        if not self._player_response_in_expected_format(self.current_player, response):
            if self._should_terminate_on_invalid_response():
                self._on_after_game()
                self.log_game_end()
                self._end_game()
                return True, info
            action = self._violated_format_action()
        else:
            action = self._create_action_from_response(response)

        self.game_environment.step(self.current_player, action)
        if self.game_environment.state["aborted"]:
            self.count_request_violation()
        self.log_to_self("state", self.game_environment.state_to_log())

        if self.is_done():
            self._on_after_game()
            self.log_game_end()
            self._end_game()
            return True, info

        if self._should_pass_turn():
            self.current_player = self._next_player()

        if self._start_next_round():
            self._on_after_round()
            self.current_round += 1
            self.log_next_round()
            self._on_before_round()

        return False, info

    def is_done(self) -> bool:
        """
        Returns True if the game is finished (terminated in the environment).
        """
        return self.game_environment.state.get("terminated", False)

    def has_started(self) -> bool:
        """
        Returns True if the game has started (current_player is set and environment is not in initial state).
        """
        return self.current_player is not None and self.game_environment.state is not None

    def _start_next_round(self) -> bool:
        """
        Subclasses can overwrite this method to specify when a next round should start after a player's turn is passed.

        Default: Start next round when we cycled through the whole list i.e. it is again the first player's turn.

        :return: True, when to start a new round
        """
        return self.current_player_idx == 0

    def _should_pass_turn(self):
        """
        Whether to pass the turn to the next player. Otherwise, the current player keeps playing
        based on the context set via set_player_context(player, content).
        """
        return True

    @abc.abstractmethod
    def _player_response_in_expected_format(self, player: Player, response: str) -> bool:
        """
        Decide if a player response is valid. An invalid response breaks the game rules. In this case, depending on _should_terminate_on_invalid_response(), the game might be terminated.

        Args:
            player: The player that gave the response.
            response: The response of the current player.
        Returns:
            True, if the response is fine. Otherwise, False.
        """
        raise NotImplementedError

    def _create_action_from_response(self, response: str) -> Action:
        """
        Create an action from a player's response.
        """
        try:
            return self._parse_action_from_response(response)
        except Exception as e:
            module_logger.warning(f"[_get_action] Error parsing action from response: {e}")
            return self._violated_format_action()

    def _violated_format_action(self) -> Action:
        """
        Create an action that represents a response that violates the format.
        """
        return {"action_type": "violated_format"}

    @abc.abstractmethod
    def _parse_action_from_response(self, response: str) -> Action:
        """Create an action from a player's response.

        Args:
            response: The textual response from the player

        Returns:
            An action dictionary with:
                - action_type: The type of action
                - body: The text response from the player
        """
        raise NotImplementedError

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
        """Executed once at the start, at the start of the play loop.

        Hook: Modify this method for game-specific functionality.
        """
        pass

    def _end_game(self):
        """
        Finishes the game by adding the episode scores to the logs and calling the after game hook.
        """
        final_state = self.game_environment.state

        aborted = int(final_state.get("aborted", False))
        success = int(final_state.get("success", False))
        lose = int(not success and not aborted)

        self.log_key(METRIC_ABORTED, aborted)
        self.log_key(METRIC_SUCCESS, success)
        self.log_key(METRIC_LOSE, lose)

        for player in self.get_players():
            player.reset()

    def _on_after_game(self):
        """Executed once at the end, at the end of the play loop.

        Hook: Modify this method for game-specific functionality.
        """
        pass
