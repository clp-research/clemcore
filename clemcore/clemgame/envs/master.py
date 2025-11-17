import abc
import collections
import logging
from typing import Dict, List, Optional, Tuple

from clemcore import backends
from clemcore.clemgame.envs.environment import Action, GameEnvironment
from clemcore.clemgame.master import GameMaster
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_LOSE, METRIC_SUCCESS
from clemcore.clemgame.player import Player
from clemcore.clemgame.registry import GameSpec

module_logger = logging.getLogger(__name__)


class EnvGameMaster(GameMaster):
    """
    Game master orchestrating players interacting with a GameEnvironment.

    Responsibilities:
    - Manage player order and rounds
    - Validate/parse player responses into environment actions
    - Call environment.observe()/step() and log reward/state
    - Emit episode-level metrics on termination
    """

    def __init__(
            self,
            game_spec: GameSpec,
            experiment: dict,
            player_models: List[backends.Model],
    ):
        """Construct the game master.

        Args:
            game_spec (GameSpec): The game specification from the registry.
            experiment (dict): The experiment (set of instances) to run.
            player_models (List[backends.Model]): Backend model adapters for one or more players.
        """
        super().__init__(game_spec, experiment, player_models)

        self.players_by_names: Dict[str, Player] = collections.OrderedDict()
        self.current_player: Optional[Player] = None
        self.current_player_idx: int = 0

        self.current_round: int = 0

        self.game_environment: GameEnvironment

    def __setstate__(self, state):
        """Restore state after unpickling.

        Args:
            state (dict): The serialized __dict__ to restore.
        """
        self.__dict__.update(state)
        for player in self.players_by_names.values():
            player.register_many(self._loggers)

    def get_players(self) -> List[Player]:
        """Get a list of the registered players in order.

        Returns:
            List[Player]: Players in the order they were added.
        """
        return list(self.players_by_names.values())

    def add_player(self, player: Player):
        """Register a player with the master and environment.

        Players act in the order they are added. Names must be unique.

        Args:
            player (Player): Player instance to add.
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
        """Subclasses can overwrite this method to determine the next player after a player's turn has been passed.

        Default: The gamer master passes the turn to the next player in the player list (order as added).
        Starting again with the first player, when all players have had their turn(s).

        Returns:
            Player: The next (current) player.
        """
        self.current_player_idx = (self.current_player_idx + 1) % len(
            self.players_by_names
        )
        return self.get_players()[self.current_player_idx]

    def setup(self, **kwargs):
        """Prepare the game for a specific game instance.

        Calls the subclass hook _on_setup(**kwargs), sets the initial current player,
        and triggers the before-game hook.

        Args:
            kwargs (dict): Instance configuration (from instances.json).
        """
        self._on_setup(**kwargs)
        self.current_player = self.get_players()[self.current_player_idx]
        self._on_before_game()

    @abc.abstractmethod
    def _on_setup(self, **kwargs):
        """Method executed at the start of the default setup method.

        Use add_player() here to add the players.
        Instantiate, add, and reset the game environment here.

        Args:
            kwargs (dict): Keyword arguments of the game instance.
        """
        raise NotImplementedError

    def observe(self) -> Tuple[Player, Dict]:
        """Return the current player and that player's observation.

        Returns:
            Tuple[Player, Dict]: (current_player, observation)
        """
        observation = self.game_environment.observe(self.current_player)
        return self.current_player, observation

    def step(self, response: str) -> Tuple[bool, Dict]:
        """Apply the player's textual response as an action, advance the environment, and return (terminated, info).

        Args:
            response (str): The raw textual response from the current player.

        Returns:
            Tuple[bool, Dict]: (terminated, info) where info is the environment's public state snapshot.
        """
        state = {}

        if not self._response_valid(self.current_player, response):
            action = self._violated_format_action()
        else:
            action = self._parse_action_from_response(response)

        reward, terminated, aborted, state = self.game_environment.step(self.current_player, action)

        self.log_to_self("state", state)
        self.log_to_self("reward", reward)

        if aborted and not terminated:
            self.count_request_violation()

        if terminated:
            self._on_after_game()
            self.log_game_end()
            self._end_game()
            return terminated, state

        if self._should_pass_turn(aborted):
            self.current_player = self._next_player()

        if self._start_next_round():
            self._on_after_round()
            self.current_round += 1
            self.log_next_round()
            self._on_before_round()

        return terminated, state

    def is_done(self) -> bool:
        """True if the environment's state indicates termination.

        Returns:
            bool: Whether the episode has terminated.
        """
        return self.game_environment.state.get("terminated", False)

    def has_started(self) -> bool:
        """True if a current player is set and the environment is initialized.

        Returns:
            bool: Whether the game has started.
        """
        return self.current_player is not None and self.game_environment.state is not None

    def _start_next_round(self) -> bool:
        """Decide whether to start a new round after passing the turn.

        Default: Start next round when we cycle back to the first player.

        Returns:
            bool: True if a new round should begin.
        """
        return self.current_player_idx == 0

    def _should_pass_turn(self, aborted: bool):
        """Decide whether to pass the turn to the next player after this step.

        Default: keep the player if action was aborted; otherwise pass.

        Args:
            aborted (bool): Whether the last step was aborted by the environment.

        Returns:
            bool: True to pass the turn; False to keep the current player.
        """
        if aborted:
            return False
        return True

    @abc.abstractmethod
    def _response_valid(self, player: Player, response: str) -> bool:
        """Validate the textual response format before parsing to an action.

        Subclasses should implement lightweight format checks (e.g., tokens in range),
        not game-state legality (which is handled in the environment).

        Args:
            player (Player): The player that gave the response.
            response (str): The raw response string.

        Returns:
            bool: True if the response matches the expected format; otherwise False.
        """
        raise NotImplementedError

    def _violated_format_action(self) -> Action:
        """Build a synthetic action representing a format violation in the response.

        Returns:
            Action: Action dict with action_type="violated_format".
        """
        return {"action_type": "violated_format"}

    @abc.abstractmethod
    def _parse_action_from_response(self, response: str) -> Action:
        """Parse a textual response into a structured action dict.

        Args:
            response (str): The textual response from the player.

        Returns:
            Action: Dictionary including at least "action_type" and any game-specific fields.
        """
        raise NotImplementedError

    def _on_before_round(self):
        """Hook executed before a new round starts."""
        pass

    def _on_after_round(self):
        """Hook executed after a round finishes (when _start_next_round() becomes True)."""
        pass

    def _on_before_game(self):
        """Hook executed once before the first turn."""
        pass

    def _end_game(self):
        """Finalize the episode: log standard metrics and reset players."""
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
        """Hook executed once after the episode ends."""
        pass
