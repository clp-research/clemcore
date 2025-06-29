"""
Base class for clembench game environments.

Environments:
- are self-contained systems that manage their own state
- include an action space of actions that can be taken within them to alter their state
- include an observation space of observations that can be made of the state of the environment
- include a termination condition that defines when the environment is finished
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

from clemcore.clemgame.player import Player
from clemcore.utils.string_utils import to_pretty_json

logger = logging.getLogger(__name__)

ActionType = str

ActionSpace = List[ActionType]


class GameState(TypedDict):
    """Base type definition for the game environment's state with required fields.

    Required fields:
    - terminated: Whether the game has terminated
    - success: Whether the game was successful
    - aborted: Whether the game was aborted
    """

    terminated: bool
    success: bool
    aborted: bool
    moves: int
    warning: str
    # add fields for game-specific state on inheritance


class Observation(TypedDict, total=False):
    """Base type definition for the game environment's observation with required fields.

    Required fields:
    - role: The role of the player
    - content: The string content (prompt) that will be sent to the model
    
    Optional fields:
    - image: List of image data (base64 strings or URLs) for multimodal models
    """

    role: Literal["user"]
    content: str
    image: List[str]


class Action(TypedDict):
    """Base type definition for the game environment's action with required fields.

    Required fields:
    - action_type: The type of action
    """

    action_type: ActionType
    # add fields for game-specific action parameters on inheritance, e.g. message for conversational responses


class GameEnvironment(ABC):
    """
    Base class for game environments in Clem.

    This class follows both the Gymnasium interface and the clembench framework.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a game environment.

        Args:
            action_spaces: Dictionary of action spaces, one key per player
            observation_spaces: Dictionary of observation spaces, one key per player
        """
        super().__init__()

        # string keys represent player names
        self.action_spaces: Dict[str, ActionSpace] = {}
        self.observations: Dict[str, Observation] = {}

        self.config = config or {}

        self.state: GameState = {
            "terminated": False,
            "success": False,
            "aborted": False,
            "moves": 0,
            "warning": "",
            # add fields for game-specific state on inheritance
        }

        self.players: List[Player] = []

        self.max_moves = self.config.get("max_moves", None)
        logger.info(f"[_init] Max moves: {self.max_moves}")

    def reset(
        self,
        initial_observations: Optional[Dict[str, Observation]] = None,
        initial_action_spaces: Optional[Dict[str, ActionSpace]] = None,
    ):
        """
        Reset the environment to its initial state.

        Overwrite this in your inheriting class to account for game-specific state.
        """
        self.state = {
            "terminated": False,
            "success": False,
            "aborted": False,
            "moves": 0,
            "warning": "",
            # add fields for game-specific state on inheritance
        }
        if initial_observations is not None:
            self.observations = initial_observations
        if initial_action_spaces is not None:
            self.action_spaces = initial_action_spaces

    def step(self, player: Player, action: Action) -> None:
        """Execute one step in the environment.

        Args:
            player: The player making the action
            action: Action dictionary with:
                - action_type: Type of action (always 'text' for this game)
                - body: The text response from the player
        """
        logger.info(f"[step] Environment step with player: {player.name}")

        # TODO: alternatively, should it check for a bool that is true only if setup was done previously?
        if not self.observations[player.name] or not self.action_spaces[player.name]:
            raise ValueError(
                f"[step] No observation or action space for player: {player.name}"
            )

        self.state["moves"] += 1

        if not self._max_moves_reached():
            if self._is_action_valid(player, action):
                self._update_state_through_action(player, action)
                logger.debug(f"[step] New game state: \n{to_pretty_json(self.state)}")

            self.update_observations()
            logger.debug(
                f"[step] Updated observation for player: {player.name if hasattr(player, 'name') else 'unknown'}"
            )

        if self.state["aborted"]:
            logger.warning(f"[step] Action aborted: {action}")
        elif self.state["success"]:
            logger.info(f"[step] Action successful: {action}")
        else:
            logger.warning(f"[step] Action unsuccessful: {action}")

    def _max_moves_reached(self) -> bool:
        """
        Check if the maximum number of moves has been reached.
        """
        if self.max_moves is not None and self.state["moves"] >= self.max_moves:
            logger.warning(f"[_max_moves_reached] Max moves reached — will abort and terminate")
            self.state["terminated"] = True
            self.state["aborted"] = True
            self.state["success"] = False
            return True
        return False

    def _is_action_valid(self, player: Player, action: Action) -> bool:
        if action.get("action_type") is None:
            raise ValueError(f"[step] No action type in action: {action}")

        if (
            self._action_violates_format(action)
            or self._action_not_in_action_space(player, action)
            or self._action_invalid_in_state(player, action)
        ):
            return False

        return True

    def _action_violates_format(self, action: Action) -> bool:
        """
        Check if an action violates the format.
        """
        if action["action_type"] == "violated_format":
            self.state["terminated"] = False
            self.state["aborted"] = True
            self.state["success"] = False
            self.state["warning"] = "Your response violated the format. Please try again."
            return True
        return False

    def _action_not_in_action_space(self, player: Player, action: Action) -> bool:
        """
        Check if an action is not in the action space.
        """
        if action["action_type"] not in self.action_spaces[player.name]:
            self.state["terminated"] = False
            self.state["aborted"] = True
            self.state["success"] = False
            self.state["warning"] = "You cannot do that. Please try again."
            return True
        return False

    def _action_invalid_in_state(self, player: Player, action: Action) -> bool:
        """
        Check if an action is invalid in the current state.
        """
        is_valid, warning = self._is_action_valid_in_state(player, action)
        if not is_valid:
            self.state["terminated"] = False
            self.state["aborted"] = True
            self.state["success"] = False
            self.state["warning"] = warning
            return True
        return False

    @abstractmethod
    def _update_state_through_action(self, player: Player, action: Action):
        """
        Update the state after an action is taken.

        This method should update state["terminated"], state["success"], state["aborted"], as well as any other game-specific state fields.
        """
        raise NotImplementedError

    @abstractmethod
    def _is_action_valid_in_state(self, player: Player, action: Action) -> Tuple[bool, str]:
        """
        Validate if an action is legal in the current state.

        Overwrite this method in your subclass to implement custom validation logic based on the current state.

        Make sure you set state["warning"] in here if the action is invalid, so that the player can get appropriate feedback.
        """
        raise NotImplementedError

    def add_player(self, player: Player):
        """
        Add a player to the environment.
        """
        self.players.append(player)

    @abstractmethod
    def update_observations(self):
        """
        Set the new observations for all players.

        Make sure you include state["warning"] in the observations if the action is invalid, so that the player can get appropriate feedback.
        """
        raise NotImplementedError

    def get_observation(self, player: Player) -> Observation:
        """
        Get the current observation for a specific player.

        Args:
            player: The player to get the observation for

        Returns:
            The observation for the player
        """
        logger.debug(f"[observe_for] Getting observation for player: {player.name}")

        if player.name not in self.observations:
            logger.warning(
                f"[observe_for] No observation found for player: {player.name}. Creating default."
            )
            raise ValueError(
                f"[observe_for] No observation found for player: {player.name}"
            )

        observation = self.observations[player.name]
        logger.debug(f"[observe_for] Observation for {player.name}: {observation}")
        return observation

    def set_action_space(self, player: Player, action_space: List[Any]):
        """
        Set the action space for a specific player.

        Args:
            player: The player to set the action space for
            action_space: The action space to set
        """
        self.action_spaces[player.name] = action_space

    def pretty_print_state(self) -> str:
        """
        Pretty print the game state.
        """
        raise NotImplementedError
