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
from typing import Any, Dict, List, Literal, Optional, TypedDict

from clemcore.clemgame.player import Player
from clemcore.utils.string_utils import to_pretty_json

module_logger = logging.getLogger(__name__)

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
    # add fields for game-specific state on inheritance


class Observation(TypedDict):
    """Base type definition for the game environment's observation with required fields.

    Required fields:
    - role: The role of the player
    - content: The string content (prompt) that will be sent to the model
    """

    role: Literal["user"]
    content: str


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

    def __init__(self, config: Optional[Dict[str, Any]] = None, ):
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
            # add fields for game-specific state on inheritance
        }

    def reset(self,
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
        module_logger.info(f"[step] Environment step with player: {player.name}")

        # TODO: alternatively, should it check for a bool that is true only if setup was done previously?
        if not self.observations[player.name] or not self.action_spaces[player.name]:
            raise ValueError(
                f"[step] No observation or action space for player: {player.name}"
            )

        self._update_state_through_action(player, action)

        module_logger.debug(f"[step] New game state: \n{to_pretty_json(self.state)}")
        if self.state["aborted"]:
            module_logger.warning(f"[step] Action aborted: {action}")
        elif self.state["success"]:
            module_logger.info(f"[step] Action was successful: {action}")
        else:
            module_logger.warning(f"[step] Action was unsuccessful: {action}")

        self.update_observation(player)

        module_logger.debug(
            f"[step] Updated observation for player: {player.name if hasattr(player, 'name') else 'unknown'}"
        )

    def _validate_action(self, player: Player, action: Action) -> bool:
        """
        Validate if an action is legal in the current state.
        """
        action_type = action["action_type"]
        if action_type not in self.action_spaces[player.name]:
            return False
        if not self._is_action_valid_in_state(player, action_type):
            return False
        return True

    def _update_state_through_action(self, player: Player, action: Action):
        """
        Update the state after an action is taken.

        This method should update state["terminated"], state["success"], state["aborted"], as well as any other game-specific state fields.
        """
        module_logger.debug("[_update_state_through_action] Validating action")
        if not self._validate_action(player, action):
            raise ValueError(f"[step] Invalid action: {action}")
        self._do_update_state(player, action)

    @abstractmethod
    def _do_update_state(self, player: Player, action: Action):
        """Subclasses must implement this method to perform the actual state update."""
        raise NotImplementedError

    def _is_action_valid_in_state(self, player: Player, action_type: str) -> bool:
        """
        Validate if an action is legal in the current state.

        Overwrite this method in your subclass to implement custom validation logic based on the current state.
        """
        return True

    def update_observation(self, player: Player):
        """
        Set the observation for a specific player.

        Args:
            player: The player to set the observation for
        """
        observation: Observation = {"role": "user", "content": self.state}

        self.observations[player.name] = observation

        module_logger.info(
            f"[update_observation] Updated observation for player: {player.name}"
        )

    def get_observation(self, player: Player) -> Observation:
        """
        Get the current observation for a specific player.

        Args:
            player: The player to get the observation for

        Returns:
            The observation for the player
        """
        module_logger.debug(f"[observe_for] Getting observation for player: {player.name}")

        if player.name not in self.observations:
            module_logger.warning(
                f"[observe_for] No observation found for player: {player.name}. Creating default."
            )
            raise ValueError(
                f"[observe_for] No observation found for player: {player.name}"
            )

        observation = self.observations[player.name]
        module_logger.debug(f"[observe_for] Observation for {player.name}: {observation}")
        return observation

    def set_action_space(self, player: Player, action_space: List[Any]):
        """
        Set the action space for a specific player.

        Args:
            player: The player to set the action space for
            action_space: The action space to set
        """
        self.action_spaces[player.name] = action_space
