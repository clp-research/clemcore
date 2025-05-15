"""
Base class for all clembench environments.

Environments:
- are self-contained systems that manage their own state
- include an action space of actions that can be taken within them to alter their state
- include an observation space of observations that can be made of the state of the environment
- include a termination condition that defines when the environment is finished
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from clemcore.utils import format_json, setup_logger

from .player import Player

logger = setup_logger(__name__)


class GameState(TypedDict):
    """Base type definition for the game environment's state with required fields.

    Required fields:
    - current_context: The current context/observation for the active player
    - terminated: Whether the game has terminated
    - success: Whether the game was successful
    - aborted: Whether the game was aborted
    """

    current_context: str
    terminated: bool
    success: bool
    aborted: bool


class GameEnvironment(ABC):
    """
    Base class for game environments in Clem.

    This class follows both the Gymnasium interface and the clembench framework.
    """

    # TODO: initial action_spaces and observation_spaces are currently only actually set in the GameMaster._on_setup method — remove those args from __init__?
    def __init__(
        self,
        # action_spaces: Optional[Dict[str, List[Any]]] = None,
        # observation_spaces: Optional[Dict[str, Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a game environment.

        Args:
            action_spaces: Dictionary of action spaces, one key per player
            observation_spaces: Dictionary of observation spaces, one key per player
        """
        super().__init__()
        # logger.info(
        #     f"Initializing game environment with action spaces: {action_spaces} and observation spaces: {observation_spaces}"
        # )

        # self.action_spaces = action_spaces or {}
        # self.observation_spaces = observation_spaces or {}
        self.action_spaces = {}
        self.observation_spaces = {}

        self.config = config or {}

        self.state: GameState = {
            "current_context": "",
            "terminated": False,
            "success": False,
            "aborted": False,
        }

    def reset(self):
        """
        Reset the environment to its initial state.

        Overwrite this in your inheriting class to add functionality.
        """
        self.state = {
            "current_context": "",
            "terminated": False,
            "success": False,
            "aborted": False,
        }

    def step(
        self, player: Player, action: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], bool, bool]:
        """Execute one step in the environment.

        Args:
            player: The player making the action
            action: Action dictionary with:
                - action_type: Type of action (always 'text' for this game)
                - body: The text response from the player

        Returns:
            Tuple of (observation, success, terminated)
        """
        logger.info(f"[step] Environment step with player: {player.name}")
        logger.debug(f"[step] Action: {action}")

        # TODO: alternatively, should it check for a bool that is true only if setup was done previously?
        if (
            not self.observation_spaces[player.name]
            or not self.action_spaces[player.name]
        ):
            raise ValueError(
                f"[step] No observation or action space for player: {player.name}"
            )

        logger.debug("[step] Validating action")
        if not self._validate_action(player, action):
            # TODO: implement an option to continue the game on invalid action?
            raise ValueError(f"[step] Invalid action: {action}")

        self._update_state_through_action(player, action)

        logger.debug(f"[step] New game state: \n{format_json(self.state)}")
        if self.state["aborted"]:
            logger.warning(f"[step] Action aborted")
        elif self.state["success"]:
            logger.info("[step] Action was successful")
        else:
            logger.warning(f"[step] Action was unsuccessful")

        self.set_observation_space(player, self.state["current_context"])
        observation = self.get_observation_space(player)

        logger.debug(f"[step] Observation: \n{format_json(observation)}")

        logger.debug(
            f"[step] Updated observation for player: {player.name if hasattr(player, 'name') else 'unknown'}"
        )

        return observation, self.state["success"], self.state["terminated"]

    def _validate_action(self, player: Player, action: Dict[str, Any]) -> bool:
        """
        Validate if an action is legal in the current state.
        """
        action_type = action["action_type"]
        if action_type not in self.action_spaces[player.name]:
            return False
        if not self._is_action_valid_in_state(player, action_type):
            return False
        return True

    @abstractmethod
    def _update_state_through_action(self, player: Player, action: Dict[str, Any]):
        """
        Update the state after an action is taken.

        This method should update state["current_context"], state["success"], state["aborted"].
        It should also update self.state["terminated"] if the game should terminate.
        """
        raise NotImplementedError

    def _is_action_valid_in_state(self, player: Player, action_type: str) -> bool:
        """
        Validate if an action is legal in the current state.

        Overwrite this method in your subclass to implement custom validation logic based on the current state.
        """
        return True

    def set_observation_space(self, player: Player, content: str):
        """
        Set the observation space for a specific player.

        Args:
            player: The player to set the observation for
            content: The text content for the observation
        """
        observation_space = {"role": "user", "content": content}

        self.observation_spaces[player.name] = observation_space

        logger.info(
            f"[set_observation_space_for] Set observation space for player: {player.name}"
        )

    def get_observation_space(self, player: Player) -> Dict[str, Any]:
        """
        Get the current observation for a specific player.

        Args:
            player: The player to get the observation for

        Returns:
            The observation for the player
        """
        logger.debug(f"[observe_for] Getting observation for player: {player.name}")

        if player.name not in self.observation_spaces:
            logger.warning(
                f"[observe_for] No observation found for player: {player.name}. Creating default."
            )
            raise ValueError(
                f"[observe_for] No observation found for player: {player.name}"
            )

        observation = self.observation_spaces[player.name]
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
