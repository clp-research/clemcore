"""
Base class for clembench game environments.

Environments:
- are self-contained systems that manage their own state
- include an action space of actions that can be taken within them to alter their state
- include an observation space of observations that can be made of the state of the environment
- include a termination condition that defines when the environment is finished
"""

import logging
import os
import base64
import imghdr
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union

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
    moves: int
    warning: str
    # add fields for game-specific state on inheritance


class Observation(TypedDict):
    """Base type definition for the game environment's observation with required fields.

    Required fields:
    - role: The role of the player
    - content: The string content (prompt) that will be sent to the model

    Optional fields:
    - image: List of image paths
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

    def __init__(self, config: Dict[str, Any] = None):
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

        self.config = config
        self.render_as = self.config.get("render_as", "string")
        self.max_moves = self.config.get("max_moves", None)
            
        self.image_counter = 0

        self.state: GameState = {
            "terminated": False,
            "success": False,
            "aborted": False,
            "moves": 0,
            "warning": "",
            # add fields for game-specific state on inheritance
        }

        self.players: List[Player] = []

    def reset(self):
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

        self.observations = {}
        self.action_spaces = {}

        self.image_counter = 0

    def step(self, player: Player, action: Action) -> None:
        """Execute one step in the environment.

        Args:
            player: The player making the action
            action: Action dictionary with:
                - action_type: Type of action
                - body: The text response from the player
        """
        module_logger.info(f"[step] Environment step with player: {player.name}")

        self.state["aborted"] = False
        self.state["terminated"] = False
        self.state["success"] = False

        self.state["moves"] += 1

        if self._max_moves_reached():
            self.state["terminated"] = True
            self.state["aborted"] = True
            return

        if self._is_action_valid(player, action):
            self._update_state_through_action(player, action)
            self.state["terminated"], self.state["success"] = self.check_won(player)
            module_logger.debug(f"[step] New game state: \n{to_pretty_json(self.state)}")
        else:
            self.state["aborted"] = True
            module_logger.warning(f"[step] Action invalid: {action}")

        self.update_observations()

    def _max_moves_reached(self) -> bool:
        """
        Check if the maximum number of moves has been reached.
        """
        if self.max_moves is not None and self.state["moves"] >= self.max_moves:
            module_logger.warning(f"[_max_moves_reached] Max moves reached — will abort and terminate")
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
            self.state["warning"] = "Your response violated the format. Please try again."
            return True
        return False

    def _action_not_in_action_space(self, player: Player, action: Action) -> bool:
        """
        Check if an action is not in the action space.
        """
        if action["action_type"] not in self.action_spaces[player.name]:
            self.state["warning"] = "You cannot do that. Please try again."
            return True
        return False

    def _action_invalid_in_state(self, player: Player, action: Action) -> bool:
        """
        Check if an action is invalid in the current state.
        """
        is_valid, warning = self._is_action_valid_in_state(player, action)
        if not is_valid:
            self.state["warning"] = warning
            return True
        return False

    @abstractmethod
    def _update_state_through_action(self, player: Player, action: Action):
        """
        Update the state after an action is taken.
        """
        raise NotImplementedError

    @abstractmethod
    def check_won(self, player: Player) -> Tuple[bool, bool]:
        """
        Check the state of the game, and return a tuple of (terminated, success).

        If the game is not yet won but the action was legal, return (False, True).
        If the game is won, return (True, True).
        If the game is lost, return (True, False).
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

    def render_state(self, player_name: Optional[str] = None) -> Union[str, bytes]:
        """Format the state for display as string or image.

        Args:
            player_name: Optional player name. If provided, uses the state of that player
                to render the state.
                If None, shows the entire state.

        Returns:
            Either a string representation of the grid (if render_as is "string"),
            or image data as bytes (if render_as is "image")
            or a pretty-printed string representation of the grid (if render_as is "human-readable")
        """
        if self.render_as == "image":
            image_data = self._render_state_as_image(player_name)
            self.image_counter += 1
            return image_data
        elif self.render_as == "string":
            return self._render_state_as_string(player_name)
        elif self.render_as == "human-readable":
            return self._render_state_as_human_readable(player_name)
        else:
            raise ValueError(f"Invalid render_as value: {self.render_as}")

    @abstractmethod
    def _render_state_as_string(self, player_name: Optional[str] = None) -> str:
        """Format the state for display as string.
        """
        raise NotImplementedError

    @abstractmethod
    def _render_state_as_image(self, player_name: Optional[str] = None) -> bytes:
        """Format the state for display as image.
        """
        raise NotImplementedError

    @abstractmethod
    def _render_state_as_human_readable(self, player_name: Optional[str] = None) -> str:
        """Format the state for display as human-readable string.
        """
        raise NotImplementedError

    def _create_observation(self, text_content: str, rendered_state: Union[str, bytes]) -> Observation:
        """
        Create an observation for a specific player.
        """
        if self.render_as == "image":
            encoded_image = base64.b64encode(rendered_state).decode('utf-8')
            data_url = f"data:image/png;base64,{encoded_image}"

            observation: Observation = {
                "role": "user",
                "content": text_content + "[State image shown below]",
                "image": [data_url],
            }
        else:
            observation: Observation = {
                "role": "user",
                "content": text_content + rendered_state,
            }

        return observation

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

    def state_to_log(self):
        """
        Log relevant game-specific state of the environment to the game master.

        This method will be called after each step in the environment.

        It should log the current state of the environment to the game master.

        This method is optional. Overwrite it if you need state variables for later computation of scores.

        Example:
            logs = {
                "player_positions": self.state["player_positions"],
                "grid": self._render_state_as_human_readable(),
                "terminated": self.state["terminated"],
                "success": self.state["success"],
                "aborted": self.state["aborted"],
            }
            return logs
        """
        pass
