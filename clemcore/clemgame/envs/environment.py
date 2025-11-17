"""
Base class for clembench game environments.

Environments:
- are self-contained systems that manage their own state
- include an action space of actions that can be taken within them to alter their state
- include an observation space of observations that can be made of the state of the environment
- include a termination condition that defines when the environment is finished
"""

import base64
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union

from clemcore.clemgame.player import Player
from clemcore.utils.string_utils import to_pretty_json

module_logger = logging.getLogger(__name__)

ActionType = str

ActionSpace = List[ActionType]


class GameState(TypedDict):
    """Base type definition for the game environment's state with required fields.

    Keys not starting with '_' are considered public.
    Keys starting with '_' are considered private and will be omitted by the
        default info() implementation.

    Required fields:
    - terminated (bool): Whether the game has terminated
    - success (bool): Whether the game was successful
    - aborted (bool): Whether the game was aborted
    - moves (int): The number of moves made in the game
    - _warning (str): A warning message to be sent to the player
    """
    terminated: bool
    success: bool
    aborted: bool
    moves: int
    _warning: str
    # add fields for game-specific state on inheritance


class Observation(TypedDict):
    """Base type definition for the game environment's observation with required fields.

    Required fields:
    - role (Literal["user"]): The role of the player
    - content (str): The string content (prompt) that will be sent to the model

    Optional fields:
    - image (List[str]): List of image paths
    """
    role: Literal["user"]
    content: str
    image: List[str]


class Action(TypedDict):
    """Base type definition for the game environment's action with required fields.

    Required fields:
    - action_type (ActionType): The type of action
    """

    action_type: ActionType
    # add fields for game-specific action parameters on inheritance, e.g. message for conversational responses


class GameEnvironment(ABC):
    """Base class for turn-based game environments.

    - Owns and mutates the game state on each step
    - Validates actions and computes rewards
    - Produces per-player observations (text and/or image)
    - Exposes a public "info" snapshot each turn for logging/scoring

    Designed to be compatible with clembench and inspired by Gymnasium's reset/step API.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize a game environment.

        Args:
            config (Dict[str, Any]): Per-episode configuration. Recognized keys include:
                - render_as: One of {"string", "image", "human-readable"}
                - max_moves: Optional integer cap after which the episode terminates as aborted
        """
        super().__init__()

        # string keys represent player names
        self.action_spaces: Dict[str, ActionSpace] = {}
        self.observations: Dict[str, Observation] = {}

        self.config = config
        self.render_as = self.config.get("render_as", "string")
        self.max_moves = self.config.get("max_moves", None)

        self.players: List[Player] = []

        self.state: GameState

    def reset(self):
        """Reset the environment to its initial state.

        Subclasses need to override _initialize_state() and possibly set action spaces.
        """
        self.state = {
            "terminated": False,
            "success": False,
            "aborted": False,
            "moves": 0,
            "_warning": "",
            # add fields for game-specific state on inheritance in _initialize_state()
        }

        self.observations = {}
        self.action_spaces = {}

        self._initialize_state()

        for player in self.players:
            action_space = self._action_space_for(player)
            self.action_spaces[player.name] = action_space

        self._update_observations()

    def observe(self, player: Player) -> Observation:
        """Get the current observation for a specific player.

        Args:
            player (Player): The player to get the observation for

        Returns:
            Observation: The observation for the player
        """
        observation = self.observations[player.name]
        return observation

    def step(self, player: Player, action: Action) -> Tuple[float, bool, bool, Dict]:
        """Execute one step in the environment.

        Args:
            player (Player): The player making the action.
            action (Action): Action dictionary with at least the key "action_type" and any game-specific fields.

        Returns:
            Tuple[float, bool, bool, Dict]:
                - reward (float): Turn-level scalar reward (default: 0 if aborted else 1)
                - terminated (bool): True if the episode reached a terminal state
                - aborted (bool): True if the step was rejected or max_moves was reached
                - info (dict): Public snapshot of the environment state
        """
        module_logger.info(f"[step] Environment step with player: {player.name}")

        self.state["terminated"] = False
        self.state["success"] = False
        self.state["aborted"] = False

        self.state["_warning"] = ""

        self.state["moves"] += 1

        if self._action_valid(player, action):
            self._update_state_through_action(player, action)
            self.state["terminated"], self.state["success"] = self._check_won(player)
            module_logger.debug(f"[step] New game state: \n{to_pretty_json(self.state)}")
        else:
            self.state["aborted"] = True
            module_logger.warning(f"[step] Action invalid: {action}")

        self._update_observations()

        reward = self.reward()

        if self._max_moves_reached():
            if not self.state["terminated"]:
                self.state["terminated"] = True
                self.state["success"] = False
                self.state["aborted"] = True

        info = self.info()

        return reward, self.state["terminated"], self.state["aborted"], info

    def _max_moves_reached(self) -> bool:
        """Check if the maximum number of moves has been reached.

        Returns:
            bool: True if max_moves is configured and the threshold has been reached; otherwise False.
        """
        if self.max_moves is not None and self.state["moves"] >= self.max_moves:
            return True
        return False

    def _action_valid(self, player: Player, action: Action) -> bool:
        """Validate action format, membership in the player's action space, and game-state legality.

        Args:
            player (Player): The player attempting the action.
            action (Action): The structured action to validate.

        Returns:
            bool: True if the action passes all checks; False otherwise. On failure, state["_warning"] is set.
        """
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
        """Check if an action violates the expected format.

        Args:
            action (Action): The action to inspect.

        Returns:
            bool: True if the action represents a format violation; otherwise False.
        """
        if action["action_type"] == "violated_format":
            self.state["_warning"] = "Your response violated the format. Please try again."
            return True
        return False

    def _action_not_in_action_space(self, player: Player, action: Action) -> bool:
        """Check whether an action's type is permitted for the given player.

        Args:
            player (Player): The player attempting the action.
            action (Action): The action whose type to verify against the player's action space.

        Returns:
            bool: True if the action type is not present in the player's action space; otherwise False.
        """
        if action["action_type"] not in self.action_spaces[player.name]:
            self.state["_warning"] = "You cannot do that. Please try again."
            return True
        return False

    def _action_invalid_in_state(self, player: Player, action: Action) -> bool:
        """Check if an action is illegal given the current environment state.

        Args:
            player (Player): The player attempting the action.
            action (Action): The action to validate against the current state.

        Returns:
            bool: True if invalid (and sets state["_warning"]); otherwise False.
        """
        is_valid, warning = self._action_valid_in_state(player, action)
        if not is_valid:
            self.state["_warning"] = warning
            return True
        return False

    @abstractmethod
    def _update_state_through_action(self, player: Player, action: Action):
        """Update the environment state after a valid action is taken.

        Args:
            player (Player): The player who took the action.
            action (Action): The validated action to apply.
        """
        raise NotImplementedError

    @abstractmethod
    def _check_won(self, player: Player) -> Tuple[bool, bool]:
        """Check the state of the game, and return a tuple of (terminated, success).

        If the game is not yet won but the action was legal, return (False, True).
        If the game is won, return (True, True).
        If the game is lost, return (True, False).

        Args:
            player (Player): The player whose last action may have changed the outcome.

        Returns:
            Tuple[bool, bool]: Tuple of (terminated, success).
        """
        raise NotImplementedError

    @abstractmethod
    def _action_valid_in_state(self, player: Player, action: Action) -> Tuple[bool, str]:
        """Validate if an action is legal in the current state.

        Implement this method in your subclass for custom validation logic based on the current state.
        Make sure you return a warning message in here if the action is invalid, which will be sent to the player as feedback.

        Args:
            player (Player): The player attempting the action.
            action (Action): The action to validate.

        Returns:
            Tuple[bool, str]: Tuple of (is_valid, warning_message). If invalid, warning_message should explain the issue.
        """
        raise NotImplementedError

    def add_player(self, player: Player):
        """Add a player to the environment.

        Args:
            player (Player): The player to add.
        """
        self.players.append(player)

    def _update_observations(self):
        """Default observation update procedure.

        Iterates players, renders state, composes a prompt via _compose_prompt,
        creates an observation and assigns it.
        """
        for player in self.players:
            rendered_state = self._render_state(player.name)
            prompt = self._compose_prompt(player.name)
            observation = self._create_observation(prompt, rendered_state)
            self.observations[player.name] = observation

    def _compose_prompt(self, player_name: Optional[str] = None) -> str:
        """Compose the textual prompt for a player's observation."""
        lines: List[str] = []
        warning = self.state.get("_warning", "")
        if warning:
            lines.append(f"Warning: {warning}\n\n")
        turn_prompt = self._compose_turn_prompt(player_name)
        lines.append(turn_prompt + "\n\n")
        return "".join(lines)

    @abstractmethod
    def _compose_turn_prompt(self, player_name: Optional[str] = None) -> str:
        """Compose the turn prompt for a player.

        Args:
            player_name (Optional[str]): Optional player name.
        """
        raise NotImplementedError

    @abstractmethod
    def _initialize_state(self) -> None:
        """Hook for subclasses to initialize game-specific state keys.

        Called by reset() after universal state is cleared; should not modify
        observations or action spaces. Use this to populate additional fields
        in self.state or prepare internal caches.
        """
        raise NotImplementedError

    def _action_space_for(self, player: Player) -> List[str]:
        """Return the action space for a given player.

        Subclasses may override to provide per-player action spaces. If an empty
        list (or other falsy value) is returned, no action space will be set.
        """
        return ["default"]

    def _render_state(self, player_name: Optional[str] = None) -> Union[str, bytes]:
        """Format the state for display as string or image.

        Args:
            player_name (Optional[str]): Optional player name. If provided, uses the state of that player
                to render the state.
                If None, shows the entire state.

        Returns:
            Union[str, bytes]: Either a string representation of the grid (if render_as is "string"),
                or image data as bytes (if render_as is "image")
                or a pretty-printed string representation of the grid (if render_as is "human-readable")
        """
        if self.render_as == "image":
            render = self._render_state_as_image(player_name)
        elif self.render_as == "string":
            render = self._render_state_as_string(player_name)
        elif self.render_as == "human-readable":
            render = self._render_state_as_human_readable(player_name)
        else:
            raise ValueError(f"Invalid render_as value: {self.render_as}")

        return render

    @abstractmethod
    def _render_state_as_string(self, player_name: Optional[str] = None) -> str:
        """Format the state for display as a compact string.

        Args:
            player_name (Optional[str]): Optional player name for player-relative rendering (if applicable).

        Returns:
            str: Representation of the environment state suitable for LLM consumption.
        """
        raise NotImplementedError

    @abstractmethod
    def _render_state_as_image(self, player_name: Optional[str] = None) -> bytes:
        """Format the state for display as an image.

        Args:
            player_name (Optional[str]): Optional player name for player-relative rendering (if applicable).

        Returns:
            bytes: PNG image bytes. Encoding to base64 data URLs is handled by _create_observation.
        """
        raise NotImplementedError

    @abstractmethod
    def _render_state_as_human_readable(self, player_name: Optional[str] = None) -> str:
        """Format the state for display as a human-friendly string.

        Args:
            player_name (Optional[str]): Optional player name for player-relative rendering (if applicable).

        Returns:
            str: Prettified representation of the environment state intended for transcripts and debugging.
        """
        raise NotImplementedError

    def _create_observation(self, text_content: str, rendered_state: Union[str, bytes]) -> Observation:
        """Create an observation payload from text and a rendered state.

        Args:
            text_content (str): Prompt text to present to the player.
            rendered_state (Union[str, bytes]): Either a string (for render_as in {"string", "human-readable"})
                or PNG bytes (for render_as == "image").

        Returns:
            Observation: Dictionary with role/content and optional base64-encoded image data.
        """
        if self.render_as == "image":
            encoded_image = base64.b64encode(rendered_state).decode('utf-8')
            data = f"data:image/png;base64,{encoded_image}"

            observation: Observation = {
                "role": "user",
                "content": text_content + "[State image shown below]",
                "image": [data],
            }
        else:
            observation: Observation = {
                "role": "user",
                "content": text_content + rendered_state,
            }

        return observation

    def reward(self) -> float:
        """Calculate the reward for the most recent step.

        Overwrite this method in your subclass to implement custom reward logic.

        Returns:
            float: Reward for the most recent step.
        """
        success = self.state["success"]
        return 1 if success else 0

    def info(self) -> Dict[str, Any]:
        """Return a dictionary with the current public state of the environment.

        By default, all state keys that do NOT start with '_' are considered public and will be exported.

        Subclasses can override to add computed values or expose additional/private fields as needed.

        Returns:
            Dict[str, Any]: Dictionary of public state keys to their current values.
        """
        return {key: value for key, value in self.state.items() if not str(key).startswith("_")}
