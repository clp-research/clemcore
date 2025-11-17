import logging
from typing import Dict, List, Literal, Optional, Tuple

from clemcore.clemgame.player import Player

from .grid_environment import GridEnvironment, GridState, Object

logger = logging.getLogger(__name__)

Position = Tuple[int, int]


class InclusiveGridState(GridState):
    """Extended game state for inclusive grid-based environments, including players as objects part of the grid.

    Additional fields:
    - _player_positions: Optional dictionary mapping player names to their positions (only required if players should be positioned in the grid)
    - _explored: Optional dictionary mapping player names to their explored part of the grid (only required if show_explored is True)
    """
    _player_positions: Optional[Dict[str, Position]]
    _explored: Optional[Dict[str, List[List[bool]]]]


class PlayerObject(Object):
    """Optionally represents a player in the grid.

    Attributes:
        position: Initial (row, col) of the player object.
        player: The Player the object represents.
    """

    def __init__(self, position: Position, player: Player):
        super().__init__(position, f"Player_{player.name}", "P", "👤")
        self.player = player


class InclusiveGridEnvironment(GridEnvironment):
    """Grid environment that embeds players as movable Objects and supports visibility masks.

    Adds helper methods for player position/object lookup, movement, exploration tracking,
    and a generic movement legality check usable by many navigation-style games.
    """

    def __init__(
        self,
        config: Dict,
    ):
        """Initialize the inclusive grid environment.

        Args:
            config (Dict): Configuration options
        """
        super().__init__(config)

        self.limited_visibility = config.get("limited_visibility", False)
        self.show_explored = config.get("show_explored", False)

        self.state: InclusiveGridState

    def _initialize_state(self) -> None:
        """Initialize player objects, positions, and explored maps (if enabled)."""
        super()._initialize_state()

        players_start = self.config.get("grid", {}).get("players_start", None)
        self.state["_player_positions"] = {}

        for i, player in enumerate(self.players):
            player_start = players_start[i]
            self._add_object(PlayerObject(position=player_start, player=player))
            self.state["_player_positions"][player.name] = tuple(player_start)

        if self.show_explored:
            self.state["_explored"] = {}
            for player in self.players:
                self.state["_explored"][player.name] = [
                    [False for _ in range(self.width)] for _ in range(self.height)
                ]
                self._mark_explored(player.name, self._get_player_position(player.name))

    def _mark_explored(self, player_name: str, pos: Position) -> None:
        """Mark a 3x3 neighborhood around pos as explored for the player (if enabled).

        Args:
            player_name (str): Player identifier.
            pos: (row, col) center to mark explored.
        """
        row, col = pos
        for i in range(row - 1, row + 2):
            for j in range(col - 1, col + 2):
                if 0 <= i < self.height and 0 <= j < self.width:
                    self.state["_explored"][player_name][i][j] = True

    def _get_player_position(self, player_name: str) -> Position:
        """Return the (row, col) position of a player from state.

        Args:
            player_name (str): Player identifier.

        Returns:
            Position: (row, col) of the player.
        """
        return self.state["_player_positions"][player_name]

    def _get_player_object(self, player_name: str) -> PlayerObject:
        """Return the PlayerObject instance representing the player in the grid.

        Args:
            player_name (str): Player identifier.

        Returns:
            PlayerObject: The object currently placed at the player's position.
        """
        return self._get_objects_at(self._get_player_position(player_name))[-1]

    def _move_player(self, player_name: str, direction: str) -> None:
        """Move a player north/south/east/west and update explored map/position.

        Args:
            player_name (str): Player identifier.
            direction (str): One of {"n", "s", "e", "w"}.
        """
        y, x = self._get_player_position(player_name)
        player_object = self._get_player_object(player_name)
        self._remove_object(player_object)

        if direction == "n":
            player_object.position = (y - 1, x)
        elif direction == "s":
            player_object.position = (y + 1, x)
        elif direction == "e":
            player_object.position = (y, x + 1)
        elif direction == "w":
            player_object.position = (y, x - 1)

        self._add_object(player_object)

        if self.show_explored:
            self._mark_explored(player_name, player_object.position)

        self.state["_player_positions"][player_name] = player_object.position

    def _action_valid_in_state(
        self, player: Player, direction: Literal["n", "s", "e", "w"]
    ) -> Tuple[bool, str]:
        """
        Basic movement legality check against grid bounds.

        Subclasses can extend this to include obstacles, doors, hazards, etc.

        Args:
            player (Player): The player attempting the move.
            direction (str): One of {"n", "s", "e", "w"}.

        Returns:
            Tuple[bool, str]: (is_valid, warning_message) where warning_message explains rejection.
        """
        y, x = self._get_player_position(player.name)

        if direction == "n":
            new_pos = (y - 1, x)
        elif direction == "s":
            new_pos = (y + 1, x)
        elif direction == "e":
            new_pos = (y, x + 1)
        elif direction == "w":
            new_pos = (y, x - 1)
        else:
            return False, f"Invalid direction: {direction}! Please try again."

        new_y, new_x = new_pos
        # check if the new position is within the grid
        if not (0 <= new_y < self.height and 0 <= new_x < self.width):
            return (
                False,
                f"The cell ({new_y}, {new_x}) is outside the grid! Please try again.",
            )

        return True, ""

    def _visible_grid(self, player_name: Optional[str]) -> List[List[bool]]:
        """
        Compute a visibility mask given limited_visibility/show_explored and player location.

        Args:
            player_name (Optional[str]): Player identifier whose perspective to render (or None for global view).

        Returns:
            List[List[bool]]: Matrix marking visible (True) vs hidden (False) cells.
        """
        if not self.limited_visibility:
            # full grid
            return [[True for _ in range(self.width)] for _ in range(self.height)]

        if self.show_explored:
            # explored map
            return self.state["_explored"].get(player_name)

        # limited visibility and no explored map: local 3x3 around player
        mask = [[False for _ in range(self.width)] for _ in range(self.height)]
        pos = self._get_player_position(player_name)
        if pos is not None:
            y, x = pos
            for i in range(max(0, y - 1), min(self.height, y + 2)):
                for j in range(max(0, x - 1), min(self.width, x + 2)):
                    mask[i][j] = True
            return mask

        # fallback: show full grid
        return [[True for _ in range(self.width)] for _ in range(self.height)]

    def _render_state_as_string(self, player_name: Optional[str] = None,
                                mask: Optional[List[List[bool]]] = None
                                ) -> str:
        """Format the grid for display as string with player-aware masking.

        Args:
            player_name (Optional[str]): Optional player name. If provided, uses the explored map of that player
                to render explored vs unexplored cells and marks the player's current position with 'player'.
                If None, shows the entire grid without fog of war.
            mask (Optional[List[List[bool]]]): Optional precomputed visibility mask.

        Returns:
            str: Compact textual representation of the visible grid.
        """
        if mask is None:
            mask = self._visible_grid(player_name)

        return super()._render_state_as_string(player_name, mask)

    def _render_state_as_image(self, player_name: Optional[str] = None,
                               mask: Optional[List[List[bool]]] = None
                               ) -> bytes:
        """Format the grid for display as image.

        Args:
            player_name (Optional[str]): Optional player name. If provided, uses the explored map of that player
                to render explored vs unexplored cells and marks the player's current position with 'player'.
                If None, shows the entire grid without fog of war.
            mask (Optional[List[List[bool]]]): Optional precomputed visibility mask.

        Returns:
            bytes: Base64-encoded PNG image data
        """
        if mask is None:
            mask = self._visible_grid(player_name)

        return super()._render_state_as_image(player_name, mask)

    def _render_state_as_human_readable(self, player_name: Optional[str] = None,
                                        mask: Optional[List[List[bool]]] = None
                                        ) -> str:
        """
        Pretty-print the grid state with player-aware masking.

        Args:
            player_name (Optional[str]): Optional player name. If provided, uses the explored map of that player
                to render explored vs unexplored cells and marks the player's current position with 'player'.
                If None, shows the entire grid without fog of war.
            mask (Optional[List[List[bool]]]): Optional precomputed visibility mask.

        Returns:
            str: Human-readable grid representation.
        """
        if mask is None:
            mask = self._visible_grid(player_name)

        return super()._render_state_as_human_readable(player_name, mask)
