import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, TypedDict

from clemcore.clemgame.environment import (
    Action,
    ActionSpace,
    GameEnvironment,
    GameState,
    Observation,
)
from clemcore.clemgame.player import Player

logger = logging.getLogger(__name__)

Position = Tuple[int, int]


@dataclass
class Object(ABC):
    """Base class for all objects in the grid environment."""
    position: Position
    name: str
    symbol: str  # char to be shown in the grid

    def __str__(self) -> str:
        return f"{self.name} at {self.position}"

    @abstractmethod
    def can_interact_with(self, other: 'Object') -> bool:
        """Check if this object can interact with another object."""
        pass

    @abstractmethod
    def interact_with(self, other: 'Object') -> None:
        """Interact with another object."""
        pass


class GridCell(TypedDict):
    object: Optional[Object]
    position: Position


Grid = list[list[GridCell]]


class GridState(GameState):
    """Extended game state for grid-based environments.

    Additional fields:
    - grid: The 2D grid of objects
    - player_positions: Dictionary mapping player names to their positions
    - partial_observability: Whether partial observability is enabled
    """
    grid: Grid
    player_positions: Dict[str, Position]
    partial_observability: bool


class GridObservation(Observation):
    """Observation for the grid environment."""
    grid: Grid


class PlayerObject(Object):
    """Represents a player in the grid."""

    def __init__(self, position: Position, player: Player):
        super().__init__(position, f"Player_{player.name}", "player")
        self.player = player

    def can_interact_with(self, other: Object) -> bool:
        return True

    def interact_with(self, other: Object) -> None:
        if other.can_interact_with(self):
            other.interact_with(self)


class GridEnvironment(GameEnvironment):
    """Base class for grid-based game environments."""

    def __init__(
        self,
        width: int,
        height: int,
        partial_observability: bool = False,
        config: Optional[Dict] = None,
        limited_visibility: bool = False
    ):
        """Initialize the grid environment.

        Args:
            width: Width of the grid
            height: Height of the grid
            partial_observability: Whether to enable partial observability
            config: Additional configuration options
            limited_
        """
        super().__init__(config)

        self.width = width
        self.height = height
        self.grid: Grid = [[GridCell(object=None, position=(x, y)) for x in range(width)] for y in range(height)]
        self.limited_visibility = limited_visibility

        self.state: GridState = {
            "grid": self.grid,
            "player_positions": {},
            "partial_observability": partial_observability
        }

    def reset(
        self,
        initial_observations: Optional[Dict[str, GridObservation]] = None,
        initial_action_spaces: Optional[Dict[str, ActionSpace]] = None,
    ):
        """Reset the environment to its initial state."""
        super().reset(initial_observations, initial_action_spaces)

        self.grid = [[GridCell(object=None, position=(x, y)) for x in range(self.width)] for y in range(self.height)]
        self.state["grid"] = self.grid
        self.state["player_positions"] = {}

    def add_object(self, obj: Object) -> None:
        """Add an object to the grid at its position."""
        x, y = obj.position
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x]["object"] = obj
        else:
            raise ValueError(f"Position {obj.position} is out of bounds")

    def remove_object(self, obj: Object) -> None:
        """Remove an object from the grid."""
        x, y = obj.position
        if self.grid[y][x]["object"] == obj:
            self.grid[y][x]["object"] = None

    def get_objects_at(self, position: Position) -> Optional[Object]:
        """Get all objects at a given position."""
        x, y = position
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]["object"]
        return None

    def is_position_valid(self, position: Position) -> bool:
        """Check if a position is within the grid bounds."""
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def get_adjacent_positions(self, position: Position) -> List[Position]:
        """Get all valid adjacent positions."""
        x, y = position
        adjacent = [
            (x + 1, y), (x - 1, y),
            (x, y + 1), (x, y - 1)
        ]
        return [pos for pos in adjacent if self.is_position_valid(pos)]

    def get_observation(self, player: Player) -> GridObservation:
        """Get the current observation for a specific player.

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

    @abstractmethod
    def update_observations(self):
        """Update observations for all players based on their current positions.

        This method is called after each step to ensure all players have up-to-date observations
        based on their current positions in the grid.

        Should use render_state per player.
        """
        raise NotImplementedError

    def render_state(self, player_name: Optional[str] = None) -> str:
        """Format the grid for display as string.

        Args:
            player_name: Optional player name. If provided, uses the explored map of that player
                to render explored vs unexplored cells and marks the player's current position with 'player'.
                If None, shows the entire grid without fog of war.
        """
        grid_str = ""
        player_pos = None
        explored = None
        if player_name is not None:
            player_pos = self.state["player_positions"][player_name]
            explored = self.explored[player_name]

        if self.limited_visibility and player_pos is not None:
            row, col = player_pos
            for i in range(max(0, row - 1), min(self.height, row + 2)):
                row_str = ""
                for j in range(max(0, col - 1), min(self.width, col + 2)):
                    cell = self.state["grid"][i][j]
                    cell_content = "player" if (i, j) == player_pos else (
                        cell["object"].symbol if cell["object"] is not None else "empty"
                    )
                    row_str += f"({i},{j}) is {cell_content}, "
                grid_str += row_str.lstrip() + "\n"
            return grid_str

        for i in range(self.height):
            row_str = ""
            for j in range(self.width):
                cell = self.state["grid"][i][j]
                if explored is not None:
                    if explored[i][j]:
                        cell_content = "player" if (i, j) == player_pos else (
                            cell["object"].symbol if cell["object"] is not None else "empty"
                        )
                    else:
                        cell_content = "❓"
                else:
                    cell_content = "player" if (player_pos is not None and (i, j) == player_pos) else (
                        cell["object"].symbol if cell["object"] is not None else "empty"
                    )
                row_str += f"({i},{j}) is {cell_content}, "
            grid_str += row_str.lstrip() + "\n"
        return grid_str
