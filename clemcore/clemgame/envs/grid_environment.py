import io
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TypedDict

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from clemcore.clemgame.envs.environment import (
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
    pretty_symbol: str  # emoji to be shown in the grid on render_state_as_human_readable

    def __str__(self) -> str:
        return f"{self.name} at {self.position}"


class GridCell(TypedDict):
    objects: List[Object]
    position: Position


Grid = list[list[GridCell]]


class GridState(GameState):
    """Extended game state for grid-based environments.

    Additional fields:
    - grid: The 2D grid of objects
    - player_positions: Dictionary mapping player names to their positions
    """
    grid: Grid
    player_positions: Dict[str, Position]


class PlayerObject(Object):
    """Represents a player in the grid."""

    def __init__(self, position: Position, player: Player):
        super().__init__(position, f"Player_{player.name}", "P", "👤")
        self.player = player


class GridEnvironment(GameEnvironment):
    """Base class for grid-based game environments."""

    def __init__(
        self,
        config: Optional[Dict] = None,
    ):
        """Initialize the grid environment.

        Args:
            config: Additional configuration options
        """
        super().__init__(config)

        self.width = config.get("width", 10)
        self.height = config.get("height", 10)
        self.limited_visibility = config.get("limited_visibility", False)
        self.render_as = config.get("render_as", "string")

        self.grid: Grid = [
            [GridCell(objects=[], position=(y, x)) for x in range(self.width)]
            for y in range(self.height)
        ]

        self.state: GridState = {
            "grid": self.grid,
            "player_positions": {},
        }

    def reset(self):
        """Reset the environment to its initial state."""
        super().reset()

        self.grid = [[GridCell(objects=[], position=(y, x)) for x in range(self.width)] for y in range(self.height)]
        self.state["grid"] = self.grid
        self.state["player_positions"] = {}

    def add_object(self, obj: Object) -> None:
        """Add an object to the grid at its position."""
        y, x = obj.position
        if 0 <= x < self.width and 0 <= y < self.height:
            self.state["grid"][y][x]["objects"].append(obj)
        else:
            raise ValueError(f"Position {obj.position} is out of bounds")

    def remove_object(self, obj: Object) -> None:
        """Remove an object from the grid."""
        y, x = obj.position
        if obj in self.state["grid"][y][x]["objects"]:
            self.state["grid"][y][x]["objects"].remove(obj)

    def get_objects_at(self, position: Position) -> List[Object]:
        """Get all objects at a given position."""
        y, x = position
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.state["grid"][y][x]["objects"]
        return []

    def get_observation(self, player: Player) -> Observation:
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

    def _render_state_as_string(self, player_name: Optional[str] = None) -> str:
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

        # render visible area of player if limited visibility is enabled
        if self.limited_visibility and player_pos is not None:
            y, x = player_pos
            for i in range(max(0, y - 1), min(self.height, y + 2)):
                row_str = ""
                for j in range(max(0, x - 1), min(self.width, x + 2)):
                    cell = self.state["grid"][i][j]
                    cell_content = cell["objects"][-1].symbol if cell["objects"] != [] else "empty"
                    row_str += f"({i},{j}) is {cell_content}, "
                grid_str += row_str.lstrip() + "\n"
            return grid_str

        # render full grid
        for i in range(self.height):
            row_str = ""
            for j in range(self.width):
                cell = self.state["grid"][i][j]
                if explored is not None:
                    if explored[i][j]:
                        cell_content = cell["objects"][-1].symbol if cell["objects"] != [] else "empty"
                    else:
                        cell_content = "❓"
                else:
                    cell_content = cell["objects"][-1].symbol if cell["objects"] != [] else "empty"
                row_str += f"({i},{j}) is {cell_content}, "
            grid_str += row_str.lstrip() + "\n"
        return grid_str

    def _render_state_as_image(self, player_name: Optional[str] = None) -> bytes:
        """Format the grid for display as image.

        Args:
            player_name: Optional player name. If provided, uses the explored map of that player
                to render explored vs unexplored cells and marks the player's current position with 'player'.
                If None, shows the entire grid without fog of war.

        Returns:
            Base64-encoded PNG image data
        """
        fig, ax = plt.subplots(figsize=(max(6, self.width * 0.8), max(4, self.height * 0.6)))
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_facecolor('white')

        player_pos = None
        explored = None
        if player_name is not None:
            player_pos = self.state["player_positions"][player_name]
            explored = self.explored[player_name]

        for i in range(self.height):
            for j in range(self.width):
                cell = self.state["grid"][i][j]

                if explored is not None and not explored[i][j]:
                    cell_content = "?"
                    cell_color = 'lightgray'
                else:
                    if cell["objects"]:
                        cell_content = cell["objects"][-1].symbol
                        if isinstance(cell["objects"][-1], PlayerObject):
                            cell_color = 'lightblue'
                        else:
                            cell_color = 'lightgreen'
                    else:
                        cell_content = " "
                        cell_color = 'white'

                rect = patches.Rectangle((j, self.height - 1 - i), 1, 1,
                                         linewidth=1, edgecolor='black',
                                         facecolor=cell_color)
                ax.add_patch(rect)

                ax.text(j + 0.5, self.height - 1 - i + 0.5, cell_content,
                        ha='center', va='center', fontsize=16, fontweight='bold')

        if player_pos is not None:
            row, col = player_pos
            rect = patches.Rectangle((col, self.height - 1 - row), 1, 1,
                                     linewidth=3, edgecolor='red',
                                     facecolor='none')
            ax.add_patch(rect)

        # if limited visibility, darken cells outside visible range
        if self.limited_visibility and player_pos is not None:
            row, col = player_pos
            for i in range(self.height):
                for j in range(self.width):
                    if abs(i - row) > 1 or abs(j - col) > 1:
                        rect = patches.Rectangle((j, self.height - 1 - i), 1, 1,
                                                 linewidth=1, edgecolor='black',
                                                 facecolor='darkgray', alpha=0.7)
                        ax.add_patch(rect)

        plt.tight_layout()

        # convert to base64-encoded PNG
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plt.close(fig)

        return buffer.getvalue()

    def _render_state_as_human_readable(self, player_name: Optional[str] = None) -> str:
        """
        Pretty print the grid state.
        """
        if player_name is not None:
            player_pos = self.state["player_positions"][player_name]
            explored = self.explored[player_name]

        pretty_grid = ""
        for row in self.state["grid"]:
            row_str = ""
            for cell in row:
                row_str += f"{cell['objects'][-1].pretty_symbol if cell['objects'] != [] else '⬜️'}"
            pretty_grid += row_str.lstrip() + "\n"
        return f"{pretty_grid}"
