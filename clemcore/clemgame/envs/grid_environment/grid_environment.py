import io
import logging
from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TypedDict

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from ..environment import GameEnvironment, GameState

logger = logging.getLogger(__name__)

Position = Tuple[int, int]


@dataclass
class Object(ABC):
    """Base class for all objects in the grid environment.

    Attributes:
        position: (row, col) position in the grid.
        name: Unique or descriptive object name.
        symbol: Short ASCII symbol for compact renders.
        pretty_symbol: Emoji or human-friendly symbol for pretty renders.
    """
    position: Position
    name: str
    symbol: str  # char to be shown in the grid
    pretty_symbol: str  # emoji to be shown in the grid on _render_state_as_human_readable

    def __str__(self) -> str:
        return f"{self.name} at {self.position}"


class GridCell(TypedDict):
    objects: List[Object]
    position: Position


Grid = list[list[GridCell]]


class GridState(GameState):
    """Extended game state for grid-based environments.

    Additional fields:
    - _grid: The 2D grid of objects
    """
    _grid: Grid


class GridEnvironment(GameEnvironment):
    """Base class for grid-based game environments.

    Provides a 2D grid with typed cells that can contain stacked Objects and rendering helpers.
    Subclasses implement game-specific legality, transitions, and observation content.
    """

    def __init__(
        self,
        config: Dict,
    ):
        """Initialize the grid environment.

        Args:
            config (Dict): Additional configuration options
        """
        super().__init__(config)

        self.width = config.get("width", 10)
        self.height = config.get("height", 10)

        self.state: GridState

    def _initialize_state(self) -> None:
        """Initialize/clear the grid for a new episode.

        If you need additional initialization, override this method in your subclass.
        Make sure to call super()._initialize_state() to ensure the base class initialization is performed.
        """
        self.state["_grid"] = [
            [GridCell(objects=[], position=(y, x)) for x in range(self.width)]
            for y in range(self.height)
        ]

    def _add_object(self, obj: Object):
        """Add an object to the grid cell at obj.position (top of stack).

        Args:
            obj (Object): The object to insert.
        """
        y, x = obj.position
        if 0 <= x < self.width and 0 <= y < self.height:
            self.state["_grid"][y][x]["objects"].append(obj)
        else:
            raise ValueError(f"Position {obj.position} is out of bounds")

    def _remove_object(self, obj: Object):
        """Remove an object instance from its current grid cell, if present.

        Args:
            obj (Object): The object to remove.
        """
        y, x = obj.position
        if obj in self.state["_grid"][y][x]["objects"]:
            self.state["_grid"][y][x]["objects"].remove(obj)

    def _get_objects_at(self, position: Position) -> List[Object]:
        """Return the stack of objects at a position; empty list if out of bounds or empty.

        Args:
            position (Position): (row, col) position to query.

        Returns:
            List[Object]: Top-to-bottom stack at the given cell (may be empty).
        """
        y, x = position
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.state["_grid"][y][x]["objects"]
        return []

    def _render_state_as_string(self,
                                player_name: Optional[str] = None,
                                mask: Optional[List[List[bool]]] = None
                                ) -> str:
        """Format the grid for display as a compact string, optionally masked for visibility.

        Args:
            player_name (Optional[str]): Optional player name. If provided, uses the explored map of that player
                to render explored vs unexplored cells and marks the player's current position with 'player'.
                If None, shows the entire grid without fog of war.
            mask (Optional[List[List[bool]]]): Optional precomputed visibility mask.

        Returns:
            str: Compact textual representation of the visible grid.
        """
        grid_str = ""

        for i in range(self.height):
            row_str = ""
            for j in range(self.width):
                cell = self.state["_grid"][i][j]
                if mask is not None and not mask[i][j]:
                    cell_content = "?"
                else:
                    if cell["objects"]:
                        cell_content = cell["objects"][-1].symbol
                    else:
                        cell_content = "empty"
                row_str += f"({i},{j}) is {cell_content}, "
            if row_str:
                grid_str += row_str.lstrip() + "\n"
        return grid_str

    def _render_state_as_image(self,
                               player_name: Optional[str] = None,
                               mask: Optional[List[List[bool]]] = None
                               ) -> bytes:
        """Format the grid for display as image.

        Args:
            player_name (Optional[str]): Optional player name. If provided, uses the explored map of that player
                to render explored vs unexplored cells and marks the player's current position with 'player'.
                If None, shows the entire grid without fog of war.

        Returns:
            bytes: Base64-encoded PNG image data
        """
        fig, ax = plt.subplots(figsize=(max(6, self.width * 0.8), max(4, self.height * 0.6)))
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_facecolor('white')

        for i in range(self.height):
            for j in range(self.width):
                cell = self.state["_grid"][i][j]
                if mask is not None and not mask[i][j]:
                    cell_content = "?"
                    cell_color = 'lightgray'
                else:
                    if cell["objects"]:
                        cell_content = cell["objects"][-1].symbol
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

        plt.tight_layout()

        # convert to base64-encoded PNG
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plt.close(fig)

        return buffer.getvalue()

    def _render_state_as_human_readable(self, player_name: Optional[str] = None,
                                        mask: Optional[List[List[bool]]] = None
                                        ) -> str:
        """
        Pretty-print the grid state using emoji symbols, optionally masked.

        Args:
            player_name (Optional[str]): Optional player name. If provided, uses the explored map of that player
                to render explored vs unexplored cells and marks the player's current position with 'player'.
                If None, shows the entire grid without fog of war.
            mask (Optional[List[List[bool]]]): Optional precomputed visibility mask.

        Returns:
            str: Human-readable grid representation.
        """
        pretty_grid = ""

        for i in range(self.height):
            row_str = ""
            for j in range(self.width):
                cell = self.state["_grid"][i][j]
                if mask is not None and not mask[i][j]:
                    cell_pretty = "❓"
                else:
                    if cell["objects"]:
                        cell_pretty = cell['objects'][-1].pretty_symbol
                    else:
                        cell_pretty = '⬜️'
                row_str += f"{cell_pretty}"
            if row_str:
                pretty_grid += row_str.lstrip() + "\n"
        return f"{pretty_grid}"
