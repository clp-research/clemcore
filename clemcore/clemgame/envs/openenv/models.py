from dataclasses import dataclass
from typing import Dict, Optional

from openenv.core import Action, Observation, State


@dataclass
class ClemGameAction(Action):
    response: str


@dataclass
class ClemGameObservation(Observation):
    context: Dict


@dataclass
class ClemGameState(State):
    game_name: Optional[str] = None
    episode_count: int = 0
