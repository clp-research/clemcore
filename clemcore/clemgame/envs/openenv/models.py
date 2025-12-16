from dataclasses import dataclass
from typing import Dict, Union, Any, Optional

from openenv_core.env_server import Action, Observation, State
from clemcore.clemgame import Player


@dataclass
class ClemGameAction(Action):
    response: str


@dataclass
class ClemGameObservation(Observation):
    context: Dict


@dataclass
class ClemGameState(State):
    game_name: Optional[str] = None
