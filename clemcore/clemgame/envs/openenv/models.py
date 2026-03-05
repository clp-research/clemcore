from typing import Dict, Optional

from openenv.core import Action, Observation, State


class ClemGameAction(Action):
    response: str


class ClemGameObservation(Observation):
    context: Dict


class ClemGameState(State):
    game_name: Optional[str] = None
    episode_count: int = 0
