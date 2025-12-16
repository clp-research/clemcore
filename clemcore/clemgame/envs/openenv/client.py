from openenv_core.http_env_client import HTTPEnvClient
from openenv_core.client_types import StepResult

from clemcore.clemgame.envs.openenv.models import ClemGameAction, ClemGameObservation, ClemGameState


class ClemGameEnv(HTTPEnvClient[ClemGameAction, ClemGameObservation]):
    def _step_payload(self, action: ClemGameAction) -> dict:
        """Convert action to JSON"""
        return {"response": action.response}

    def _parse_result(self, payload: dict) -> StepResult:
        """Parse JSON to observation"""
        # NOTE: done and reward are part of Observation,
        # but before transmission they are removed from the obs instance
        # and put into the payload directly
        observation = ClemGameObservation(**payload["observation"])
        # Hence Observation:
        #     done: bool = False
        #     reward: Union[bool, int, float, None] = None
        # are never set by the openenv server
        return StepResult(
            observation=observation,  # this is only context
            reward=payload["reward"],
            done=payload["done"]
        )

    def _parse_state(self, payload: dict) -> ClemGameState:
        return ClemGameState(**payload)
