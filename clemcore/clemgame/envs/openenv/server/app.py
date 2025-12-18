import os
from typing import Dict, Optional, Any

from openenv_core.env_server import create_app

from clemcore.utils.string_utils import read_query_string
from clemcore.clemgame.envs.openenv.models import ClemGameAction, ClemGameObservation
from clemcore.clemgame.envs.openenv.server.environment import ClemGameEnvironment

CLEMV_GAME = os.getenv("CLEMV_GAME")
CLEMV_GAME_SPLIT = os.getenv("CLEMV_GAME_SPLIT")
CLEMV_SINGLE_PASS = os.getenv("CLEMV_SINGLE_PASS", "False").lower() in ("true", "1", "t")
CLEMV_LEARNER_AGENT = os.getenv("CLEMV_LEARNER_AGENT")
CLEMV_OTHER_AGENTS = read_query_string(os.getenv("CLEMV_OTHER_AGENTS"))
CLEMV_GEN_ARGS = read_query_string(os.getenv("CLEMV_GEN_ARGS"))


def create_clemv_app(
        game_name: str = CLEMV_GAME,
        *,
        learner_agent: str = CLEMV_LEARNER_AGENT,
        other_agents: Optional[Dict[str, str]] = None,
        game_instance_split: str = CLEMV_GAME_SPLIT,
        single_pass: bool = CLEMV_SINGLE_PASS,
        gen_args: Optional[Dict[str, Any]] = None
):
    # Fallback to env vars if not provided as arguments
    other_agents = other_agents if other_agents is not None else CLEMV_OTHER_AGENTS
    gen_args = gen_args if gen_args is not None else CLEMV_GEN_ARGS

    # Validation: Ensure required configuration is present
    config_values = {
        "game_name": game_name,
        "learner_agent": learner_agent
    }
    missing = [k for k, v in config_values.items() if v is None]
    if missing:
        raise ValueError(f"Missing required configuration for: {', '.join(missing)}. "
                         "Provide them as arguments or set the corresponding CLEM_GAME_* env vars.")

    env = ClemGameEnvironment(game_name,
                              game_instance_split=game_instance_split,
                              single_pass=single_pass,
                              learner_agent=learner_agent,
                              other_agents=other_agents,
                              gen_args=gen_args
                              )
    return create_app(env, ClemGameAction, ClemGameObservation, env_name="clem_env")
