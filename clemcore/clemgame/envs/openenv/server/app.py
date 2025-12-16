import os
from typing import Dict, Optional

from openenv_core.env_server import create_app

from clemcore.clemgame.envs.openenv.models import ClemGameAction, ClemGameObservation
from clemcore.clemgame.envs.openenv.server.environment import ClemGameEnvironment


def read_query_string(query_string: str) -> Optional[Dict[str, str]]:
    if query_string is None:
        return None
    if not query_string:
        return {}
    kv_pairs = query_string.split(",")
    kv_dict = {}
    for kv_pair in kv_pairs:
        assert "=" in kv_pair, f"Invalid query string: {query_string}"
        k, v = kv_pair.split("=")
        kv_dict[k.strip()] = v.strip()
    return kv_dict


def create_clem_game_app(
        game_name: Optional[str] = None,
        *,
        game_instance_split=None,
        single_pass: bool = False,
        learner_agent: str = "player_0",
        other_agents: Dict[str, str] = "player_1=mock",
):
    game_name = os.getenv("CLEM_GAME", game_name)
    game_instance_split = os.getenv("CLEM_GAME_SPLIT", game_instance_split)
    single_pass = os.getenv("CLEM_GAME_SINGLE_PASS", str(single_pass)).lower() == "true"
    learner_agent = os.getenv("CLEM_GAME_LEARNER_AGENT", "player_0")
    other_agents = read_query_string(os.getenv("CLEM_GAME_OTHER_AGENTS", "player_1=mock"))
    gen_args = read_query_string(os.getenv("CLEM_GAME_GEN_ARGS", "temperature=0.0,max_tokens=300"))
    env = ClemGameEnvironment(game_name,
                              game_instance_split=game_instance_split,
                              single_pass=single_pass,
                              learner_agent=learner_agent,
                              other_agents=other_agents,
                              gen_args=gen_args
                              )
    return create_app(env, ClemGameAction, ClemGameObservation, env_name="clemgame_env")
