import os
from typing import Dict, Optional, Any

from openenv_core.env_server import create_app

from clemcore.utils.string_utils import read_query_string
from clemcore.clemgame.callbacks import episode_results_folder_callbacks
from clemcore.clemgame.callbacks.base import GameBenchmarkCallbackList
from clemcore.clemgame.envs.openenv.models import ClemGameAction, ClemGameObservation
from clemcore.clemgame.envs.openenv.server.environment import ClemGameEnvironment

CLEMV_GAME = os.getenv("CLEMV_GAME")
CLEMV_GAME_SPLIT = os.getenv("CLEMV_GAME_SPLIT")
CLEMV_SINGLE_PASS = os.getenv("CLEMV_SINGLE_PASS", "False").lower() in ("true", "1", "t")
CLEMV_LEARNER_AGENT = os.getenv("CLEMV_LEARNER_AGENT")
CLEMV_ENV_AGENTS = read_query_string(os.getenv("CLEMV_ENV_AGENTS"))
CLEMV_GEN_ARGS = read_query_string(os.getenv("CLEMV_GEN_ARGS"))
CLEMV_RESULTS_DIR = os.getenv("CLEMV_RESULTS_DIR", "openenv-records")
CLEMV_RUN_ID = os.getenv("CLEMV_RUN_ID")


def create_clemv_app(
        game_name: str = CLEMV_GAME,
        *,
        learner_agent: str = CLEMV_LEARNER_AGENT,
        env_agents: Optional[Dict[str, str]] = None,
        game_instance_split: str = CLEMV_GAME_SPLIT,
        single_pass: bool = CLEMV_SINGLE_PASS,
        gen_args: Optional[Dict[str, Any]] = None,
        callbacks: Optional[GameBenchmarkCallbackList] = None,
        results_dir: Optional[str] = None,
        run_id: Optional[str] = None
):
    # Fallback to env vars if not provided as arguments
    env_agents = env_agents if env_agents is not None else CLEMV_ENV_AGENTS
    gen_args = gen_args if gen_args is not None else CLEMV_GEN_ARGS
    results_dir = results_dir if results_dir is not None else CLEMV_RESULTS_DIR
    run_id = run_id if run_id is not None else CLEMV_RUN_ID

    # Validation: Ensure required configuration is present
    config_values = {
        "game_name": game_name,
        "learner_agent": learner_agent
    }
    missing = [k for k, v in config_values.items() if v is None]
    if missing:
        raise ValueError(f"Missing required configuration for: {', '.join(missing)}. "
                         "Provide them as arguments or set the corresponding CLEM_GAME_* env vars.")

    # Create default episode recording callbacks if results_dir is specified
    if callbacks is None and results_dir is not None:
        # Derive run_id from env_agent model names if not explicitly provided
        if run_id is None and env_agents:
            run_id = "-".join(env_agents.values())  # todo: can we use Model.to_identifier here?
        callbacks = episode_results_folder_callbacks(
            run_dir=run_id or "run",
            result_dir_path=results_dir
        )

    env = ClemGameEnvironment(game_name,
                              game_instance_split=game_instance_split,
                              single_pass=single_pass,
                              learner_agent=learner_agent,
                              env_agents=env_agents,
                              gen_args=gen_args,
                              callbacks=callbacks
                              )
    return create_app(env, ClemGameAction, ClemGameObservation, env_name="clem_env")
