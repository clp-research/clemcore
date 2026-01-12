from typing import Dict

from pettingzoo.utils.env import AgentID

from clemcore.clemgame.registry import GameSpec
from clemcore.clemgame.envs.pettingzoo.wrappers import (
    AECToGymWrapper,
    SinglePlayerWrapper,
    AgentControlWrapper,
    GameBenchmarkWrapper,
    GameInstanceIteratorWrapper,
    EnvAgent,
)
from clemcore.clemgame.envs.pettingzoo.master import (
    GameMasterEnv,
    env,
    gym_env
)

__all__ = [
    "GameMasterEnv",
    "AECToGymWrapper",
    "SinglePlayerWrapper",
    "AgentControlWrapper",
    "GameBenchmarkWrapper",
    "GameInstanceIteratorWrapper",
    "EnvAgent",
    "env",
    "gym_env",
    "check_agent_mapping_for_training",
    "check_agent_mapping"
]


def check_agent_mapping(game_spec: GameSpec, agent_mapping: Dict[AgentID, str]):
    assert game_spec is not None, "Game spec must be provided for check."
    assert agent_mapping is not None, "Agent mapping must be provided for check."
    if game_spec.is_single_player():
        if len(agent_mapping) > 1:
            raise ValueError(
                f"For single-player game {game_spec.game_name} number of agents cannot be {len(agent_mapping)}")
    # Check that there are as many player_n in the mapping as there are players in the game spec
    missing_players = []
    for player_idx in range(game_spec.players):
        player_name = f"player_{player_idx}"
        if player_name not in agent_mapping:
            missing_players.append(player_name)
    assert len(missing_players) == 0, f"Missing players in agent mapping: {missing_players}"
    # Note: Any additional player in the mapping will simply be ignored during gameplay


def check_agent_mapping_for_training(game_spec: GameSpec, agent_mapping: Dict[AgentID, str]):
    check_agent_mapping(game_spec, agent_mapping)
    # For training, we need a learner in any case. However, we allow multiple learner markers for multi-player games.
    if "learner" not in list(agent_mapping.values()):
        # Note: The general check already assures that player_0=learner for single-player games
        raise ValueError(f"For training, the agent mapping must contain a 'learner' agent")
