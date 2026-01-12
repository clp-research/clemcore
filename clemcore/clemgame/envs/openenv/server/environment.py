import logging
from dataclasses import asdict
from typing import Dict, Any

from datasets import load_dataset
from openenv_core import Environment

from clemcore.backends import load_models
from clemcore.clemgame.callbacks.base import GameBenchmarkCallbackList
from clemcore.clemgame.envs.openenv.models import ClemGameState, ClemGameObservation, ClemGameAction
from clemcore.clemgame.envs.pettingzoo import gym_env, check_agent_mapping_for_training
from clemcore.clemgame.registry import GameRegistry
from clemcore.clemgame.instances import to_instance_filter

module_logger = logging.getLogger(__name__)


class ClemGameEnvironment(Environment):

    def __init__(self,
                 game_name: str,
                 *,
                 game_instance_split: str = None,
                 single_pass: bool = False,
                 learner_agent: str = "player_0",
                 env_agents: Dict[str, str] = None,
                 gen_args: Dict[str, Any] = None,
                 callbacks: GameBenchmarkCallbackList = None
                 ):
        super().__init__()
        module_logger.info("Initialize ClemGameEnvironment: "
                           "game_name=%s, game_instance_split=%s, learner_agent=%s, env_agents=%s",
                           game_name, game_instance_split, learner_agent, env_agents)

        # Fail quickly if the given agent mapping is not compatible with the game's specifications
        env_agents = env_agents or {}
        game_spec = GameRegistry.from_directories_and_cwd_files().get_game_spec(game_name)
        check_agent_mapping_for_training(game_spec, {learner_agent: "learner", **env_agents})

        game_instance_filter = None  # use all the default game instances of the game
        if game_instance_split:
            # We only use the training instances so that we can properly evaluate on the validation set later
            dataset = load_dataset("colab-potsdam/playpen-data", "instances", split=game_instance_split)
            game_instance_filter = to_instance_filter(dataset)

        # Finally, load the opponent models, which can take a long time for large models
        if game_spec.is_multi_player():  # if single_player env_agents should be None; or check has failed above
            agent_models = load_models(list(env_agents.values()), gen_args)
            env_agents = {
                agent_id: agent_model
                for agent_id, agent_model in zip(env_agents.keys(), agent_models)
            }

        # todo: also allows to provide Player's (not just the model)
        self._game_name = game_name
        self._state = ClemGameState(game_name=game_name, episode_id="episode_0", step_count=0, episode_count=0)
        self._game_env = gym_env(game_name,
                                 game_instance_filter=game_instance_filter,
                                 single_pass=single_pass,
                                 learner_agent=learner_agent,
                                 env_agents=env_agents,
                                 callbacks=callbacks
                                 )

    def close(self):
        module_logger.info(f"Close ClemGameEnvironment {self._state.game_name}")
        self._game_env.close()

    def reset(self) -> ClemGameObservation:
        observation, info = self._game_env.reset()
        self._state.step_count = 0
        self._state.episode_count += 1
        self._state.episode_id = f"episode_{self._state.episode_count}"
        module_logger.info(f"Reset ClemGameEnvironment {self._state.game_name} for episode {self._state.episode_id}")
        return ClemGameObservation(context=observation)

    def step(self, action: ClemGameAction) -> ClemGameObservation:
        if module_logger.isEnabledFor(logging.DEBUG):
            module_logger.debug(f"Step ClemGameEnvironment with action={asdict(action)}")
        observation, reward, done, truncated, info = self._game_env.step(action.response)
        if module_logger.isEnabledFor(logging.DEBUG):
            print(f"Step ClemGameEnvironment result is "
                  f"observation={observation}, reward={reward}, done={done}, "
                  f"truncated={truncated}, info={info}")
        self._state.step_count += 1
        return ClemGameObservation(
            context=observation,
            reward=float(reward),
            done=done,
            metadata={**info, **dict(truncated=truncated)}
        )

    @property
    def state(self) -> ClemGameState:
        if module_logger.isEnabledFor(logging.DEBUG):
            module_logger.debug(f"State for ClemGameEnvironment is {self._state}")
        return self._state
