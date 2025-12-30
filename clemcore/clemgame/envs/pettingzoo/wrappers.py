import collections
import logging
from typing import Callable, Union, Literal, Dict, Any, SupportsFloat, Tuple
import gymnasium

from clemcore.backends.model_registry import Model, CustomResponseModel, ModelSpec
from clemcore.clemgame.registry import GameSpec
from clemcore.clemgame.instances import GameInstanceIterator
from clemcore.clemgame.benchmark import GameBenchmark

from gymnasium.core import ActType, ObsType
from pettingzoo.utils.env import ActionType, AgentID
from pettingzoo import AECEnv
from pettingzoo.utils import BaseWrapper

stdout_logger = logging.getLogger("clemcore.run")


class AECToGymWrapper(gymnasium.Env):

    def __init__(self, env: AECEnv):
        self.env = env

        # Get the learner agent (assumes exactly one)
        if hasattr(env, 'learner_agent'):
            # Should be set by SinglePlayerWrapper
            self.learner_agent = env.learner_agent
        elif hasattr(env, 'learner_agents') and len(env.learner_agents) == 1:
            # Should be set by AgentControlWrapper
            self.learner_agent = next(iter(env.learner_agents))
        else:
            raise ValueError(
                "AECToGymWrapper requires an env with exactly one learner agent. "
                "Wrap with SinglePlayerWrapper first."
            )

        # Set up Gym spaces from the learner's perspective
        self.observation_space = env.observation_space(self.learner_agent)
        self.action_space = env.action_space(self.learner_agent)

        # Track current episode state
        self._last_obs = None
        self._cumulative_reward = 0.0

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore
        """Reset environment and return learner's first observation."""
        self._last_obs = None
        self._cumulative_reward = 0.0
        # reset already steps through when AutoControlWrapper is used
        self.env.reset(seed, options)
        # reset stops at the learner's turn, so this is its first observation
        obs, reward, done, truncated, info = self.env.last()
        self._last_obs = dict(obs=obs, reward=reward, done=done, truncated=truncated, info=info)
        return obs, info

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute learner's action, iterate through and return the next observation."""
        self.env.step(action)
        # step stops at the learner's turn, so this is its next observation'
        obs, reward, done, truncated, info = self.env.last()
        self._last_obs = dict(obs=obs, reward=reward, done=done, truncated=truncated, info=info)
        self._cumulative_reward += reward
        return obs, reward, done, truncated, info

    def render(self):
        """Delegate rendering to wrapped env."""
        return self.env.render()

    def close(self):
        """Close the wrapped environment."""
        self.env.close()


def order_agent_mapping_by_agent_id(agent_mapping: Dict[AgentID, Any]):
    """Returns the given agent mappings sorted by agent id.

    For example, an order in keys like player_0, player_1, ...
    """

    def agent_key(entry: Tuple[AgentID, Any]):
        agent_id = entry[0]
        agent_number = agent_id.split('_')[1]
        return int(agent_number)

    return collections.OrderedDict(sorted(agent_mapping.items(), key=agent_key))


class AgentControlWrapper(BaseWrapper):
    """
    This wrapper allows configuring mixed control settings:
    Learner agents remain externally controlled, but other agents are stepped automatically internally.

    Specifically, agents marked as "learner" return control to the caller.
    Other agents are automated with provided models or players.

    Note: When there is no "learner" in the agent mapping,
    then the control will only be given back to the caller when the episode ended.
    This behavior can be useful for full simulations or evaluation runs without a learner.
    """

    def __init__(
            self,
            env: AECEnv,
            agent_mapping: Dict[AgentID, Union[Literal["learner"], Model]]
    ):
        super().__init__(env)
        self.agent_mapping = order_agent_mapping_by_agent_id(agent_mapping)
        self.learner_agents = [agent_id for agent_id, agent in agent_mapping.items() if agent == "learner"]

    def reset(self, seed: int | None = None, options: dict | None = None):
        options = options or {}
        # Augment reset options with player_models but don't overwrite if the caller provided it
        if "player_models" not in options:
            player_models = []
            # assume an order in keys like player_0, player_1, ...
            for agent in self.agent_mapping.values():
                if agent == "learner":
                    player_models.append(CustomResponseModel(ModelSpec(model_name="learner")))
                else:
                    player_models.append(agent)
            options["player_models"] = player_models
        super().reset(seed, options)
        """ If the learner is only on later turns, simulate the interaction up to that turn."""
        self.auto_step()

    def step(self, action: ActionType) -> None:
        # Execute the provided action
        super().step(action)
        self.auto_step()

    def auto_step(self):
        """Automatically play automated agents until the next learner's turn."""
        for agent_id in self.env.agent_iter():
            if agent_id in self.learner_agents:
                return
            obs, reward, done, truncated, info = self.env.last()
            if done or truncated:  # Episode ended before reaching the learner
                return  # caller will observe done for learner b.c. env sets done for all players
            # use the player from the game_env
            # todo: add option to use a passed player here directly
            player = self.unwrapped.player_by_agent_id[agent_id]
            auto_action = player(obs)
            super().step(auto_action)


class SinglePlayerWrapper(AgentControlWrapper):
    """
    This wrapper exposes all game environments as single-agent RL environments.

    This means that any other player than "learner" is automatically controlled by the provided other agents.
    """

    def __init__(
            self,
            env: AECEnv,
            learner_agent: AgentID = "player_0",
            env_agents: Dict[AgentID, Model] = None
    ):
        env_agents = env_agents or {}  # single-player game anyway
        super().__init__(env, {learner_agent: "learner", **env_agents})

        if "learner" in env_agents:
            raise ValueError(
                f"SinglePlayerWrapper requires exactly 1 learner, "
                f"but got env_agents={list(env_agents.keys())}"
            )

        self.learner_agent = learner_agent


class GameBenchmarkWrapper(BaseWrapper):
    """
    A wrapper that loads a GameBenchmark from a GameSpec and passes it to the wrapped environment.
    """

    def __init__(self, env_class: Callable[[GameBenchmark], AECEnv], game_spec: GameSpec, **env_kwargs):
        self.game_benchmark = GameBenchmark.load_from_spec(game_spec)
        super().__init__(env_class(self.game_benchmark, **env_kwargs))

    def close(self) -> None:
        super().close()
        self.game_benchmark.close()


class GameInstanceIteratorWrapper(BaseWrapper):
    """
    A wrapper that iterates through a GameInstanceIterator, either once or infinitely.

    Args:
        wrapped_env: A pettingzoo AECEnv instance.
        game_iterator: An instance of GameInstanceIterator pre-loaded with instances.
        single_pass: If True, the iterator stops after passed once through all instances (e.g., for evaluation).
                     If False (default), the iterator cycles infinitely (e.g., for RL training).
    """

    def __init__(self, wrapped_env: AECEnv, game_iterator: GameInstanceIterator, single_pass: bool = False):
        super().__init__(wrapped_env)
        self.game_iterator = game_iterator.__deepcopy__()
        self.game_iterator.reset(verbose=True)
        if not single_pass:
            stdout_logger.info("Detected single_pass=False, cycling through instances infinitely.")
            from itertools import cycle
            self.game_iterator = cycle(self.game_iterator)
        else:
            stdout_logger.info("Detected single_pass=True, stopping after first pass through all instances.")

    def reset(self, seed: int | None = None, options: dict | None = None):
        experiment, game_instance = next(self.game_iterator)
        options = options or {}
        options["experiment"] = experiment
        options["game_instance"] = game_instance
        super().reset(seed=seed, options=options)
