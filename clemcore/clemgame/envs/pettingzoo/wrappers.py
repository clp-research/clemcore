from typing import Callable, Union, Literal, Dict, Any, SupportsFloat

import gymnasium
from gymnasium.core import ActType, ObsType
from pettingzoo.utils.env import ActionType, AgentID

from clemcore.backends import Model, CustomResponseModel, ModelSpec
from clemcore.clemgame import GameInstanceIterator, GameSpec, GameBenchmark
from pettingzoo import AECEnv
from pettingzoo.utils import BaseWrapper


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
        """Execute learner's action, iterate through and return next observation."""
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


class AgentControlWrapper(BaseWrapper):
    """
    Flexible wrapper for mixed control: some agents automated, some externally controlled.

    Agents marked as "learner" return control to the caller.
    Other agents are automated with provided models.

    Note: The models for the other agents must be
    """

    def __init__(
            self,
            env: AECEnv,
            agent_mapping: Dict[AgentID, Union[Literal["learner"], Model]]
    ):
        super().__init__(env)
        self.agent_mapping = agent_mapping

        self.learner_agents = {
            agent_id for agent_id, player in agent_mapping.items()
            if player == "learner"
        }
        self.automated_agents = {
            agent_id: player
            for agent_id, player in agent_mapping.items()
            if player != "learner"
        }

        if not self.learner_agents:
            raise ValueError("At least one agent must be marked as 'learner'")

    def reset(self, seed: int | None = None, options: dict | None = None):
        options = options or {}
        # Augment reset options with player_models but don't overwrite if the caller provided it
        if "player_models" not in options:
            player_models = []
            for agent_id in self.env.possible_agents:
                if agent_id in self.learner_agents:
                    player_models.append(CustomResponseModel(ModelSpec(model_name="learner")))
                else:
                    player_models.append(self.automated_agents[agent_id])
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
            player = self.unwrapped.player_by_agent_id[agent_id]
            auto_action = player(obs)
            super().step(auto_action)


class SinglePlayerWrapper(AgentControlWrapper):
    """
    Restricted wrapper: exactly one learner agent, all others automated.

    Simpler API for the common single-agent RL case.
    """

    def __init__(
            self,
            env: AECEnv,
            learner_agent: AgentID = "player_0",
            other_agents: Dict[AgentID, Model] = None
    ):
        # todo: implement logic when wrapping a single player game with this sinlge player wrapper
        # in this case the other agents can be null and the learner agent must be player_0
        agent_mapping = {learner_agent: "learner", **other_agents}
        super().__init__(env, agent_mapping)

        if len(self.learner_agents) != 1:
            raise ValueError(
                f"SinglePlayerWrapper requires exactly 1 learner, "
                f"got {len(self.learner_agents)}"
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
        self.game_iterator.reset()
        self.options = {}
        if not single_pass:
            from itertools import cycle
            self.game_iterator = cycle(self.game_iterator)

    def reset(self, seed: int | None = None, options: dict | None = None):
        experiment, game_instance = next(self.game_iterator)
        self.options = options or {}
        self.options["experiment"] = experiment
        self.options["game_instance"] = game_instance
        super().reset(seed=seed, options=options)
