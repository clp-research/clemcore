import collections
import logging
from typing import Callable, Literal, Any, SupportsFloat, TypeAlias
import gymnasium

from clemcore.backends.model_registry import Model, CustomResponseModel, ModelSpec
from clemcore.clemgame import GameBenchmarkCallbackList
from clemcore.clemgame.registry import GameSpec
from clemcore.clemgame.instances import GameInstanceIterator
from clemcore.clemgame.benchmark import GameBenchmark

from gymnasium.core import ActType, ObsType
from pettingzoo.utils.env import ActionType, AgentID
from pettingzoo import AECEnv
from pettingzoo.utils import BaseWrapper

stdout_logger = logging.getLogger("clemcore.run")

# Type alias for env agents: either a Model (to create a Player) or a Callable (to use directly as policy)
EnvAgent: TypeAlias = Model | Callable[[ObsType], ActionType]


class AECToGymWrapper(gymnasium.Env):

    def __init__(self, env: "AgentControlWrapper"):
        self.env = env
        if hasattr(env, 'learner_agent'):
            self.learner_agent = env.learner_agent  # Should be set by SinglePlayerWrapper
        elif hasattr(env, 'learner_agents') and len(env.learner_agents) == 1:
            self.learner_agent = next(iter(env.learner_agents))  # Should be set by AgentControlWrapper
        else:
            raise ValueError(
                "AECToGymWrapper requires an env with exactly one learner agent. "
                "Wrap with SinglePlayerWrapper first."
            )
        # Set up Gym spaces from the learner's perspective
        self.observation_space = env.observation_space(self.learner_agent)
        self.action_space = env.action_space(self.learner_agent)

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore
        """Reset environment and return learner's first observation."""
        # reset already steps through when AutoControlWrapper is used
        self.env.reset(seed, options)
        # reset stops at the learner's turn, so this is its first observation
        obs, reward, done, truncated, info = self.env.last()
        return obs, info

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute learner's action, iterate through and return the next observation."""
        self.env.step(action)
        # The downstream call to self.env.step() stops at the learner's turn, so this is its next observation
        obs, reward, done, truncated, info = self.env.last()
        if done or truncated:
            # If the game ended, we remove the learner agent from the environment and continue playing till the end.
            # The learner will not observe anything happening after this, so autoplay could also be stopped.
            # However, proper game_end logging only happens when all agents have terminated.
            # So, this is not the most efficient, but for now the cleanest way to handle this case.
            self.env.step(None)
        return obs, reward, done, truncated, info

    def render(self):
        """Delegate rendering to wrapped env."""
        return self.env.render()

    def close(self):
        """Close the wrapped environment."""
        self.env.close()


def order_agent_mapping_by_agent_id(agent_mapping: dict[AgentID, Any]):
    """Returns the given agent mappings sorted by agent id.

    For example, an order in keys like player_0, player_1, ...
    """

    def agent_key(entry: tuple[AgentID, Any]):
        agent_id = entry[0]
        agent_number = agent_id.split('_')[1]
        return int(agent_number)

    return collections.OrderedDict(sorted(agent_mapping.items(), key=agent_key))


class AgentControlWrapper(BaseWrapper):
    """
    This wrapper allows configuring mixed control settings:
    Learner agents remain externally controlled, but other agents are stepped automatically internally.

    Specifically, agents marked as "learner" return control to the caller.
    Other agents are automated with provided models, players, or callable policies.

    Note: When there is no "learner" in the agent mapping,
    then the control will only be given back to the caller when the episode ended.
    This behavior can be useful for full simulations or evaluation runs without a learner.
    """

    def __init__(
            self,
            env: AECEnv,
            agent_mapping: dict[AgentID, Literal["learner"] | EnvAgent]
    ):
        super().__init__(env)
        self.agent_mapping = order_agent_mapping_by_agent_id(agent_mapping)
        self.learner_agents = [agent_id for agent_id, agent in agent_mapping.items() if agent == "learner"]
        # Store callable agents separately (these bypass the Player and are called directly)
        self.callable_agents = {
            agent_id: agent for agent_id, agent in agent_mapping.items()
            if callable(agent) and not isinstance(agent, Model)
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        options = options or {}
        # Augment reset options with player_models but don't overwrite if the caller provided it
        if "player_models" not in options:
            player_models = []
            # assume an order in keys like player_0, player_1, ...
            for agent_id, agent in self.agent_mapping.items():
                if agent == "learner":  # marker model for the learner (called outside the loop)
                    player_models.append(CustomResponseModel(ModelSpec(model_name="learner")))
                elif agent_id in self.callable_agents:  # marker model for callable agents (called directly)
                    player_models.append(CustomResponseModel(ModelSpec(model_name="callable")))
                else:  # other player models are backed by the passed models, e.g., loaded via load_model("gpt5")
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
                return  # Let the caller invoke last(), e.g., to remove the agent when termination=True
            observation, reward, termination, truncation, info = self.env.last()
            if termination or truncation:
                super().step(None)  # Remove terminated env agents
                continue
            env_agent = self.get_env_agent(agent_id)
            auto_action = env_agent(observation)
            super().step(auto_action)
        # Here the episode ended before the learner took control again, i.e., all agents have been removed

    def get_env_agent(self, agent_id: AgentID):
        # Return the env agent for the agent_id
        # If a callable was provided, use it directly; otherwise use the Player from the game master
        if agent_id in self.callable_agents:
            return self.callable_agents[agent_id]
        return self.unwrapped.player_by_agent_id[agent_id]


class SinglePlayerWrapper(AgentControlWrapper):
    """
    This wrapper exposes all game environments as single-agent RL environments.

    This means that any other player than "learner" is automatically controlled by the provided other agents.
    """

    def __init__(
            self,
            env: AECEnv,
            learner_agent: AgentID = "player_0",
            env_agents: dict[AgentID, EnvAgent] = None
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

    def __init__(
            self,
            env_class: Callable[[GameBenchmark], AECEnv],
            *,
            game_spec: GameSpec,
            **env_kwargs
    ):
        self.callbacks = env_kwargs.get("callbacks") or GameBenchmarkCallbackList()
        self.game_benchmark = GameBenchmark.load_from_spec(game_spec)
        self.callbacks.on_benchmark_start(self.game_benchmark)
        super().__init__(env_class(self.game_benchmark, **env_kwargs))

    def close(self) -> None:
        super().close()
        self.callbacks.on_benchmark_end(self.game_benchmark)
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
