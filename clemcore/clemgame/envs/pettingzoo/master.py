from typing import Callable

import gymnasium

from clemcore.backends.model_registry import CustomResponseModel
from clemcore.clemgame.callbacks.base import GameBenchmarkCallbackList, GameStep
from clemcore.clemgame.registry import GameRegistry
from clemcore.clemgame.instances import GameInstanceIterator
from clemcore.clemgame.benchmark import GameBenchmark
from clemcore.clemgame.master import DialogueGameMaster
from clemcore.clemgame.envs.pettingzoo.wrappers import (
    GameInstanceIteratorWrapper,
    GameBenchmarkWrapper,
    SinglePlayerWrapper,
    AECToGymWrapper,
    EnvAgent
)

from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils.wrappers import OrderEnforcingWrapper
from pettingzoo.utils.env import AgentID, ObsType, ActionType


def gym_env(game_name: str,
            *,
            game_instance_filter: Callable[[str, str], list[int]] | None = None,
            single_pass: bool = False,
            learner_agent: AgentID = "player_0",
            env_agents: dict[AgentID, EnvAgent] | None = None,
            callbacks: GameBenchmarkCallbackList | None = None
            ) -> gymnasium.Env:
    """
    Factory method for Gymnasium style game envs.

    This creates first a normal AECEnv and then wraps it into a gymnasium.Env with SinglePlayerWrapper.

    Note:

        The callback methods are called on the following events:
         - `on_benchmark_start()` during env.init() (in GameBenchmarkWrapper)
         - on_benchmark_end() during env.close() (in GameBenchmarkWrapper)
         - on_game_start() during env.reset() (in GameMasterEnv)
         - on_game_end() during env.step() when all agents reached a terminal state (in GameMasterEnv)
         - on_game_step() for actions during env.step() when the agent has not reached terminal state (in GameMasterEnv)

    Args:
        game_name: The name of the clem-game to wrap as a PZ env
        game_instance_filter: A callable mapping from (game_name, experiment_name) tuples to lists of game instance ids.
        single_pass: Whether to run through the game instances only once or multiple times.
        learner_agent: the agent id of the learner agent (e.g. player_0)
        env_agents: a mapping from agent ids to Models or Callables (e.g. {player_1: gpt5} or {player_1: lambda obs: "action"})
        callbacks: a list of callbacks to be applied to the environment lifecycle

    Returns:
        A fully initialized game env ready for RL-like training
    """
    game_env = env(game_name, game_instance_filter=game_instance_filter, single_pass=single_pass, callbacks=callbacks)
    game_env = SinglePlayerWrapper(game_env, learner_agent, env_agents=env_agents)
    game_env = AECToGymWrapper(game_env)
    return game_env


def env(game_name: str,
        *,
        game_instance_filter: Callable[[str, str], list[int]] | None = None,
        single_pass: bool = False,
        callbacks: GameBenchmarkCallbackList | None = None
        ) -> AECEnv:
    """
    Factory method for Pettingzoo style game envs.

    We do not perform an agent mapping here, but the caller has to define this in his training loop.

    Note:

        The callback methods are called on the following events:
         - on_benchmark_start() during env.init() (in GameBenchmarkWrapper)
         - on_benchmark_end() during env.close() (in GameBenchmarkWrapper)
         - on_game_start() during env.reset() (in GameMasterEnv)
         - on_game_end() during env.step() when a terminal state is reached (in GameMasterEnv)
         - on_game_step() for actions during env.step() when no terminal state is reached (in GameMasterEnv)

    Args:
        game_name: The name of the clem-game to wrap as a PZ env
        game_instance_filter: A callable mapping from (game_name, experiment_name) tuples to lists of game instance ids.
        single_pass: Whether to run through the game instances only once or multiple times.
        callbacks: a list of callbacks to be applied to the environment lifecycle

    Returns:
        A fully initialized game env ready for RL-like training
    """
    # Load game registry
    game_registry = GameRegistry.from_directories_and_cwd_files()
    game_spec = game_registry.get_game_specs_that_unify_with(game_name)[0]
    game_env = GameBenchmarkWrapper(GameMasterEnv, game_spec=game_spec, callbacks=callbacks)

    # Warn env users in case of wrong method execution order
    game_env = OrderEnforcingWrapper(game_env)

    # Load the packaged default instances.json to be played and pass an optional filter
    game_iterator = GameInstanceIterator.from_game_spec(game_spec, sub_selector=game_instance_filter)
    game_env = GameInstanceIteratorWrapper(game_env, game_iterator, single_pass=single_pass)
    return game_env


class GameMasterEnv(AECEnv):

    def __init__(self, game_benchmark: GameBenchmark, *, callbacks: GameBenchmarkCallbackList | None = None):
        super().__init__()
        self.game_benchmark = game_benchmark
        self.callbacks = callbacks or GameBenchmarkCallbackList()
        self.game_master: DialogueGameMaster | None = None  # initialized on reset()
        self.game_instance: dict | None = None  # initialized on reset()
        self.experiment: dict | None = None  # initialized on reset()
        self.player_by_agent_id = {}  # mapping between agent ids and player instances
        self.player_to_agent_id = {}  # mapping player names to agent ids

        # initialize pettingzoo env
        self.options = {}
        self.metadata = dict(name=self.game_benchmark.game_spec.game_name)
        self.observation_spaces = dict()
        self.action_spaces = dict()
        self.rewards = dict()
        self.terminations = dict()
        self.truncations = dict()
        self._cumulative_rewards = dict()
        self.infos = dict()
        self.agents = []
        self.possible_agents = []

        # default spaces for all agents
        self._observation_space = spaces.Dict(
            {
                "role": spaces.Text(max_length=128),  # should be enough chars for a role name
                "content": spaces.Text(max_length=8192)  # should be enough chars for prompt and context
            }
        )
        self._action_space = spaces.Text(max_length=8192)

    def get_current_agent(self):
        """ Mapping the current player to an agent id """
        return self.player_to_agent_id[self.game_master.current_player.name]

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.options = options or {}
        assert "experiment" in self.options, "Missing 'experiment' in reset options"
        assert "game_instance" in self.options, "Missing 'game_instance' in reset options"
        # GM.setup() adds players, i.e., is not idempotent. Therefore, we create a new GM instance here.
        self.experiment = self.options["experiment"]
        self.game_instance = self.options["game_instance"]
        player_models = (self.options.get("player_models", None)
                         or [CustomResponseModel()] * self.game_benchmark.game_spec.players)
        self.game_master: DialogueGameMaster = self.game_benchmark.create_game_master(self.experiment, player_models)
        self.game_master.setup(**self.game_instance)
        self.callbacks.on_game_start(self.game_master, self.game_instance)
        # Only after setup() the players are set
        self.player_by_agent_id = {f"player_{idx}": player
                                   for idx, player in enumerate(self.game_master.get_players())}
        self.player_to_agent_id = {player.name: f"player_{idx}"
                                   for idx, player in enumerate(self.game_master.get_players())}
        self.agents = list(self.player_to_agent_id.values())
        self.possible_agents = self.agents.copy()
        self.agent_selection = self.get_current_agent()

        for agent in self.agents:
            # GameMaster should implement this by default;
            # OK maybe the implemented game should provide a more concrete upper bound on the content length
            # If you have images, then you should also define them here
            self.observation_spaces[agent] = self._observation_space
            self.action_spaces[agent] = self._action_space
            self.terminations[agent] = False
            self.truncations[agent] = False
            self.rewards[agent] = 0.
            self._cumulative_rewards[agent] = 0.
            self.infos[agent] = {}

    def step(self, action: ActionType) -> None:
        """Accepts, executes, and logs the action of the current agent in the environment.

        Note:
            - The transcript logging assumes that the actions are responses to the context of the current agent.
            - The transcript logging is disabled for None actions (agent cleanup), so these won't appear in transcripts.

        Args:
            action: the agent's response to the current context of it
        Returns:
            None - internal state transition to the (supposedly) next agent
        """
        # Standard PettingZoo check: handles the final "None" step for dead agents to observe their final reward
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            # Note: This removes the agent and selects the next (dead) agent in self.agents
            # or selects the next live agent stored during _deads_step_first() at the end of this step
            self._was_dead_step(action)
            return

        # After step() current_player might have changed, so we reference it here already
        current_agent = self.get_current_agent()

        # Get the context that was given from GM -> Player (logging happens in game_master.step)
        current_context = self.game_master.get_context_for(self.player_by_agent_id[current_agent])

        # Step possibly transitions the current agent (as specified by the game master)
        # Log the response action from Player -> GM
        done, info = self.game_master.step(action)

        # Update current rewards and info for the current agent (response_score is returned in legacy master)
        self._cumulative_rewards[current_agent] = 0
        self.rewards[current_agent] = info.get("turn_score", info.get("response_score", 0.))
        self.infos[current_agent] = info

        # Inform callbacks about the game step results
        game_step = GameStep(current_context, action, done, info)
        self.callbacks.on_game_step(self.game_master, self.game_instance, game_step)

        # The terminal case:
        # - Flag everyone as terminated so they all get a final "None" turn to observe the final reward
        # - Distribute the game outcome as a team reward to all agents together (overwriting the individual turn_score)
        if done:
            for agent_id in self.agents:
                # Note: we do not handle truncations separately yet, e.g., running out of turns
                self.terminations[agent_id] = True
                self.rewards[agent_id] = info.get("episode_score", 0.)
            self.callbacks.on_game_end(self.game_master, self.game_instance)

        # Collect and reset the rewards for all agents
        # Note: We accumulate the rewards to collect all environmental impacts on an agent until its next call of last()
        self._accumulate_rewards()
        self._clear_rewards()

        # Select the next player determined by the game master after game_master.step()
        # Note: For games done, this might return the same player as before. However, this is okay (can be ignored),
        # because in our games all agents terminate at the same turn simultaneously and are cleaned up all together
        self.agent_selection = self.get_current_agent()

        # Store the current self.agent_selection to continue with after cleanup of agents that terminated at this step
        # Note: This already supports games where not all agents terminate together at the end of the game
        self._deads_step_first()

    def observe(self, agent: AgentID) -> ObsType | None:
        """Returns the observation an agent currently can make.

        `last()` calls this function.

        Note:
            If no context has been set for the player (e.g., game aborted before their turn),
            returns the initial prompt if available, otherwise None.
        """
        player = self.player_by_agent_id[agent]
        try:
            return self.game_master.get_context_for(player)
        except AssertionError:
            # Handle case where context isn't available (e.g., early game abort before player's turn)
            # Fall back to initial prompt if available
            return self.game_master.initial_prompt_for_player.get(player.name)

    def observation_space(self, agent: AgentID):
        """All agents share the same observation space.

        If necessary, use AEC wrapper to change the action space, e.g., to include images.
        """
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentID):
        """All agents share the same action space. The agents are supposedly generalist models.

        If necessary, use AEC wrapper to change the action space.
        """
        return self.action_spaces[agent]
