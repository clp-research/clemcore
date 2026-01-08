from typing import Callable

import gymnasium

from clemcore.backends.model_registry import Model, CustomResponseModel
from clemcore.clemgame.callbacks.base import GameBenchmarkCallbackList, GameStep
from clemcore.clemgame.registry import GameRegistry
from clemcore.clemgame.instances import GameInstanceIterator
from clemcore.clemgame.benchmark import GameBenchmark
from clemcore.clemgame.master import DialogueGameMaster
from clemcore.clemgame.envs.pettingzoo.wrappers import (
    GameInstanceIteratorWrapper,
    GameBenchmarkWrapper,
    SinglePlayerWrapper,
    AECToGymWrapper
)

from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils.env import AgentID, ObsType, ActionType


def gym_env(game_name: str,
            *,
            game_instance_filter: Callable[[str, str], list[int]] | None = None,
            single_pass: bool = False,
            learner_agent: AgentID = "player_0",
            env_agents: dict[AgentID, Model] | None = None,
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
        env_agents: a mapping from agent ids to players (e.g. {player_1: "gpt5"})
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
    callbacks = callbacks or GameBenchmarkCallbackList()

    # Load game registry
    game_registry = GameRegistry.from_directories_and_cwd_files()
    game_spec = game_registry.get_game_specs_that_unify_with(game_name)[0]
    game_env = GameBenchmarkWrapper(GameMasterEnv, game_spec=game_spec, callbacks=callbacks)

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

    def get_current_agent(self):
        """ Mapping the current player to an agent id """
        return self.player_to_agent_id[self.game_master.current_player.name]

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.options = options or {}
        assert "experiment" in options, "Missing 'experiment' in reset options"
        assert "game_instance" in options, "Missing 'game_instance' in reset options"
        # GM.setup() adds players, i.e., is not idempotent. Therefore, we create a new GM instance here.
        self.experiment = options["experiment"]
        self.game_instance = options["game_instance"]
        player_models = (options.get("player_models", None)
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
            self.observation_spaces[agent] = self.observation_space(agent)
            self.action_spaces[agent] = self.action_space(agent)
            self.terminations[agent] = False
            self.truncations[agent] = False
            self.rewards[agent] = 0.
            self._cumulative_rewards[agent] = 0.
            self.infos[agent] = {}

    def step(self, action: ActionType) -> None:
        """Accepts and executes the action of the current agent_selection in the environment.

        Automatically switches control to the next agent.
        """
        # after step() current_player might have changed, so we reference it here already
        current_agent = self.get_current_agent()
        current_context = self.observe(current_agent)
        # step possibly transitions the current agent
        done, info = self.game_master.step(action, log_event=True)
        game_step = GameStep(current_context, action, done, info)
        self.callbacks.on_game_step(self.game_master, self.game_instance, game_step)
        # for now we only have the case that all players end at the same time
        for agent_id in self.agents:
            self.terminations[agent_id] = done
            self.truncations[agent_id] = done
        self.infos[current_agent] = info
        # response_score is returned in legacy master
        self.rewards[current_agent] = info["response_score"] if "response_score" in info else info["turn_score"]
        self._accumulate_rewards()
        if done:
            self.callbacks.on_game_end(self.game_master, self.game_instance)
            self.agent_selection = None
            self.agents = []  # this signals the play loop to terminate
        # next player
        self.agent_selection = self.get_current_agent()

    def observe(self, agent: AgentID) -> ObsType | None:
        """Returns the observation an agent currently can make.

        `last()` calls this function.
        """
        player = self.player_by_agent_id[agent]
        return self.game_master.get_context_for(player, log_event=True)

    def observation_space(self, agent: AgentID):
        """All agents share the same observation space.

        If necessary, use AEC wrapper to change the action space, e.g., to include images.
        """
        return spaces.Dict(
            {
                "role": spaces.Text(max_length=128),  # should be enough chars for a role name
                "content": spaces.Text(max_length=8192)  # should be enough chars for prompt and context
            }
        )

    def action_space(self, agent: AgentID):
        """All agents share the same action space. The agents are supposedly generalist models.

        If necessary, use AEC wrapper to change the action space.
        """
        return spaces.Text(max_length=8192)
