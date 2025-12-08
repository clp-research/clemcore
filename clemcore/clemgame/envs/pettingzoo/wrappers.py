from typing import ContextManager

from clemcore.backends import CustomResponseModel
from clemcore.clemgame import GameInstanceIterator, GameBenchmark, GameRegistry, GameSpec
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType, ActionType


def env(game_name: str, iterator_config=None, single_pass: bool = False):
    # Load game registry
    game_registry = GameRegistry.from_directories_and_cwd_files()
    game_spec = game_registry.get_game_specs_that_unify_with(game_name)[0]
    # Load the packaged default instances.json to be played
    game_iterator = GameInstanceIterator.from_game_spec(game_spec, config=iterator_config)
    return env_from_spec(game_spec, game_iterator, single_pass=single_pass)


def env_from_spec(
        game_spec: GameSpec,
        game_iterator: GameInstanceIterator,
        *,
        single_pass: bool = False
):
    # Load the game and pre-set the default instance
    game_context_manager = GameBenchmark.load_from_spec(game_spec)
    # Wrap everything in a pettingzoo style env
    return GameInstanceIteratorWrapper(
        PettingZooGameMasterEnv(game_context_manager),
        game_iterator,
        single_pass)


class GameInstanceIteratorWrapper(BaseWrapper):
    """
    A wrapper that iterates through a GameInstanceIterator, either once or infinitely.

    Args:
        wrapped_env: The base PettingZoo environment that accepts task configuration via options['task_config'].
        game_iterator: An instance of GameInstanceIterator pre-loaded with instances.
        single_pass: If True, the iterator stops after one pass through all instances (for evaluation).
                     If False (default), the iterator cycles infinitely (for training).
                     This is useful when using RL libs (like Ray RLLib) which expected an infinite stream of episodes.
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
        options = options or {}
        options["experiment"] = experiment
        options["game_instance"] = game_instance
        super().reset(seed=seed, options=options)

    def unwrapped(self) -> AECEnv:
        return self.wrapped_env


class PettingZooGameMasterEnv(AECEnv):

    def __init__(self, game_context_manager: ContextManager[GameBenchmark]):
        super().__init__()
        self.game_context_manager = game_context_manager
        self.game_benchmark = game_context_manager.__enter__()
        self.game_master = None  # initialized on reset()

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

    def reset(self, seed: int | None = None, options: dict | None = None):
        # game instance infos are given via options dict
        self.options = options or {}
        experiment, game_instance = options["experiment"], options["game_instance"]
        # Clearing the players must go into the GM; we didnt need this before; but setup() must become a proper reset()
        # Hence, we reset() here by creating a whole new GM
        self.game_master = self.game_benchmark.create_game_master(
            experiment,
            [CustomResponseModel(), CustomResponseModel()]
        )
        # Only now setup()
        self.game_master.setup(**game_instance)
        # Only after setup() the players are set (which is a bit weird)
        self.agents = self.game_master.get_players()
        self.possible_agents = self.agents
        self.agent_selection = self.game_master.current_player

        for agent in self.agents:
            # GameMaster should implement this by default;
            # OK maybe the the implemented game should provide a more concrete upper bound on the content length
            # If you have images, then you should also define them here
            self.observation_spaces[agent] = spaces.Dict(
                {
                    "role": spaces.Text(max_length=128),  # should be enough chars for a role name
                    "content": spaces.Text(max_length=8192)  # should be enough chars for prompt and context
                }
            )
            # only a general descriptor of the action space
            self.action_spaces[agent] = spaces.Text(max_length=8192)
            self.terminations[agent] = False
            self.truncations[agent] = False
            self.rewards[agent] = 0.
            self._cumulative_rewards[agent] = 0.
            self.infos[agent] = {}

    def step(self, action: ActionType) -> None:
        """Accepts and executes the action of the current agent_selection in the environment.

        Automatically switches control to the next agent.
        """
        # after step current_player might have changed, so we reference it here already
        # current_player should move into GameMaster
        current_player = self.game_master.current_player
        done, info = self.game_master.step(action)
        # for now we only have the case that all players end at the same time
        for player in self.game_master.get_players():
            self.terminations[player] = done
            self.truncations[player] = done
        self.infos[current_player] = info
        # response_score is returned in legacy master
        self.rewards[current_player] = info["response_score"] if "response_score" in info else info["turn_score"]
        self._accumulate_rewards()
        if done:
            self.agent_selection = None
            self.agents = []  # this signals the play loop to terminate
        # next player
        self.agent_selection = self.game_master.current_player

    def observe(self, agent: AgentID) -> ObsType | None:
        """Returns the observation an agent currently can make.

        `last()` calls this function.
        """
        return self.game_master.get_context_for(agent)

    def observation_space(self, agent: AgentID):
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentID):
        """ Use AEC wrapper to change the action space """
        return self.action_spaces[agent]

    def close(self):
        self.game_context_manager.__exit__(None, None, None)
