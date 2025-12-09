from typing import ContextManager, Optional

from clemcore.clemgame import GameInstanceIterator, GameBenchmark, GameRegistry
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils.env import AgentID, ObsType, ActionType

from clemcore.clemgame.master import DialogueGameMaster


def env(game_name: str):
    """
    PettingZoo style factory method for game envs.

    Args:
        game_name: the name of the clem-game to wrap as a PZ env
    Returns: a fully initialized game env ready for RL-like training

    """
    # Load game registry
    game_registry = GameRegistry.from_directories_and_cwd_files()
    game_spec = game_registry.get_game_specs_that_unify_with(game_name)[0]
    # Load the packaged default instances.json to be played
    game_iterator = GameInstanceIterator.from_game_spec(game_spec)
    game_context_manager = GameBenchmark.load_from_spec(game_spec)
    # return env_from_spec(game_spec, game_iterator, single_pass=single_pass)


class GameMasterEnv(AECEnv):

    def __init__(self, game_benchmark: GameBenchmark):
        super().__init__()
        self.game_benchmark = game_benchmark
        self.game_master: Optional[DialogueGameMaster] = None  # initialized on reset()

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
        self.options = options or {}
        assert "experiment" in options, "Missing 'experiment' in reset options"
        assert "player_models" in options, "Missing 'player_models' in reset options"
        assert "game_instance" in options, "Missing 'game_instance' in reset options"
        # GM.setup() adds players, i.e., is not idempotent. Therefore, we create a new GM instance here.
        experiment = options["experiment"]
        player_models = options["player_models"]
        game_instance = options["game_instance"]
        self.game_master: DialogueGameMaster = self.game_benchmark.create_game_master(experiment, player_models)
        self.game_master.setup(**game_instance)
        # Only after setup() the players are set
        self.agents = [player.name for player in self.game_master.get_players()]
        self.possible_agents = self.agents.copy()
        self.agent_selection = self.game_master.current_player.name

        for agent in self.agents:
            # GameMaster should implement this by default;
            # OK maybe the implemented game should provide a more concrete upper bound on the content length
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
        current_player_name = self.game_master.current_player.name
        done, info = self.game_master.step(action)
        # for now we only have the case that all players end at the same time
        for player in self.game_master.get_players():
            self.terminations[player.name] = done
            self.truncations[player.name] = done
        self.infos[current_player_name] = info
        # response_score is returned in legacy master
        self.rewards[current_player_name] = info["response_score"] if "response_score" in info else info["turn_score"]
        self._accumulate_rewards()
        if done:
            self.agent_selection = None
            self.agents = []  # this signals the play loop to terminate
        # next player
        self.agent_selection = self.game_master.current_player.name

    def observe(self, agent: AgentID) -> ObsType | None:
        """Returns the observation an agent currently can make.

        `last()` calls this function.
        """
        player = self.game_master.players_by_names[agent]
        return self.game_master.get_context_for(player)

    def observation_space(self, agent: AgentID):
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentID):
        """ Use AEC wrapper to change the action space """
        return self.action_spaces[agent]
