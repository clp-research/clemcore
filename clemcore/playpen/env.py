import abc
from contextlib import contextmanager
from typing import List, Tuple, Dict, Callable, Union

from clemcore.backends import Model
from clemcore.clemgame import GameSpec, benchmark, GameBenchmark, DialogueGameMaster
from clemcore.playpen.game_tree import GameTree


@contextmanager
def make_env(game_spec: GameSpec, players: List[Model],
             instances_name: str = None, shuffle_instances: bool = False,
             branching_factor: int = 1, pruning_fn=lambda candidates: candidates):
    with benchmark.load_from_spec(game_spec, do_setup=True, instances_name=instances_name) as game:
        assert branching_factor > 0, "The branching factor must be greater than zero"
        if branching_factor == 1:
            # this could also resolve to a tree env with branching factor one, but
            # for clarity we instantiate here the game benchmark environment directly
            yield GameEnv(game, players=players, shuffle_instances=shuffle_instances)
        else:
            yield GameTreeEnv(game, players=players, shuffle_instances=shuffle_instances,
                              branching_factor=branching_factor, pruning_fn=pruning_fn)


class State:
    ...


class PlayPenEnv(abc.ABC):

    @abc.abstractmethod
    def reset(self) -> None:
        pass

    @abc.abstractmethod
    def observe(self) -> Tuple[Callable, Union[List, Dict], State]:
        pass

    @abc.abstractmethod
    def step(self, response: Union[str, List]) -> Tuple[bool, Dict]:
        pass


class GameEnv(PlayPenEnv):

    def __init__(self, game: GameBenchmark, player_models: List[Model], shuffle_instances: bool = False):
        self._game = game
        self._player_models = player_models
        # setup iterator to go through tasks / game instances
        self._task_iterator = game.create_game_instance_iterator(shuffle_instances)
        if len(self._task_iterator) < 1:
            raise RuntimeError(f"No game instances given for the game: '{self._game.game_name}'")
        # variables initialized on reset()
        self._game_instance = None
        self._experiment_config = None
        self._master: DialogueGameMaster = None
        # reset here so that game env is fully functional after init
        self.reset()

    @property
    def master(self):
        return self._master

    @master.setter
    def master(self, master):
        self._master = master

    def reset(self) -> None:
        try:
            self.experiment_config, self.game_instance = next(self._task_iterator)
            self.master = self._game.create_game_master(self.experiment_config, self._player_models)
            self.master.setup(**self.game_instance)
        except StopIteration:
            self._task_iterator.reset()
            self.reset()

    def observe(self) -> Tuple[Callable, Union[List, Dict], State]:
        player = self.master.get_current_player()
        context = self.master.get_context_for(player)
        state = self.master.get_game_state()  # todo
        return player, context, State()

    def step(self, response: Union[str, List]) -> Tuple[bool, Dict]:
        return self.master.step(response)

    def clone(self) -> "GameEnv":
        ...


class BranchingCandidate:

    def __init__(self, parent_env: GameEnv, branch_env: GameEnv, done: bool, info: Dict):
        self._parent_env = parent_env
        self._branch_env = branch_env
        self._done = done
        self._info = info

    @property
    def done(self):
        return self.done

    @property
    def parent_env(self):
        return self._parent_env

    @property
    def branch_env(self):
        return self._branch_env


class BranchingResponse:

    def __init__(self, parent_env: GameEnv, branch_env: GameEnv, response: str):
        self._parent_env = parent_env
        self._branch_env = branch_env
        self._response = response

    def branch(self) -> BranchingCandidate:
        done, info = self._branch_env.step(self._response)
        return BranchingCandidate(self._parent_env, self._branch_env, done, info)


class GameTreeEnv(PlayPenEnv):
    """
    A game benchmark environment that branches after each step, that is,
    the games states multiply as determined by the branching factor.
    This allows to collect at each step multiple responses for the same context.
    A pruning function can be given to reduce the growing number of environments.
    """

    def __init__(self, game: GameBenchmark, player_models: List[Model], shuffle_instances: bool,
                 branching_factor: int, pruning_fn: Callable):
        assert branching_factor > 0, "The branching factor must be greater than zero"
        self._root = GameEnv(game, player_models, shuffle_instances)
        self._leaves = [self._root]
        self._game_tree = GameTree(self._root)
        self._branching_factor = branching_factor
        self._pruning_fn = pruning_fn

    def reset(self) -> None:
        # for simplicity, all game environment always operate on the same episode
        # (if the instances are not shuffled) or are at least reset at the same time
        self._game_tree.reset()

    def observe(self) -> Tuple[Callable, Union[List, Dict], State]:
        return GameTreePlayer(self._branching_factor), self._leaves, State()

    def step(self, responses: Union[str, List]) -> Tuple[bool, Dict]:
        assert isinstance(responses, list), f"GameTreeEnv expects a list of responses and not {responses.__class__}"
        candidates: List[BranchingCandidate] = []
        for response in responses:  # each response represents a possible branch in the tree
            candidates.append(response.branch())  # step

        # establish such responses as branches in the tree that were not pruned
        # these responses will determine the new leaves of the tree
        selected_candidates = self._pruning_fn([candidates])

        for selected_candidate in selected_candidates:
            self._game_tree.add_branch(selected_candidate.parent_env, selected_candidate.branch_env)

        # memorize leaves so that we do not have to find them again
        self._leaves = [candidate.branch_env for candidate in selected_candidates]
        # only done when all branches are done
        done = all([candidate.done for candidate in selected_candidates])
        return done, dict()  # todo


class GameTreePlayer(Callable):
    """    Applies a player to a given context as many times as determined by the branching factor. """

    def __init__(self, branching_factor: int = 1):
        assert branching_factor > 0, "The branching factor must be greater than zero"
        self._branching_factor = branching_factor

    def __call__(self, context: List):
        assert isinstance(context, List), "The context for TreePlayer must be TreePlayerContext"
        tree_responses = []
        for game_env in context:
            for _ in range(self._branching_factor):
                branch_env = game_env.clone()
                master = branch_env.master
                player = master.get_current_player()
                context = master.get_context_for(player)
                response = player(context)  # this already changes the player state in branch env
                tree_responses.append(BranchingResponse(game_env, branch_env, response))
        return tree_responses
