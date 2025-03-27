from typing import List, Dict, Callable, Tuple, Union

from clemcore.backends import Model
from clemcore.clemgame import GameBenchmark
from clemcore.playpen import GameEnv
from clemcore.playpen.envs import PlayPenEnv


class BranchingCandidate:

    def __init__(self, parent_env: GameEnv, branch_env: GameEnv, done: bool, info: Dict):
        self.parent_env = parent_env
        self.branch_env = branch_env
        self.done = done
        self.info = info


class BranchingResponse:

    def __init__(self, parent_env: GameEnv, parent_context: str, branch_env: GameEnv, branch_response: str):
        self.parent_env = parent_env
        self.parent_context = parent_context
        self.branch_env = branch_env
        self.branch_response = branch_response
        self.done = None
        self.info = None


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
        self._root: GameEnv = GameEnv(game, player_models, shuffle_instances)
        self._active_branches: List[GameEnv] = [self._root]
        self._game_tree: GameTree = GameTree(self._root)
        self._branching_factor: int = branching_factor
        self._pruning_fn: Callable = pruning_fn

    def reset(self) -> None:
        # for simplicity, all game environment always operate on the same episode
        # (if the instances are not shuffled) or are at least reset at the same time
        self._game_tree.reset()

    def observe(self) -> Tuple[Callable, Union[List, Dict]]:
        return GameTreePlayer(self._branching_factor), self._active_branches

    def step(self, responses: Union[str, List]) -> Tuple[bool, Dict]:
        assert isinstance(responses, list), f"GameTreeEnv expects a list of responses and not {responses.__class__}"

        def branching_step(_response: BranchingResponse) -> BranchingCandidate:
            _response.done, _response.info = _response.branch_env.step(_response.branch_response)
            return BranchingCandidate(_response.parent_env, _response.branch_env, _response.done, _response.info)

        candidates: List[BranchingCandidate] = []
        for branching_response in responses:  # each response represents a possible branch in the tree
            candidate = branching_step(branching_response)
            candidates.append(candidate)

        # establish such responses as branches in the tree that were not pruned
        # these responses will determine the new leaves of the tree
        selected_candidates = self._pruning_fn([candidates])

        for selected_candidate in selected_candidates:
            self._game_tree.add_branch(selected_candidate.parent_env, selected_candidate.branch_env)

        # memorize leaves so that we do not have to find them again
        self._active_branches = [candidate.branch_env for candidate in selected_candidates]
        # todo: handle pruned branches that became orphans
        # only done when all branches are done
        done = all([candidate.done for candidate in selected_candidates])
        return done, {"collector": "GameTreeEnvCollector"}


class GameTreePlayer(Callable):
    """    Applies a player to a given context as many times as determined by the branching factor. """

    def __init__(self, branching_factor: int = 1):
        assert branching_factor > 0, "The branching factor must be greater than zero"
        self._branching_factor = branching_factor

    def __call__(self, game_envs: List[GameEnv]):
        assert isinstance(game_envs, List), "The context for TreePlayer must be a list of game environments"
        branching_responses = []
        for parent_env in game_envs:
            parent_player = parent_env.master.get_current_player()
            parent_context = parent_env.master.get_context_for(parent_player)
            for _ in range(self._branching_factor):
                branch_env = parent_env.clone()
                player = branch_env.master.get_current_player()  # we need the branch player, because it keeps state
                branch_response = player(parent_context)  # this already changes the player state in branch env
                branching_responses.append(BranchingResponse(parent_env, parent_context, branch_env, branch_response))
        return branching_responses


class GameTreeNode:
    def __init__(self, game_env: GameEnv):
        self._game_env = game_env
        self._branches: List[GameTreeNode] = []
        self._parent: GameTreeNode = self  # root is its own parent

    def __iter__(self):
        return iter(self._branches)

    def __bool__(self):
        return bool(self._branches)

    def unwrap(self):
        return self._game_env

    def wraps(self, game_env: GameEnv) -> bool:
        return self._game_env is game_env

    def add_branch(self, branch_node: "GameTreeNode"):
        self._branches.append(branch_node)
        branch_node._parent = self


class GameTree:

    def __init__(self, root: GameEnv):
        self._root: GameTreeNode = GameTreeNode(root)

    def reset(self):
        ...  # todo

    def find_node(self, target_env: GameEnv):
        def _find_node(node):
            if node.wraps(target_env):  # check for object identity
                return node

            for branch in node:
                target_node = _find_node(branch)
                if target_node:
                    return target_node

            return None

        return _find_node(self._root)

    def find_leaves(self, unwrap=False):
        def _find_leaves(node):
            if not node:
                if unwrap:
                    return [node.unwrap()]
                return [node]
            leaves = []
            for branch in node:
                leaves.extend(_find_leaves(branch))
            return leaves

        return _find_leaves(self._root)

    def add_branch(self, parent_env, branch_env):
        # Find parent node and add child
        parent_node = self.find_node(parent_env)
        assert parent_node is not None, "There must be a parent node that wraps the candidates parent env"
        branch_node = GameTreeNode(branch_env)
        parent_node.add_branch(branch_node)
