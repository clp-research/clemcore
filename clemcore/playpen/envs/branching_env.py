import os
from copy import deepcopy
from typing import List, Dict, Callable, Tuple, Union

from clemcore.backends import Model
from clemcore.clemgame import GameBenchmark, GameInstanceIterator, Player
from clemcore.playpen import GameEnv
from clemcore.playpen.envs import PlayPenEnv


class BranchingCandidate:

    def __init__(self, parent_env: GameEnv, branch_env: GameEnv, done: bool, info: Dict):
        self.parent_env = parent_env
        self.branch_env = branch_env
        self.done = done
        self.info = info


class BranchingResponse:

    def __init__(self, parent_env: GameEnv, branch_env: GameEnv, branch_response: str):
        self.parent_env = parent_env
        self.branch_env = branch_env
        self.branch_response = branch_response

    def __str__(self):
        return self.branch_response


class GameBranchingEnv(PlayPenEnv):
    """
    A game benchmark environment that branches after each step, that is,
    the games states multiply as determined by the branching factor.
    This allows to collect at each step multiple responses for the same context.
    """

    def __init__(self, game: GameBenchmark, player_models: List[Model], task_iterator: GameInstanceIterator,
                 branching_factor: int, branching_model=None):
        super().__init__()
        assert branching_factor > 0, "The branching factor must be greater than zero"
        self._root: GameEnv = GameEnv(game, player_models, task_iterator)
        self._active_envs: List[GameEnv] = [self._root]
        self._branching_factor: int = branching_factor
        self._branching_model = branching_model

    def reset(self) -> None:  # all game branches always operate on the same task / episode
        self._root.reset()
        self._active_envs: List[GameEnv] = [self._root]

    def observe(self) -> Tuple[Union[Player, Callable], Union[Dict, List[Dict]]]:
        contexts: List[Dict] = []
        players: List[Player] = []
        for game_env in self._active_envs:
            player, context = game_env.observe()
            players.append(player)
            contexts.append(context)
        # GameBranchingPlayer assumes that (parent_env, parent_context) can be re-assembled by zipping (using the order)
        branching_player = GameBranchingPlayer(self._active_envs, players,
                                               self._branching_factor, self._branching_model)
        return branching_player, contexts

    def step(self, responses: Union[str, List]) -> Tuple[Union[bool, List], Union[Dict, List]]:
        assert isinstance(responses, list), f"GameTreeEnv expects a list of responses and not {responses.__class__}"

        context_dones = []
        context_infos = []
        candidates: List[BranchingCandidate] = []  # called candidates because we considered to apply a pruning function
        for context_responses in responses:
            response_dones = []
            response_infos = []
            for response in context_responses:  # each response represents a possible branch in the tree
                done, info = response.branch_env.step(response.branch_response)
                response_dones.append(done)
                response_infos.append(info)
                candidate = BranchingCandidate(response.parent_env, response.branch_env, done, info)
                candidates.append(candidate)
            context_dones.append(response_dones)
            context_infos.append(response_infos)

        self._done = all([candidate.done for candidate in candidates])

        # memorize leaves so that we do not have to find them again
        self._active_envs = [candidate.branch_env for candidate in candidates]

        # return all dones and infos so that they match the quantity of the responses
        return context_dones, context_infos

    def store_records(self, top_dir: str, rollout_dir: str, episode_dir: str,
                      store_experiment: bool = False, store_instance: bool = False):
        for branch_idx, game_env in enumerate(self._active_envs):
            game_env.store_records(top_dir, rollout_dir, os.path.join(episode_dir, f"branch_{branch_idx}"),
                                   store_experiment, store_instance)

    def is_done(self) -> bool:
        return self._done


class GameBranchingPlayer(Callable):
    """    Applies a player to a given context as many times as determined by the branching factor. """

    def __init__(self, current_envs: List[GameEnv], current_players: List[Player],
                 branching_factor: int = 1, branching_model=None):
        assert branching_factor > 0, "The branching factor must be greater than zero"
        self._branching_factor = branching_factor
        self._branching_model = branching_model
        self._current_envs = current_envs
        self._current_players = current_players

    @property
    def model(self):
        if not self._current_players:
            return None
        # for now, we assume that all current player share the same model
        return self._current_players[0].model

    def _should_branch(self):
        if self._branching_model is None:
            return True
        return self.model is self._branching_model

    def __call__(self, contexts: List[str]) -> List[List[BranchingResponse]]:
        """
        For each context we return multiple responses that possibly transition the environment.
        :param contexts:
        :return:
        """
        assert isinstance(contexts, List), "The context for TreePlayer must be a list of game environments"
        assert len(self._current_envs) == len(contexts), "There must be as many active branches as given contexts"
        context_responses = []
        for parent_env, parent_context in zip(self._current_envs, contexts):
            branch_responses = []
            if self._should_branch():
                for _ in range(self._branching_factor):
                    branch_env: GameEnv = deepcopy(parent_env)
                    branch_player = branch_env.master.get_current_player()  # we use the branch player as it keeps state
                    branch_response = branch_player(
                        parent_context)  # this already changes the player state in branch env
                    branch_responses.append(BranchingResponse(parent_env, branch_env, branch_response))
            else:
                player = parent_env.master.get_current_player()
                response = player(parent_context)
                branch_responses.append(BranchingResponse(parent_env, parent_env, response))
            context_responses.append(branch_responses)
        return context_responses


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
