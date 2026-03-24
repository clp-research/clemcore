"""
Branching game runner for exploring multiple trajectories.

This module provides functionality to run games with branching, where at certain decision points
the game state is copied and multiple different responses are explored independently.

Key concepts:
- **Branching factor**: Number of parallel branches created when the condition is met
- **Branching condition**: A callable that determines when to create branches
- **Leaf branches**: The final completed game trajectories (e.g., 64 independent gameplays)

The runner creates a tree structure where:
- Each node represents a game state
- Each branch explores a different response/action
- Leaf nodes are complete game episodes

Use cases:
- Collecting multiple model responses for the same context
- Exploring different dialogue paths
- Generating diverse training data
- Stochastic evaluation with repeated sampling
"""
import logging
from typing import List, Optional
from copy import deepcopy

from tqdm import tqdm

from clemcore.backends import Model
from clemcore.clemgame import GameBenchmarkCallbackList, GameInstances, GameBenchmark
from clemcore.clemgame.envs.pettingzoo.master import GameMasterEnv
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from clemcore.clemgame.player import Player

module_logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("clemcore.run")

# Type alias for branching condition callable
BranchingCondition = Callable[..., bool]


def is_player_role(game_role: str) -> BranchingCondition:
    """
    Create a branching condition that triggers when a specific player role is active.

    Args:
        game_role: The role name to branch on, e.g., "WordDescriber"

    Returns:
        A callable that returns True when the specified role is active

    Example:
        condition = player_role_condition("Describer")
    """

    def condition(player: 'Player' = None, **_) -> bool:
        return player is not None and hasattr(player, 'game_role') and player.game_role == game_role

    return condition


def is_player_model(model: Model) -> BranchingCondition:
    """
    Create a branching condition that triggers when a specific model is active.

    Args:
        model: The model to branch on

    Returns:
        A callable that returns True when the specified model is active

    Example:
        condition = player_model_condition(learner_model)
    """

    def condition(player: 'Player' = None, **_) -> bool:
        return player is not None and hasattr(player, 'model') and player.model is model

    return condition


def is_round(round_number: int) -> BranchingCondition:
    """
    Create a branching condition that triggers at a specific round.

    Args:
        round_number: The round number to branch on (0-indexed)

    Returns:
        A callable that returns True at the specified round

    Example:
        condition = round_condition(0)  # Branch only at first round
    """

    def condition(env: GameMasterEnv = None, **_) -> bool:
        gm = env.game_master if env is not None else None
        return gm is not None and hasattr(gm, 'current_round') and gm.current_round == round_number

    return condition


def combined_condition(*conditions: BranchingCondition) -> BranchingCondition:
    """
    Combine multiple branching conditions with AND logic.

    Args:
        *conditions: Variable number of branching conditions

    Returns:
        A callable that returns True only when all conditions are True

    Example:
        condition = combined_condition(
            is_player_role("Describer"),
            is_round(0)
        )
    """

    def condition(player: 'Player', env: GameMasterEnv) -> bool:
        return all(cond(player=player, env=env) for cond in conditions)

    return condition


def run(game_benchmark: GameBenchmark,
        game_instances: GameInstances,
        player_models: List[Model],
        *,
        callbacks: GameBenchmarkCallbackList,
        branching_factor: int = 1,
        branching_condition: Optional[BranchingCondition] = None
        ):
    """
    Run game instances with optional branching to explore multiple trajectories.

    This function executes games where, at certain decision points determined by the
    branching_condition, the game state is copied and multiple different responses
    are explored in parallel. This creates a tree of game trajectories.

    Args:
        game_benchmark: The game benchmark configuration
        game_instances: Game instances to play (from instances.json)
        player_models: List of player models to use
        callbacks: Callbacks for benchmark lifecycle events
        branching_factor: Number of branches to create when condition is met.
            - 1 = no branching (default, standard gameplay)
            - 2+ = create this many parallel branches at each branching point
        branching_condition: A callable(player, env, context) -> bool that determines
            when to branch. If None, no branching occurs.
            - player: The Player object (access to model, role, etc.)
            - env: The GameMasterEnv (access to game state, round, etc.)
            Returns True to trigger branching at this step.

    Returns:
        None. Results are saved via callbacks.

    Example:
        # Branch when the Describer role is active
        run(..., branching_factor=3, branching_condition=is_player_role("Describer"))

        # Branch when a specific model is active
        run(..., branching_factor=2, branching_condition=is_player_model(learner))

        # Branch only at first round
        run(..., branching_factor=3, branching_condition=is_round(0))

        # Combine conditions: branch when learner model is active AND it's the first round
        run(..., branching_factor=2, branching_condition=combined_condition(
            is_player_model(learner),
            is_round(0)
        ))

        # Custom condition using lambda
        run(...,  branching_factor=2, branching_condition=lambda player, **_: player.game_role == "Guesser")

    Notes:
        - With branching_factor=N and M branching points, you get N^M leaf branches
        - Each leaf branch is a complete, independent game trajectory
        - Branching creates deep copies of the game state, so branches don't interfere
        - Use **_ in lambda conditions to ignore unused parameters
    """
    callbacks.on_benchmark_start(game_benchmark)
    error_count = 0
    progress_bar = tqdm(game_instances, desc="Playing game instances")
    for row in progress_bar:
        try:
            game_env = GameMasterEnv(game_benchmark, callbacks=callbacks)
            game_env.reset(options={
                "player_models": player_models,
                "experiment": row["experiment"],
                "game_instance": row["game_instance"]
            })
            for model in player_models:
                model.reset()
            runner = BranchingRunner(game_env, branching_factor, branching_condition, progress_bar=progress_bar)
            runner.run()
        except Exception:
            message = f"{game_benchmark.game_name}: Exception for instance {row['game_instance']['game_id']} (but continue)"
            module_logger.exception(message)
            error_count += 1
            for model in player_models:
                model.reset()
    if error_count > 0:
        stdout_logger.error(
            f"{game_benchmark.game_name}: '{error_count}' exceptions occurred: See clembench.log for details.")
    callbacks.on_benchmark_end(game_benchmark)


class BranchingRunner:
    """
    Manages the execution of a branching game episode.

    This runner maintains a list of active game environments and processes them
    in parallel, creating new branches when the branching condition is met.

    The runner uses a breadth-first approach:
    1. Start with the root game environment
    2. For each active environment:
       - Check if branching should occur
       - Create copies if branching
       - Execute the next step for each branch
    3. Continue until all branches complete

    Attributes:
        _root: The initial game environment
        branching_factor: Number of branches to create when condition is met
        branching_condition: Callable that determines when to branch
        _current_envs: List of currently active game environments

    Example:
        runner = BranchingRunner(game_env, branching_factor=3, branching_condition=is_round(0))
        runner.run()
        # Results in 3 independent game trajectories
    """

    def __init__(
            self,
            game_env: GameMasterEnv,
            branching_factor: int = 1,
            branching_condition: Optional[BranchingCondition] = None,
            progress_bar=None
    ):
        self._root = game_env
        self.branching_factor = branching_factor
        self.branching_condition = branching_condition

        self._progress_bar = progress_bar
        self._current_envs: List[GameMasterEnv] = [self._root]

    def should_branch(self, game_env):
        """
        Determine if branching should occur at the current step.

        Args:
            game_env: The current game environment

        Returns:
            bool: True if branching should occur, False otherwise

        The method checks:
        1. branching_factor > 1 (branching is enabled)
        2. branching_condition is not None (a condition is specified)
        3. branching_condition returns True for the current player/env
        """
        agent_id = game_env.agent_selection
        if agent_id is None:
            return False  # Don't branch terminal envs
        if game_env.terminations.get(agent_id) or game_env.truncations.get(agent_id):
            return False  # Don't branch on cleanup steps
        player = game_env.player_by_agent_id[agent_id]
        return (
                self.branching_factor > 1 and
                self.branching_condition is not None and
                self.branching_condition(player=player, env=game_env)
        )

    def run(self):
        while self._current_envs:  # As long as we have remaining game envs to be played ...
            if self._progress_bar is not None:
                self._progress_bar.set_postfix(branches=len(self._current_envs))
            remaining_envs = []
            for game_env in self._current_envs:  # ... we iterate over all of them
                if self.should_branch(game_env):  # ... and branch for each game env if necessary
                    branch_envs = [deepcopy(game_env) for _ in range(self.branching_factor)]
                else:
                    branch_envs = [deepcopy(game_env)]  # Note: Still copy to keep single nodes immutable
                continued_branches = self._single_step_all(branch_envs)
                remaining_envs.extend(continued_branches)
            self._current_envs = remaining_envs
        if self._progress_bar is not None:
            self._progress_bar.set_postfix(branches=0)

    def _single_step_all(self, branch_envs):
        continued_branches = []
        for branch_env in branch_envs:
            agent_id = branch_env.agent_selection
            if agent_id is None:  # This was a terminal branch (we can safely ignore it)
                branch_env.close()
                continue
            context, reward, termination, truncation, info = branch_env.last(observe=True)
            if termination or truncation:
                # None actions remove the agent from the game during step(None)
                # This is essential to observe the final reward, e.g., for the describer, when the guesser wins
                response = None
            else:
                player = branch_env.player_by_agent_id[agent_id]
                response = player(context)
            branch_env.step(response)
            # If we made it to here, then the branch is to be continued (agent_id was not None)
            continued_branches.append(branch_env)
        return continued_branches
