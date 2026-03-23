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
    Run game instances with optional branching.

    Args:
        game_benchmark: The game benchmark configuration
        game_instances: Instances to play
        player_models: List of player models
        callbacks: Callbacks for benchmark events
        branching_factor: Number of branches to create when condition is met (1 = no branching)
        branching_condition: A callable(player, env, context) -> bool that determines
            when to branch. If None, no branching occurs. The player parameter gives access
            to the Player object including its model.

    Example:
        # Branch when the Describer role is active
        run(..., branching_factor=3, branching_condition=player_role_condition("Describer"))

        # Branch when a specific model is active
        run(..., branching_factor=2, branching_condition=player_model_condition("gpt-4"))

        # Branch only at first round
        run(..., branching_factor=3, branching_condition=round_condition(1))

        # Combine conditions: branch when learner model is active AND it's the first round
        run(..., branching_factor=2, branching_condition=combined_condition(
            player_model_condition("learner-model"),
            round_condition(1)
        ))

        # Or use a lambda for custom conditions
        run(..., branching_factor=2, branching_condition=lambda player, **_: player.game_role == "GM")
    """
    callbacks.on_benchmark_start(game_benchmark)
    error_count = 0
    for row in tqdm(game_instances, desc="Playing game instances"):
        try:
            game_env = GameMasterEnv(game_benchmark, callbacks=callbacks)
            game_env.reset(options={
                "player_models": player_models,
                "experiment": row["experiment"],
                "game_instance": row["game_instance"]
            })
            for model in player_models:
                model.reset()
            runner = BranchingRunner(game_env, branching_factor, branching_condition)
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

    def __init__(
            self,
            game_env: GameMasterEnv,
            branching_factor: int = 1,
            branching_condition: Optional[BranchingCondition] = None
    ):
        self._root = game_env
        self.branching_factor = branching_factor
        self.branching_condition = branching_condition

        self._current_envs: List[GameMasterEnv] = [self._root]

    def should_branch(self, game_env):
        agent_id = game_env.agent_selection
        player = game_env.player_by_agent_id[agent_id]
        return (
                self.branching_factor > 1 and
                self.branching_condition is not None and
                self.branching_condition(player=player, env=game_env)
        )

    def run(self):
        while self._current_envs:  # As long as we have remaining game envs to be played ...
            remaining_envs = []
            for game_env in self._current_envs:  # ... we iterate over all of them
                if self.should_branch(game_env):  # ... and branch for each game env if necessary
                    branch_envs = [deepcopy(game_env) for _ in range(self.branching_factor - 1)]
                else:
                    branch_envs = [deepcopy(game_env)]  # Note: Still copy to keep single nodes immutable
                for branch_env in branch_envs:  # ... then we continue the branches, or the game_env itself
                    agent_id = branch_env.agent_selection  # Only select the next agent
                    if agent_id is None:  # When there is no agent left, the episode is done for this game env
                        branch_env.close()
                        continue
                    context, reward, termination, truncation, info = game_env.last(observe=True)
                    if termination or truncation:
                        # None actions remove the agent from the game during step(None)
                        # This is essential to observe the final reward, e.g., for the describer, when the guesser wins
                        response = None
                    else:
                        player = branch_env.player_by_agent_id[agent_id]
                        response = player(context)
                    branch_env.step(response)
                    # If we made it to here, then the branch is to be continued (agent_id was not None)
                    remaining_envs.append(branch_env)
            self._current_envs = remaining_envs
