import logging
from typing import List, Callable, Optional, TYPE_CHECKING
from copy import deepcopy

from tqdm import tqdm

from clemcore.backends import Model
from clemcore.clemgame import GameBenchmarkCallbackList, GameInstances, GameBenchmark
from clemcore.clemgame.envs.pettingzoo.master import GameMasterEnv

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
    def condition(player: 'Player', env: GameMasterEnv, context: any) -> bool:
        return all(cond(player=player, env=env, context=context) for cond in conditions)
    return condition


class GameBranch:
    """Represents a single branch of gameplay."""
    
    def __init__(self, branch_id: int, game_env: GameMasterEnv, player_models: List[Model]):
        self.branch_id = branch_id
        self.game_env = game_env
        self.player_models = player_models
        self.completed = False
        self.error = None


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
            # Initialize the first branch
            game_env = GameMasterEnv(game_benchmark, callbacks=callbacks)
            game_env.reset(options={
                "player_models": player_models,
                "experiment": row["experiment"],
                "game_instance": row["game_instance"]
            })
            
            for model in player_models:
                model.reset()
            
            # Track active branches
            branches = [GameBranch(0, game_env, player_models)]
            branch_counter = 1
            has_branched = False
            
            # Play all branches
            while branches:
                # Process each active branch
                new_branches = []
                
                for branch in branches:
                    if branch.completed:
                        continue
                    
                    try:
                        # Get next agent in this branch
                        agent_iter = branch.game_env.agent_iter()
                        
                        for agent_id in agent_iter:
                            context, reward, termination, truncation, info = branch.game_env.last(observe=True)
                            
                            if termination or truncation:
                                response = None
                                branch.game_env.step(response)
                                branch.completed = True
                                break
                            
                            player = branch.game_env.player_by_agent_id[agent_id]
                            
                            # Check if we should branch at this step
                            should_branch = (
                                not has_branched and 
                                branching_factor > 1 and 
                                branching_condition is not None and
                                branching_condition(player=player, env=branch.game_env, context=context)
                            )
                            
                            if should_branch:
                                # Create additional branches
                                for i in range(branching_factor - 1):
                                    # Create a new game environment for the branch
                                    new_env = GameMasterEnv(game_benchmark, callbacks=callbacks)
                                    new_models = [deepcopy(model) for model in branch.player_models]
                                    
                                    # Reset with same configuration
                                    new_env.reset(options={
                                        "player_models": new_models,
                                        "experiment": row["experiment"],
                                        "game_instance": row["game_instance"]
                                    })
                                    
                                    # TODO: Replay history to reach current state
                                    # This is a simplified version - full implementation would need
                                    # to replay all previous actions to reach the current game state
                                    
                                    new_branch = GameBranch(branch_counter, new_env, new_models)
                                    new_branches.append(new_branch)
                                    branch_counter += 1
                                
                                has_branched = True
                                module_logger.info(
                                    f"Branched instance {row['game_instance']['game_id']} "
                                    f"into {branching_factor} branches at agent {agent_id}"
                                )
                            
                            # Generate response for current branch
                            response = player(context)
                            branch.game_env.step(response)
                            
                    except Exception as e:
                        branch.error = e
                        branch.completed = True
                        module_logger.exception(
                            f"Exception in branch {branch.branch_id} of instance "
                            f"{row['game_instance']['game_id']}"
                        )
                        error_count += 1
                
                # Add new branches to active branches
                branches.extend(new_branches)
                
                # Remove completed branches
                branches = [b for b in branches if not b.completed]
            
            # Clean up
            for branch in branches:
                branch.game_env.close()
            
            # Reset models after all branches complete
            for model in player_models:
                model.reset()
                
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
