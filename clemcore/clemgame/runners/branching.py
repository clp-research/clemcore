import logging
from typing import List, Callable, Optional
from copy import deepcopy

from tqdm import tqdm

from clemcore.backends import Model
from clemcore.clemgame import GameBenchmarkCallbackList, GameInstances, GameBenchmark
from clemcore.clemgame.envs.pettingzoo.master import GameMasterEnv

module_logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("clemcore.run")


# Type alias for branching condition callable
BranchingCondition = Callable[[GameMasterEnv, str, any], bool]


def player_role_condition(role_name: str) -> BranchingCondition:
    """
    Create a branching condition that triggers when a specific player role is active.
    
    Args:
        role_name: The role name to branch on (e.g., "Describer", "GM")
        
    Returns:
        A callable that returns True when the specified role is active
        
    Example:
        condition = player_role_condition("Describer")
    """
    def condition(game_env: GameMasterEnv, agent_id: str, context: any) -> bool:
        player = game_env.player_by_agent_id.get(agent_id)
        if player is None:
            return False
        return hasattr(player, 'role') and player.role == role_name
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
        branching_condition: A callable(game_env, agent_id, context) -> bool that determines
            when to branch. If None, no branching occurs.
            
    Example:
        # Branch when the Describer role is active
        run(..., branching_factor=3, branching_condition=player_role_condition("Describer"))
        
        # Or use a lambda for simple conditions
        run(..., branching_factor=2, branching_condition=lambda env, agent, ctx: agent == "player_1")
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
                            
                            # Check if we should branch at this step
                            should_branch = (
                                not has_branched and 
                                branching_factor > 1 and 
                                branching_condition is not None and
                                branching_condition(branch.game_env, agent_id, context)
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
                            player = branch.game_env.player_by_agent_id[agent_id]
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
