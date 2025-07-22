import logging
from collections import defaultdict
from typing import List, Dict, Tuple

from tqdm import tqdm

from clemcore.backends import Model
from clemcore.clemgame import GameBenchmark, GameBenchmarkCallbackList, GameMaster, Player

module_logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("clemcore.run")

from torch.utils.data import IterableDataset, DataLoader


class GameSession(IterableDataset):
    """
    Wraps a single game master instance producing observations as an iterable dataset.

    Each iteration yields a single observation tuple consisting of:
    - session_id: int, unique identifier of this game session
    - player: Player instance observed at this step
    - context: Dict representing the current context or game state from the GameMaster

    Iteration ends when the game master signals completion via `is_done()`.
    """

    def __init__(self, session_id: int, game_master: GameMaster, game_instance: Dict):
        """
        Initialize a game session wrapper.

        Args:
            session_id: Unique identifier for the session.
            game_master: The GameMaster instance managing the game logic.
            game_instance: The dictionary containing the game instance configuration/state.
        """
        self.session_id = session_id
        self.game_master = game_master
        self.game_instance = game_instance

    def __iter__(self):
        """
        Yield the current observation (session_id, player, context) once if not done.

        Yields:
            Tuple[int, Player, Dict]: session id, player, and context data.
        """
        if self.game_master.is_done():
            return
        player, context = self.game_master.observe()
        yield self.session_id, player, context


class MultiGameRoundRobinScheduler(IterableDataset):
    """
    IterableDataset that iterates over multiple GameSession datasets in a round-robin manner.

    - On each iteration, yields at most one item from each active (non-exhausted) session.
    - Once a session dataset is exhausted (raises StopIteration), it is permanently skipped.
    - Assumes each GameSession is stateful: calling iter() multiple times resumes where it left off.
    - Useful for fairly interleaving game observations across multiple concurrent sessions.
    """

    def __init__(self, game_sessions: List[GameSession]):
        """
        Initialize the round-robin scheduler.

        Args:
            game_sessions: List of GameSession instances to interleave.
        """
        self.game_sessions = game_sessions
        self.exhausted = [False] * len(game_sessions)

    def __iter__(self):
        """
        Iterate once over all active sessions, yielding one observation per session if available.

        Yields:
            Tuple[int, Player, Dict]: session id, player, and context from each active session.
        """
        if all(self.exhausted):
            return

        for i, session in enumerate(self.game_sessions):
            if self.exhausted[i]:
                continue
            try:
                it = iter(session)
                yield next(it)
            except StopIteration:
                self.exhausted[i] = True


def support_batching(model: Model):
    """
    Check if the given model supports batch generation of responses.

    Args:
        model: Model instance to check.

    Returns:
        bool: True if the model implements `generate_batch_response` as a callable, False otherwise.
    """
    return hasattr(model, 'generate_batch_response') and callable(getattr(model, 'generate_batch_response'))


def run(game_benchmark: GameBenchmark,
        player_models: List[Model],
        *,
        callbacks: GameBenchmarkCallbackList,
        batch_size: int | str):
    """
    Run the game benchmark with the provided player models, invoking callbacks at key events.

    This function:
    - Validates that all player models support batching.
    - Prepares game sessions for each instance in the benchmark.
    - Runs the game sessions, stepping through their progress using a round-robin scheduler.
    - Invokes callbacks on benchmark start/end and on game start/end.

    Args:
        game_benchmark: The GameBenchmark object to run.
        player_models: List of player models participating in the benchmark.
        callbacks: Callback list to notify about benchmark and game events.
        batch_size: Batch size to use for processing game observations;
                    can be a positive integer or "auto".

    Raises:
        AssertionError: If any model does not support batching or batch_size is invalid.
    """
    # If not all support batching, then this doesn't help, because the models have to wait for the slowest one
    assert all(support_batching(player_model) for player_model in player_models), \
        "Not all player models support batching"
    assert batch_size > 0 or batch_size == "auto", f"batch_size must be >0 or 'auto' but is {batch_size}"

    callbacks.on_benchmark_start(game_benchmark)
    game_sessions = __prepare_game_sessions(game_benchmark, player_models, callbacks)
    if len(game_sessions) == 0:
        message = f"{game_benchmark.game_name}: Could not prepare any game session. See clembench.log for details."
        stdout_logger.warning(message)
    __run_game_sessions(game_sessions)
    callbacks.on_benchmark_end(game_benchmark)


def __prepare_game_sessions(game_benchmark: GameBenchmark,
                            player_models: List[Model],
                            callbacks: GameBenchmarkCallbackList):
    """
    Prepare GameSession instances for each game instance in the benchmark.

    Iterates over the game instances, creating GameMaster objects and
    corresponding GameSession wrappers.

    Logs and counts exceptions, continuing with remaining instances on failure.

    Args:
        game_benchmark: The GameBenchmark providing game instances.
        player_models: List of player models to pass to the GameMaster.
        callbacks: Callback list to notify on game start.

    Returns:
        List[GameSession]: The list of prepared game sessions.
    """
    error_count = 0
    game_sessions: List[GameSession] = []
    for session_id, experiment, game_instance in enumerate(tqdm(game_benchmark.game_instance_iterator,
                                                                desc="Prepare game sessions")):
        try:
            game_master = game_benchmark.create_game_master(experiment, player_models)
            callbacks.on_game_start(game_master, game_instance)
            game_master.setup(**game_instance)
            game_sessions.append(GameSession(session_id, game_master, game_instance))
        except Exception:  # continue with other instances if something goes wrong
            message = f"{game_benchmark.game_name}: Exception for instance {game_instance['game_id']} (but continue)"
            module_logger.exception(message)
            error_count += 1
    if error_count > 0:
        message = f"{game_benchmark.game_name}: '{error_count}' exceptions occurred: See clembench.log for details."
        stdout_logger.error(message)
    return game_sessions


def __run_game_sessions(game_sessions: List[GameSession], callbacks: GameBenchmarkCallbackList):
    """
    Run multiple game sessions concurrently using a round-robin scheduler.

    Processes batches of game observations, invokes Player.batch_response to generate
    model responses, steps the GameMaster with responses, and notifies callbacks on game end.

    Args:
        game_sessions: List of active GameSession instances.
        callbacks: Callback list to notify on game end.
    """
    round_robin_scheduler = MultiGameRoundRobinScheduler(game_sessions)
    data_loader = DataLoader(round_robin_scheduler, batch_size=8)
    for batch in data_loader:
        session_ids, batch_players, batch_contexts = batch
        response_by_session_id = Player.batch_response(batch_players, batch_contexts, row_ids=session_ids)
        # Use session_ids to map outputs back to game sessions for stepping
        for sid, response in response_by_session_id.items():
            session = game_sessions[sid]  # assuming session_id is an index (see __prepare_game_sessions)
            done, _ = session.game_master.step(response)
            if done:
                callbacks.on_game_end(session.game_master, session.game_instance)
