import logging
from typing import List, Dict, Callable, Optional, Tuple, Any, Iterable

from tqdm import tqdm

from clemcore.backends import Model
from clemcore.backends.model_registry import BatchGenerativeModel
from clemcore.clemgame import (
    GameBenchmark,
    GameBenchmarkCallbackList,
    GameMaster,
    GameStep,
    Player,
    GameInstanceIterator
)

module_logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("clemcore.run")


class GameSession(Iterable):
    """
    Wraps a single game master instance producing observations as an iterable.

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

    @staticmethod
    def collate_fn(batch) -> Tuple[List[int], List[Player], List[Dict]]:
        """
        Collate a batch of (session_id, player, context) tuples into a tuple of lists.
        Returns:
            The session ids, players and contexts as separate lists
        """
        session_ids, players, contexts = zip(*batch)
        return list(session_ids), list(players), list(contexts)


class SinglePassGameSessionPoller(Iterable):
    """
    Iterable that yields one item (if available) from each provided GameSession in a single pass.

    - Iterates over each GameSession once, yielding at most one observation per session.
    - If a session is already exhausted (raises StopIteration), it is skipped.
    - Designed to collect a single snapshot from each session (e.g., initial moves or states).
    - Does NOT perform full round-robin scheduling or repeated cycling.
    """

    def __init__(self, game_sessions: List[GameSession]):
        """
        Initialize the session poller.

        Args:
            game_sessions: List of GameSession instances to sample from.
        """
        self.game_sessions = game_sessions
        self.exhausted = [False] * len(game_sessions)

    def __iter__(self):
        """
        Iterates over the game sessions and yields one observation from each, if available.

        For each GameSession:
        - Attempts to yield the next item using its iterator.
        - If the session is exhausted, it is marked as such and skipped.
        - Sessions are not revisited during this pass.

        Note:
            To poll multiple rounds of observations, this iterable must be re-instantiated.

        Yields:
            Tuple[int, Player, Dict]: A tuple containing:
                - the index of the session (int),
                - the Player object,
                - and a context dictionary (Dict) representing the next observation.
        """
        for i, session in enumerate(self.game_sessions):
            if self.exhausted[i]:
                continue
            try:
                it = iter(session)
                yield next(it)
            except StopIteration:
                self.exhausted[i] = True


class DynamicBatchDataLoader(Iterable):
    """
    A custom DataLoader for stateful IterableDatasets that supports dynamically shrinking batch sizes.

    This loader is designed for datasets where sources may be independently exhausted over time,
    such as multiple concurrently running environments or game sessions. It preserves the internal
    iterator state of the dataset and gracefully adjusts batch sizes as fewer data sources remain active.

    Key Features:
        - Supports stateful datasets (e.g., those that resume iteration across calls).
        - Adapts batch size dynamically: yields smaller batches as individual data sources are exhausted.
        - Compatible with custom collate functions for flexible batching.
        - Suitable for use cases like streaming rollouts (e.g., SinglePassGameSessionPoller).

    Note:
        The final batch in a polling round may be smaller than the specified `batch_size`, especially
        when few data sources remain. Fixing this to always yield full batches would require additional
        buffering and coordination logic, which is intentionally avoided here to keep the implementation simple.

    Unlike PyTorch's built-in DataLoader, this implementation:
        - Avoids resetting dataset iterators on each pass.
        - Handles partial batches naturally without requiring drop_last logic.
    """

    def __init__(self, dataset: Any, *, collate_fn: Callable, batch_size: int):
        """
         Initialize the dynamic batch loader.

         Args:
             dataset (IterableDataset): The dataset to draw items from. Must expose an `exhausted` attribute
                 (e.g., a list of booleans indicating which sub-datasets are still active).
             batch_size (int): Maximum number of items to include in each batch.
             collate_fn (Callable): Function used to merge a list of items into a single batch.
         """
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.data_iter = iter(self.dataset)

    def __iter__(self):
        data_iter = iter(self.dataset)
        while True:
            if all(self.dataset.exhausted):
                break
            batch_items = []
            try:
                for _ in range(self.batch_size):
                    item = next(data_iter)
                    batch_items.append(item)
            except StopIteration:
                # End of a pass or all sources exhausted; re-initialize for next round
                data_iter = iter(self.dataset)
            if batch_items:
                yield self.collate_fn(batch_items)


def run(game_benchmark: GameBenchmark,
        game_instance_iterator: GameInstanceIterator,
        player_models: List[BatchGenerativeModel],
        *,
        callbacks: GameBenchmarkCallbackList,
        batch_size: int):
    """
    Executes a batchwise evaluation of the given game benchmark using one or more player models.

    The runner plays as many games as returned by the game instance iterator (without resetting it).

    This function handles:
    - Validating that all player models support batch inference.
    - Preparing game sessions for evaluation.
    - Runs the game sessions, stepping through their progress using a round-robin scheduler.
    - Invokes callbacks on benchmark start/end and on game start/end.

    Args:
        game_benchmark: The GameBenchmark to run.
        game_instance_iterator: An iterator over the game instances to be played.
        player_models: List of player models participating in the benchmark.
        callbacks: Callback list to notify about benchmark and game events.
        batch_size: The batch size to use for all player models.

    Raises:
        AssertionError: If any model does not support batching.
    """
    # If not all support batching, then this doesn't help, because the models have to wait for the slowest one
    assert Model.all_support_batching(player_models), \
        "Not all player models support batching. Use the sequential runner instead."

    callbacks.on_benchmark_start(game_benchmark)
    game_sessions = __prepare_game_sessions(game_benchmark, game_instance_iterator, player_models, callbacks,
                                            verbose=True)
    num_sessions = len(game_sessions)
    if batch_size > num_sessions:
        stdout_logger.info("Reduce batch_size=%s to number of game sessions %s", batch_size, num_sessions)
    __run_game_sessions(game_sessions, callbacks, min(batch_size, num_sessions))
    callbacks.on_benchmark_end(game_benchmark)


def __prepare_game_sessions(game_benchmark: GameBenchmark,
                            game_instance_iterator: GameInstanceIterator,
                            player_models: List[BatchGenerativeModel],
                            callbacks: Optional[GameBenchmarkCallbackList] = None,
                            verbose: bool = False):
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

    Raises:
        RuntimeError: If not even a single game session could be prepared.
    """
    callbacks = callbacks or GameBenchmarkCallbackList()
    error_count = 0
    game_sessions: List[GameSession] = []
    if verbose:
        pbar = tqdm(total=len(game_instance_iterator), desc="Setup game instances", dynamic_ncols=True)
    for session_id, (experiment, game_instance) in enumerate(game_instance_iterator):
        try:
            game_master = game_benchmark.create_game_master(experiment, player_models)
            callbacks.on_game_start(game_master, game_instance)
            game_master.setup(**game_instance)
            game_sessions.append(GameSession(session_id, game_master, game_instance))
        except Exception:  # continue with other instances if something goes wrong
            message = f"{game_benchmark.game_name}: Exception for instance {game_instance['game_id']} (but continue)"
            module_logger.exception(message)
            error_count += 1
        if verbose:
            pbar.update(1)
    if verbose:
        pbar.close()
    if error_count > 0:
        message = f"{game_benchmark.game_name}: '{error_count}' exceptions occurred: See clembench.log for details."
        stdout_logger.error(message)
    if len(game_sessions) == 0:
        message = f"{game_benchmark.game_name}: Could not prepare any game sessions. See clembench.log for details."
        raise RuntimeError(message)
    return game_sessions


def __run_game_sessions(game_sessions: List[GameSession], callbacks: GameBenchmarkCallbackList, batch_size: int):
    """
    Run multiple game sessions concurrently using a round-robin scheduler.

    Processes batches of game observations, invokes Player.batch_response to generate
    model responses, steps the GameMaster with responses, and notifies callbacks on game end.

    Args:
        game_sessions: List of active GameSession instances.
        callbacks: Callback list to notify on game end.
    """
    # Progress bar for completed games (known total)
    pbar_instances = tqdm(total=len(game_sessions), desc="Completed game instances", dynamic_ncols=True)
    # Progress bar for total steps (unknown total, so no 'total' arg)
    pbar_responses = tqdm(desc="Total responses", unit="response", dynamic_ncols=True)
    # Progress bar for batch size (approaching one)
    pbar_batches = tqdm(bar_format="{desc}", dynamic_ncols=True)

    start_batch_size = batch_size
    batch_sizes = []

    def scaled_sparkline(values, *, max_val, min_val=1, levels="▁▁▂▂▃▃▄▄▅▅▆▆▇▇██"):
        span = max_val - min_val or 1
        return ''.join(
            levels[int((min(max(v, min_val), max_val) - min_val) / span * (len(levels) - 1))]
            for v in values
        )

    round_robin_scheduler = SinglePassGameSessionPoller(game_sessions)
    data_loader = DynamicBatchDataLoader(
        round_robin_scheduler,
        collate_fn=GameSession.collate_fn,
        batch_size=batch_size
    )
    for batch in data_loader:
        session_ids, batch_players, batch_contexts = batch

        # Display batch_size
        current_batch_size = len(session_ids)
        batch_sizes.append(current_batch_size)
        trend = scaled_sparkline(batch_sizes[-40:], max_val=start_batch_size)
        pbar_batches.set_description_str(
            f"Batch sizes: {trend} [start={start_batch_size}, "
            f"current={current_batch_size}, "
            f"mean={sum(batch_sizes) / len(batch_sizes):.2f}]"
        )
        pbar_batches.refresh()

        # Apply batch to receives responses
        context_response_by_session_id = Player.batch_response(batch_players, batch_contexts, row_ids=session_ids)

        # Use session_ids to map outputs back to game sessions for stepping
        for sid, (context, response) in context_response_by_session_id.items():
            session = game_sessions[sid]  # assuming session_id is an index (see __prepare_game_sessions)
            done, info = session.game_master.step(response)
            game_step = GameStep(context, response, done, info)
            callbacks.on_game_step(session.game_master, session.game_instance, game_step)
            pbar_responses.update(1)
            if done:
                pbar_instances.update(1)
                callbacks.on_game_end(session.game_master, session.game_instance)
    pbar_instances.close()
    pbar_responses.close()
    pbar_batches.close()
