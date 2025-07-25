import logging
from typing import List, Dict, Callable, Optional, Tuple, Any, Iterable

import torch
from tqdm import tqdm

from clemcore.backends import Model, ModelSpec
from clemcore.backends.model_registry import BatchGenerativeModel
from clemcore.clemgame import GameBenchmark, GameBenchmarkCallbackList, GameMaster, Player

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


def auto_estimate_batch_size(player_model: BatchGenerativeModel,
                             *,
                             upper_limit: int,
                             get_batch_inputs_fn: Callable[[int], List[List[Dict]]]) -> int:
    """
    Estimates the largest batch size that can be used with the given player model
    without triggering a CUDA out-of-memory (OOM) error.

    The function performs a binary-like search starting from `upper_limit`, halving the batch size
    on each OOM failure until a size is found that fits in available GPU memory. The batch inputs
    are generated by calling `get_batch_inputs_fn(batch_size)`.

    Args:
        player_model: The model to test for batch generation capacity. Must implement
                              `generate_batch_response(batch_inputs)`.
        upper_limit: The initial (maximum) batch size to try.
        get_batch_inputs_fn: A function that, given a batch size, returns input data

    Returns:
        int: The largest batch size that can be processed without causing an out-of-memory error.

    Raises:
        RuntimeError: If even batch size 1 cannot be processed due to memory constraints.
    """
    batch_size = upper_limit  # Start with max possible size
    while batch_size > 0:
        try:
            batch_inputs = get_batch_inputs_fn(batch_size)
            player_model.generate_batch_response(batch_inputs)
            return batch_size
        except torch.cuda.OutOfMemoryError:
            stdout_logger.info("Cannot fit batch_size=%s for model=%s", batch_size, player_model.name)
            torch.cuda.empty_cache()
            batch_size //= 2

    # If we get here, even batch size 1 failed
    raise RuntimeError("Cannot fit any batch_size for model=%s. "
                       "Model or data is too large for available GPU memory.",
                       player_model.name)


def run(game_benchmark: GameBenchmark,
        player_models: List[BatchGenerativeModel],
        *,
        callbacks: GameBenchmarkCallbackList):
    """
    Executes a batchwise evaluation of the given game benchmark using one or more player models.

    This function handles:
    - Validating that all player models support batch inference.
    - Determining or estimating the optimal batch size for each model.
    - Preparing game sessions for evaluation.
    - Runs the game sessions, stepping through their progress using a round-robin scheduler.
    - Invokes callbacks on benchmark start/end and on game start/end.

    Batch Size Behavior:
        - If `batch_size` is explicitly set in a model's spec, it is used as-is.
        - If not set (and only one model is present), the batch size is estimated via `estimate_batch_size`.
        - The lowest batch size across all models is selected to ensure compatibility.

    Note:
        Batch size estimation uses the current benchmark's data to probe GPU memory capacity.
        If the dataset is small, this may underestimate the optimal batch size for larger benchmarks.
        You can preset the batch size in the model_spec or use the unification mechanism, for example,
        passing a model spec to the run command like -m '{"model_name": "llama3-8b", "batch_size": 8}'.

    Args:
        game_benchmark: The GameBenchmark to run.
        player_models: List of player models participating in the benchmark.
        callbacks: Callback list to notify about benchmark and game events.

    Raises:
        AssertionError: If any model does not support batching or batch_size is invalid.
    """
    # If not all support batching, then this doesn't help, because the models have to wait for the slowest one
    assert Model.all_support_batching(player_models), \
        "Not all player models support batching. Use sequential runner."

    # Determine batch size for the run
    batch_sizes = []
    for player_model in player_models:
        model_batch_size = getattr(player_model.model_spec, "batch_size", None)
        if model_batch_size is not None:
            stdout_logger.info("Found batch_size=%s for model=%s", model_batch_size, player_model.name)
            batch_sizes.append(model_batch_size)
            continue

        # Estimate batch size if not specified using initial contexts of the games
        def __load_initial_contexts(_batch_size: int) -> List[List[Dict]]:
            initial_game_sessions = __prepare_game_sessions(game_benchmark, player_models)
            single_pass_poller = SinglePassGameSessionPoller(initial_game_sessions)
            data_loader = DynamicBatchDataLoader(
                single_pass_poller,
                collate_fn=GameSession.collate_fn,
                batch_size=_batch_size
            )
            batch = next(iter(data_loader))
            _, batch_players, batch_contexts = batch
            if player_model.model_spec.is_programmatic():  # hack to let the mock backend access the player states
                player_model.set_gen_arg("players", batch_players)
            # Now, this is only a list of dicts (the contexts), but  to invoke generate_batch_response, we need
            # proper lists of lists (messages) as used by the Player (messages = memorized perspective + context)
            batch_messages = [[context] for context in batch_contexts]
            # Note: Giving the initial context only likely results in an overestimate of the batch_size!
            return batch_messages

        stdout_logger.info("Estimate batch_size for model=%s", player_model.name)
        all_game_sessions = __prepare_game_sessions(game_benchmark, player_models)
        model_batch_size = auto_estimate_batch_size(player_model,
                                                    upper_limit=len(all_game_sessions),
                                                    get_batch_inputs_fn=__load_initial_contexts)

        # Update model spec with the estimated batch size for consistency
        player_model.model_spec = player_model.model_spec.unify(ModelSpec(batch_size=model_batch_size))
        stdout_logger.info("Found batch_size=%s for model=%s", model_batch_size, player_model.name)
        batch_sizes.append(model_batch_size)

    # Use the smallest batch size to ensure all models can run simultaneously without issue
    batch_size = min(batch_sizes)
    if len(player_models) > 1:  # notify only when more than one model; otherwise already mentioned above
        stdout_logger.info("Use minimal batch_size=%s of %s", batch_size, batch_sizes)

    # Begin benchmark run with callbacks
    callbacks.on_benchmark_start(game_benchmark)
    game_sessions = __prepare_game_sessions(game_benchmark, player_models, callbacks, verbose=True)
    num_sessions = len(game_sessions)
    if batch_size > num_sessions:
        stdout_logger.info("Reduce batch_size=%s to number of game sessions %s", batch_size, num_sessions)
    __run_game_sessions(game_sessions, callbacks, min(batch_size, num_sessions))
    callbacks.on_benchmark_end(game_benchmark)


def __prepare_game_sessions(game_benchmark: GameBenchmark,
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
    # Note: We must reset iterator here, otherwise it is already exhausted after single invocation of prepare.
    instance_iterator = game_benchmark.game_instance_iterator
    instance_iterator.reset(verbose=verbose)
    if verbose:
        pbar = tqdm(total=len(instance_iterator), desc="Setup game instances", dynamic_ncols=True)
    for session_id, (experiment, game_instance) in enumerate(instance_iterator):
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
        response_by_session_id = Player.batch_response(batch_players, batch_contexts, row_ids=session_ids)

        # Use session_ids to map outputs back to game sessions for stepping
        for sid, response in response_by_session_id.items():
            session = game_sessions[sid]  # assuming session_id is an index (see __prepare_game_sessions)
            done, _ = session.game_master.step(response)
            pbar_responses.update(1)
            if done:
                pbar_instances.update(1)
                callbacks.on_game_end(session.game_master, session.game_instance)
    pbar_instances.close()
    pbar_responses.close()
    pbar_batches.close()
