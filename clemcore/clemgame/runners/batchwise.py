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
    # Wraps a single game master instance producing observations

    def __init__(self, session_id: int, game_master: GameMaster, game_instance: Dict):
        self.session_id = session_id
        self.game_master = game_master
        self.game_instance = game_instance

    def __iter__(self):
        if self.game_master.is_done():
            return
        player, context = self.game_master.observe()
        yield self.session_id, player, context


class MultiGameRoundRobinScheduler(IterableDataset):
    """
    An IterableDataset that yields at most one item from each active (non-exhausted)
    dataset in a single round-robin pass. Once a dataset is exhausted, it is
    permanently skipped in all future iterations.

    Assumes that the sub-datasets are *stateful*, meaning calling `iter(d)` multiple times
    continues from where it last left off.

    - Each __iter__ call performs a new round.
    - The datasets themselves must manage their internal state.
    - Exhausted datasets are permanently skipped once they raise StopIteration.
    """

    def __init__(self, game_sessions: List[GameSession]):
        self.game_sessions = game_sessions
        self.exhausted = [False] * len(game_sessions)

    def __iter__(self):
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


def generate_batch_response_per_model(session_ids: List[int],
                                      players: List[Player],
                                      contexts: List[Dict]
                                      ) -> Dict[int, str]:
    """
    Generates responses in batches by grouping players according to their model backend.

    Each player processes a context to form a prompt (via `perceive()`), and responses are
    generated in batches using their respective models. Responses are then routed back to
    the originating players via `update_perspective_with()`.

    Args:
        session_ids: A list of session identifiers (parallel to players and contexts).
        players: A list of Player instances.
        contexts: A list of context dictionaries (one per player).

    Returns:
        A dictionary mapping session IDs to the textual responses generated.
    """
    # Index models by name (since Model objects may not be hashable)
    model_by_name = {player.model.name: player.model for player in players}

    # Group inputs by model, tracking (session_id, player, perspective)
    input_batch_by_model: Dict[str, List[Tuple[int, Player, List[Dict]]]] = defaultdict(list)
    for session_id, player, context in zip(session_ids, players, contexts):
        perspective = player.perceive_context(context)
        input_batch_by_model[player.model.name].append((session_id, player, perspective))

    # Collect responses per session_id
    response_by_session_id = {}
    for model_name, batched_inputs in input_batch_by_model.items():
        # Build input batch and track mapping back to session/player
        session_map: Dict[int, Tuple[int, Player]] = {}
        batched_perspectives: List[List[Dict]] = []
        for prompt_idx, (session_id, player, prompt) in enumerate(batched_inputs):
            session_map[prompt_idx] = (session_id, player)
            batched_perspectives.append(prompt)

        # Run batched generation (assumes order-preserving)
        model = model_by_name[model_name]
        results = model.generate_batch_response(batched_perspectives)
        assert len(results) == len(batched_perspectives), "Mismatching batch sizes in model's batched response"

        # Each result is assumed to be (prompt, response_object, response_text)
        for prompt_idx, (perspective, response_object, response_text) in enumerate(results):
            session_id, player = session_map[prompt_idx]
            response_by_session_id[session_id] = response_text
            metadata = dict(prompt=perspective, response_object=response_object)
            player.perceive_response(response_text, metadata=metadata)
    return response_by_session_id


def support_batching(model: Model):
    return hasattr(model, 'generate_batch_response') and callable(getattr(model, 'generate_batch_response'))


def run(game_benchmark: GameBenchmark,
        player_models: List[Model],
        *,
        callbacks: GameBenchmarkCallbackList,
        batch_size: int | str):
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
    round_robin_scheduler = MultiGameRoundRobinScheduler(game_sessions)
    data_loader = DataLoader(round_robin_scheduler, batch_size=8)
    for batch in data_loader:
        session_ids, batch_players, batch_contexts = batch
        response_by_session_id = generate_batch_response_per_model(session_ids, batch_players, batch_contexts)
        # Use session_ids to map outputs back to game sessions for stepping
        for sid, response in response_by_session_id.items():
            session = game_sessions[sid]  # assuming session_id is an index (see __prepare_game_sessions)
            done, _ = session.game_master.step(response)
            if done:
                callbacks.on_game_end(session.game_master, session.game_instance)
