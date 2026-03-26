import abc
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, TYPE_CHECKING, Dict

if TYPE_CHECKING:  # to satisfy pycharm
    from clemcore.clemgame import GameMaster, GameBenchmark, GameState


@dataclass
class GameStep:
    context: dict
    response: str
    done: bool = False
    info: dict = field(default_factory=dict)
    player_name: str | None = None
    model_name: str | None = None


@dataclass(frozen=True)
class GameSnapshot:
    state: "GameState" = field(compare=False)
    timestamp: datetime = field(default_factory=datetime.now, compare=False)
    origin: uuid.UUID = field(default_factory=uuid.uuid4)

    def __str__(self):
        return f"{self.origin}@{self.timestamp.strftime('%Y%m%d-%H%M%S-%f')}"

    @classmethod
    def create_from(cls, game_master: "GameMaster") -> "GameSnapshot":
        return cls(state=deepcopy(game_master.state))


class GameBenchmarkCallback(abc.ABC):

    def on_benchmark_start(self, game_benchmark: "GameBenchmark"):
        pass

    def on_game_start(self, game_master: "GameMaster", game_instance: Dict):
        pass

    def on_branching_point(self, game_master: "GameMaster", game_instance: Dict, snapshot: GameSnapshot):
        pass

    def on_game_step(self, game_master: "GameMaster", game_instance: Dict, game_step: GameStep):
        pass

    def on_game_end(self, game_master: "GameMaster", game_instance: Dict,
                    exception: Exception = None, rewards: dict[str, float] = None):
        """Called when a game episode ends, whether normally or due to an unexpected exception.

        If exception is None, the episode completed normally. If exception is set, the episode
        was aborted by an error. Implementors that only handle normal completion should guard
        with ``if exception is not None: return`` at the top of their implementation.
        """
        pass

    def on_benchmark_end(self, game_benchmark: "GameBenchmark"):
        pass


class GameBenchmarkCallbackList(GameBenchmarkCallback):

    def __init__(self, callbacks: List[GameBenchmarkCallback] = None):
        super().__init__()
        if callbacks is None:
            callbacks = []
        self.callbacks = callbacks

    def append(self, callback: GameBenchmarkCallback):
        self.callbacks.append(callback)

    def on_benchmark_start(self, game_benchmark: "GameBenchmark"):
        for callback in self.callbacks:
            callback.on_benchmark_start(game_benchmark)

    def on_game_start(self, game_master: "GameMaster", game_instance: Dict):
        for callback in self.callbacks:
            callback.on_game_start(game_master, game_instance)

    def on_branching_point(self, game_master: "GameMaster", game_instance: Dict, snapshot: GameSnapshot):
        for callback in self.callbacks:
            callback.on_branching_point(game_master, game_instance, snapshot)

    def on_game_step(self, game_master: "GameMaster", game_instance: Dict, game_step: GameStep):
        for callback in self.callbacks:
            callback.on_game_step(game_master, game_instance, game_step)

    def on_game_end(self, game_master: "GameMaster", game_instance: Dict,
                    exception: Exception = None, rewards: dict[str, float] = None):
        for callback in self.callbacks:
            callback.on_game_end(game_master, game_instance, exception, rewards)

    def on_benchmark_end(self, game_benchmark: "GameBenchmark"):
        for callback in self.callbacks:
            callback.on_benchmark_end(game_benchmark)
