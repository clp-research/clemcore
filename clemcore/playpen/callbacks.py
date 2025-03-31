import abc
from typing import Dict, Any, List

from tqdm import tqdm

from clemcore.playpen.envs import PlayPenEnv


class BaseCallback(abc.ABC):

    def __init__(self):
        self.locals: Dict[str, Any] = {}

    def on_rollout_start(self, game_env: PlayPenEnv, num_timesteps: int):
        pass

    def on_rollout_end(self):
        pass

    def on_training_start(self):
        pass

    def on_training_end(self):
        pass

    @abc.abstractmethod
    def on_step(self):
        pass

    def update_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        self.locals.update(locals_)

    def is_learning_player(self):
        if "player" not in self.locals:
            return False
        if "self" not in self.locals:
            return False
        player = self.locals["player"]
        other_self = self.locals["self"]
        learner = other_self.learner
        return player.model is learner

    def is_done(self):
        assert "game_env" in self.locals, "There must be an env in the callback locals to determine the terminal state."
        game_env: PlayPenEnv = self.locals["game_env"]
        return game_env.is_done()


class CallbackList(BaseCallback):

    def __init__(self, callbacks: List[BaseCallback] = None):
        super().__init__()
        if callbacks is None:
            callbacks = []
        self.callbacks = callbacks

    def append(self, callback: BaseCallback):
        self.callbacks.append(callback)

    def on_rollout_start(self, game_env: PlayPenEnv, num_timesteps: int):
        for callback in self.callbacks:
            callback.on_rollout_start(game_env, num_timesteps)

    def on_rollout_end(self):
        for callback in self.callbacks:
            callback.on_rollout_end()

    def on_training_start(self):
        for callback in self.callbacks:
            callback.on_training_start()

    def on_training_end(self):
        for callback in self.callbacks:
            callback.on_training_end()

    def on_step(self):
        for callback in self.callbacks:
            callback.on_step()

    def update_locals(self, locals_: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback.update_locals(locals_)


class GameRecordCallback(BaseCallback):
    """
    Stores the game records after each episode for inspection.
    The records can be transcribed into an HTML readable format.
    """

    def __init__(self, top_dir="playpen"):
        super().__init__()
        self.game_env = None
        self.rollout_idx = 0
        self.num_rollout_steps = 0
        self.episode_start_step = 0
        self.episode_idx = 0
        self.top_dir = top_dir

    def on_rollout_start(self, game_env: PlayPenEnv, num_timesteps: int):
        self.game_env = game_env
        self.rollout_idx += 1
        self.num_rollout_steps = 0
        self.episode_start_step = 0
        self.episode_idx = num_timesteps

    def on_step(self):
        # Note: It might happen that the learner's turn is not reached because the teacher already
        # screws up and the game ends early. Then the rollout steps do not increase and the stored
        # records are eventually overwritten by the next episode where the learner has a turn.
        # todo: Do we actually want to record these episodes as well?
        if self.is_learning_player():
            self.num_rollout_steps += 1
        if self.is_done():
            rollout_dir = f"rollout{self.rollout_idx:04d}"
            episode_dir = f"episode_{self.episode_idx}"
            self.game_env.store_records(self.top_dir, rollout_dir, episode_dir)
            self.episode_idx += (self.num_rollout_steps - self.episode_start_step)
            self.episode_start_step = self.num_rollout_steps


class RolloutProgressCallback(BaseCallback):

    def __init__(self, rollout_steps: int):
        super().__init__()
        self.rollout_steps = rollout_steps
        self.progress_bar = None

    def on_rollout_start(self, game_env: PlayPenEnv, num_timesteps: int):
        self.progress_bar = tqdm(total=self.rollout_steps, desc="Rollout steps collected")

    def on_rollout_end(self):
        self.progress_bar.refresh()
        self.progress_bar.close()

    def on_step(self):
        if self.is_learning_player():
            self.progress_bar.update(1)
