import abc
from typing import Dict, Any, List

from tqdm import tqdm

from clemcore.playpen.envs.game_env import GameEnv


class BaseCallback(abc.ABC):

    def __init__(self):
        self.locals: Dict[str, Any] = {}

    def on_rollout_start(self, game_env: GameEnv, num_timesteps: int):
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
        if "done" not in self.locals:
            return False
        return self.locals["done"]


class CallbackList(BaseCallback):

    def __init__(self, callbacks: List[BaseCallback] = None):
        super().__init__()
        if callbacks is None:
            callbacks = []
        self.callbacks = callbacks

    def append(self, callback: BaseCallback):
        self.callbacks.append(callback)

    def on_rollout_start(self, game_env: GameEnv, num_timesteps: int):
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

    def __init__(self, results_dir="playpen"):
        super().__init__()
        self.game_env = None
        self.rollout_idx = 0
        self.num_rollout_steps = 0
        self.episode_start_step = 0
        self.episode_idx = 0
        self.results_dir = results_dir

    def on_rollout_start(self, game_env: GameEnv, num_timesteps: int):
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
            self.store_records(self.game_env)
            self.episode_idx += (self.num_rollout_steps - self.episode_start_step)
            self.episode_start_step = self.num_rollout_steps

    def store_records(self, game_env: GameEnv):
        """
        Stores the records in a similar structure as for running clembench, so that transcribe can be applied.
        """
        # Note: Cannot use underscores in experiment dir, except the experiment name e.g. 0_high_en, because the
        # transcribe logic splits on underscore to get the experiment name, i.e., everything after the first underscore
        experiment_dir = f"rollout{self.rollout_idx:04d}-{game_env.experiment['index']}_{game_env.experiment['name']}"
        game_env.store_experiment_config(experiment_dir, self.results_dir)

        episode_dir = f"{experiment_dir}/episode_{self.episode_idx}"
        game_env.store_game_instance(episode_dir, self.results_dir)
        game_env.store_game_interactions(episode_dir, self.results_dir)


class RolloutProgressCallback(BaseCallback):

    def __init__(self, rollout_steps: int):
        super().__init__()
        self.rollout_steps = rollout_steps
        self.progress_bar = None

    def on_rollout_start(self, game_env: GameEnv, num_timesteps: int):
        self.progress_bar = tqdm(total=self.rollout_steps, desc="Rollout steps collected")

    def on_rollout_end(self):
        self.progress_bar.refresh()
        self.progress_bar.close()

    def on_step(self):
        if self.is_learning_player():
            self.progress_bar.update(1)
