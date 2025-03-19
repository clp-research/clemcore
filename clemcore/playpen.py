import abc
from contextlib import contextmanager
from typing import List, Dict, Any

from tqdm import tqdm

from clemcore.backends import Model
from clemcore.clemgame import GameRegistry, GameSpec, GameBenchmark
from clemcore.clemgame import benchmark


class BaseCallback(abc.ABC):

    def __init__(self):
        self.locals: Dict[str, Any] = {}

    def on_rollout_start(self, game_env: "GameEnv", num_timesteps: int):
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

    def on_rollout_start(self, game_env: "GameEnv", num_timesteps: int):
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

    def on_rollout_start(self, game_env: "GameEnv", num_timesteps: int):
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

    def store_records(self, game_env: "GameEnv"):
        """
        Stores the records in a similar structure as for running clembench, so that transcribe can be applied.
        """
        experiment_config = game_env.experiment_config
        dialogue_pair_desc = game_env.game.get_dialogue_pair_descriptor(game_env.players)
        # Note: Cannot use underscores in experiment dir, except the experiment name e.g. 0_high_en, because the
        # transcribe logic splits on underscore to get the experiment name, i.e., everything after the first underscore
        experiment_dir = f"rollout{self.rollout_idx:04d}-{experiment_config['dir']}"
        game_env.game.store_results_file(experiment_config,
                                         f"experiment_{experiment_config['name']}.json",
                                         dialogue_pair_desc,
                                         sub_dir=experiment_dir,
                                         results_dir=self.results_dir)
        episode_dir = f"{experiment_dir}/episode_{self.episode_idx}"
        game_env.game.store_results_file(game_env.game_instance,
                                         f"instance.json",
                                         dialogue_pair_desc,
                                         sub_dir=episode_dir,
                                         results_dir=self.results_dir)
        game_env.master.store_records(self.results_dir, dialogue_pair_desc, episode_dir)


class RolloutProgressCallback(BaseCallback):

    def __init__(self, rollout_steps: int):
        super().__init__()
        self.rollout_steps = rollout_steps
        self.progress_bar = None

    def on_rollout_start(self, game_env: "GameEnv", num_timesteps: int):
        self.progress_bar = tqdm(total=self.rollout_steps, desc="Rollout steps collected")

    def on_rollout_end(self):
        self.progress_bar.refresh()
        self.progress_bar.close()

    def on_step(self):
        if self.is_learning_player():
            self.progress_bar.update(1)


class BasePlayPen(abc.ABC):

    def __init__(self, learner: Model, teacher: Model):
        self.learner = learner
        self.teacher = teacher
        self.rollout_buffer = []
        self.num_timesteps = 0
        self.callbacks = CallbackList()

    def add_callback(self, callback: BaseCallback):
        self.callbacks.append(callback)

    @abc.abstractmethod
    def learn_interactive(self, game_registry: GameRegistry):
        pass

    def _collect_rollouts(self, game_env, rollout_steps):
        # reset() sets up the next game instance;
        # we should notify somehow when all instances were run so users can intervene if wanted?
        self.callbacks.on_rollout_start(game_env, self.num_timesteps)
        num_rollout_steps = 0
        while num_rollout_steps < rollout_steps:
            # like: agent, observation, action, reward, done, info
            player, context, response, feedback, done, info = game_env.step()
            if self.is_learning_player(player):
                self.rollout_buffer.append((context, response, feedback, done, info))
                num_rollout_steps += 1
                self.num_timesteps += 1
            self.callbacks.update_locals(locals())
            self.callbacks.on_step()
            if done:
                game_env.reset()
        self.callbacks.on_rollout_end()

    def is_learning_player(self, player):
        return player.model is self.learner


@contextmanager
def make_env(game_spec: GameSpec, players: List[Model],
             instances_name: str = None, shuffle_instances: bool = False):
    with benchmark.load_from_spec(game_spec, do_setup=True, instances_name=instances_name) as game:
        yield GameEnv(game, players=players, shuffle_instances=shuffle_instances)


class GameEnv:

    def __init__(self, game: GameBenchmark, players: List[Model], shuffle_instances: bool = False):
        self.game = game
        self.players = players
        # setup iterator to go through tasks / game instances
        self.task_iterator = game.create_game_instance_iterator(shuffle_instances)
        if len(self.task_iterator) < 1:
            raise RuntimeError(f"No game instances given for the game: '{self.game.game_name}'")
        # variables initialized on reset()
        self.game_instance = None
        self.experiment_config = None
        self.master = None
        # reset here so that game env is fully functional after init
        self.reset()

    def reset(self):
        try:
            self.experiment_config, self.game_instance = next(self.task_iterator)
            self.master = self.game.create_game_master(self.experiment_config, self.players)
            self.master.setup(**self.game_instance)
        except StopIteration:
            self.task_iterator.reset()
            self.reset()

    def step(self):
        return self.master.step()
