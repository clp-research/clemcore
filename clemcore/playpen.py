import abc
from contextlib import contextmanager
from typing import List

from tqdm import tqdm

from clemcore.backends import Model
from clemcore.clemgame import GameRegistry, GameSpec, GameBenchmark
from clemcore.clemgame import benchmark


class BasePlayPen(abc.ABC):

    def __init__(self, learner: Model, teacher: Model, rollout_steps):
        self.learner = learner
        self.teacher = teacher
        self.rollout_steps = rollout_steps
        self.rollout_buffer = []
        self.num_timesteps = 0
        self.num_rollouts = 0

    @abc.abstractmethod
    def learn_interactive(self, game_registry: GameRegistry):
        pass

    def _collect_rollouts(self, game_env):
        # reset() sets up the next game instance;
        # we should notify somehow when all instances were run so users can intervene if wanted?
        self.num_rollouts += 1
        num_rollout_steps = 0
        episode_start_step = 0
        records_start_step = self.num_timesteps
        with tqdm(total=self.rollout_steps, desc="Rollout steps collected") as pbar:
            while num_rollout_steps < self.rollout_steps:
                # like: agent, observation, action, reward, done, info
                player, context, response, feedback, done, info = game_env.step()
                if player.model is self.learner:
                    self.rollout_buffer.append((context, response, feedback, done, info))
                    num_rollout_steps += 1
                    self.num_timesteps += 1
                    pbar.update(1)
                if done:
                    # Note: It might happen that the learner's turn is not reached because the teacher already
                    # screws up and the game ends early. Then the rollout steps do not increase and the stored
                    # records are eventually overwritten by the next episode where the learner has a turn.
                    # todo: Do we actually want to record these episodes as well?
                    game_env.store_records(self.num_rollouts, records_start_step)
                    game_env.reset()
                    records_start_step += (num_rollout_steps - episode_start_step)
                    episode_start_step = num_rollout_steps


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

    def store_records(self, rollout_idx, episode_idx):
        dialogue_pair_desc = self.game.get_dialogue_pair_descriptor(self.players)
        experiment_name = self.experiment_config['name']
        # Note: Cannot use underscores in experiment dir, except the experiment name e.g. 0_high_en, because the
        # transcribe logic splits on underscore to get the experiment name, i.e., everything after the first underscore
        experiment_dir = f"rollout{rollout_idx:04d}-{self.experiment_config['dir']}"
        self.game.store_results_file(self.experiment_config,
                                     f"experiment_{experiment_name}.json",
                                     dialogue_pair_desc,
                                     sub_dir=experiment_dir,
                                     results_dir="playpen")
        episode_dir = f"{experiment_dir}/episode_{episode_idx}"
        self.game.store_results_file(self.game_instance,
                                     f"instance.json",
                                     dialogue_pair_desc,
                                     sub_dir=episode_dir,
                                     results_dir="playpen")
        self.master.store_records("playpen", dialogue_pair_desc, episode_dir)
