import abc
import random
from typing import Tuple, Dict, List

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

    @abc.abstractmethod
    def learn_interactive(self, game_registry: GameRegistry):
        pass

    def _collect_rollouts(self, game_env):
        # reset() sets up the next game instance;
        # we should notify somehow when all instances were run so users can intervene if wanted?
        num_rollout_steps = 0
        num_timesteps_start = self.num_timesteps
        while num_rollout_steps < self.rollout_steps:
            # like: agent, observation, action, reward, done, info
            player, context, response, feedback, done, info = game_env.step()
            if player == self.learner:
                self.rollout_buffer.append((context, response, feedback, done, info))
                num_rollout_steps += 1
                self.num_timesteps += 1
            if done:
                game_env.store_records(num_timesteps_start)
                game_env.reset()
                num_timesteps_start += num_rollout_steps


def make_env(game_spec: GameSpec, players: List[Model],
             instances_name: str = None, shuffle_instances: bool = False):
    game = benchmark.load_from_spec(game_spec, do_setup=True, instances_name=instances_name)
    return GameEnv(game, players=players, shuffle_instances=shuffle_instances)


class GameInstanceIterator:

    def __init__(self, instances, do_shuffle=False):
        assert instances is not None, "Instances must be given"
        self.instances = instances
        self.do_shuffle = do_shuffle
        self.queue = []

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[Dict, Dict]:
        try:
            return self.queue.pop()
        except IndexError:
            raise StopIteration()

    def __len__(self):
        return len(self.queue)

    def clone(self):
        return GameInstanceIterator(self.instances, do_shuffle=self.do_shuffle)

    def reset(self):
        self.queue = []
        for idx, experiment in enumerate(self.instances):
            experiment_config = {k: experiment[k] for k in experiment if k != 'game_instances'}
            experiment_config["dir"] = f"{idx}_{experiment_config['name']}"
            for game_instance in experiment["game_instances"]:
                self.queue.append((experiment_config, game_instance))
        if self.do_shuffle:
            random.shuffle(self.queue)


class GameEnv:

    def __init__(self, game: GameBenchmark, players: List[Model],
                 shuffle_instances: bool = False):
        self.game = game
        self.players = players
        self.master = None
        self.experiment_config = None
        self.game_instance = None
        self.game_instances = GameInstanceIterator(game.instances, do_shuffle=shuffle_instances)
        self.reset()  # fully functional after init

    def reset(self):
        try:
            self.experiment_config, self.game_instance = next(self.game_instances)
            self.master = self.game.create_game_master(self.experiment_config, self.players)
            self.master.setup(**self.game_instance)
        except StopIteration:
            if len(self.game_instances) < 1:
                raise RuntimeError(f"No game instances given for the game: '{self.game.game_name}'")
            self.game_instances.reset()
            self.reset()

    def step(self):
        return self.master.step()

    def store_records(self, episode_idx):
        dialogue_pair_desc = self.game.get_dialogue_pair_descriptor(self.players)
        experiment_name = self.experiment_config['name']
        experiment_dir = self.experiment_config["dir"]
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
