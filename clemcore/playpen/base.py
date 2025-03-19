import abc
from clemcore.backends import Model
from clemcore.clemgame import GameRegistry
from clemcore.playpen.callbacks import CallbackList, BaseCallback


class BasePlayPen(abc.ABC):

    def __init__(self, learner: Model, teacher: Model):
        self.learner = learner
        self.teacher = teacher
        self.rollout_buffer = []
        self.num_timesteps = 0
        self.callbacks = CallbackList()

    def add_callback(self, callback: BaseCallback):
        self.callbacks.append(callback)

    def _collect_rollouts(self, game_env, rollout_steps):
        # reset() sets up the next game instance;
        # we should notify somehow when all instances were run so users can intervene if wanted?
        self.callbacks.on_rollout_start(game_env, self.num_timesteps)
        num_rollout_steps = 0
        while num_rollout_steps < rollout_steps:
            player, observation, state = game_env.get_observation()
            action = player(observation)
            done, info = game_env.step(action)
            if self.is_learning_player(player):
                self.rollout_buffer.append((state, observation, action, done, info))
                num_rollout_steps += 1
                self.num_timesteps += 1
            self.callbacks.update_locals(locals())
            self.callbacks.on_step()
            if done:
                game_env.reset()
        self.callbacks.on_rollout_end()

    def is_learning_player(self, player):
        return player.model is self.learner

    @abc.abstractmethod
    def learn_interactive(self, game_registry: GameRegistry):
        pass
