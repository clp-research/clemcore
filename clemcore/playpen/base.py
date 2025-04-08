import abc

from clemcore.backends import Model
from clemcore.clemgame import GameRegistry
from clemcore.playpen.envs import PlayPenEnv
from clemcore.playpen.buffers import RolloutBuffer
from clemcore.playpen.callbacks import CallbackList, BaseCallback


class BasePlayPen(abc.ABC):

    def __init__(self, learner: Model, teacher: Model):
        self.learner = learner
        self.teacher = teacher
        self.num_timesteps = 0
        self.callbacks = CallbackList()

    def add_callback(self, callback: BaseCallback):
        self.callbacks.append(callback)

    def _collect_rollouts(self, game_env: PlayPenEnv, rollout_steps: int, rollout_buffer: RolloutBuffer):
        # reset() sets up the next game instance;
        # we should notify somehow when all instances were run so users can intervene if wanted?
        self.callbacks.on_rollout_start(game_env, self.num_timesteps)
        rollout_buffer.initial_prompts = game_env.initial_prompts
        num_rollout_steps = 0
        while num_rollout_steps < rollout_steps:
            player, context = game_env.observe()
            response = player(context)
            done, info = game_env.step(response)
            num_rollout_steps += 1
            self.num_timesteps += 1
            rollout_buffer.on_step(context, response, done, info)
            self.callbacks.update_locals(locals())
            self.callbacks.on_step()
            if game_env.is_done():
                rollout_buffer.on_done()
                game_env.reset()
        self.callbacks.on_rollout_end()

    def is_learner(self, player):
        return player.model is self.learner

    def is_teacher(self, player):
        return player.model is self.teacher

    @abc.abstractmethod
    def learn_interactive(self, game_registry: GameRegistry):
        pass
