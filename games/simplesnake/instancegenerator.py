import os
import random
import logging
from clemcore.clemgame import GameInstanceGenerator

MAX_TURNS = 20
N_INSTANCES = 10
GAME_NAME = 'simplesnake'
SEED = -413

logger = logging.getLogger(__name__)

LANGUAGE = "en"


class SimpleSnakeInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        super().__init__(os.path.dirname(__file__))

    def on_generate(self, variant=None):
        matrices = self.load_file('resources/matrices.txt').strip('\n').split('\n')

        for matrix in matrices:
            # create new experiment
            experiment_name = matrix
            if variant is not None:
                experiment_name = f'{experiment_name}_with{variant}'
            experiment = self.add_experiment(experiment_name)

            # store important game values
            dim = int(matrix[0])
            experiment['dim'] = dim
            experiment['max_turns'] = MAX_TURNS

            # load needed template resources
            prompt_path = 'resources/initial_prompts/navigator_prompt'
            if variant is not None:
                prompt_path = f'{prompt_path}_with_{variant}'
            experiment['navigator_initial_prompt'] = self.load_template(prompt_path)

            prompt_path = 'resources/reprompts/navigator_reprompt'
            experiment['navigator_reprompt'] = self.load_template(prompt_path)

            prompt_path = 'resources/patterns/navigator_response'
            if variant == 'planning':
                prompt_path = f'{prompt_path}_with_{variant}'
            experiment['navigator_response_pattern'] = self.load_template(prompt_path)

            instances = []
            for game_id in range(N_INSTANCES):
                instance = self.add_game_instance(experiment, game_id)

                snake_start_loc = random.choice(range(dim * dim))
                instance['snake_start_loc'] = snake_start_loc

                possible_prey_locs = list(range(dim * dim))
                possible_prey_locs.remove(snake_start_loc)
                prey_start_loc = random.choice(possible_prey_locs)
                instance['prey_start_loc'] = prey_start_loc

                # add obstacle information
                if variant == 'obstacles':
                    possible_obs_locs = list(range(dim * dim))
                    possible_obs_locs = possible_obs_locs[dim+1:-dim]

                    # obs cannot exist on outsider border of grid
                    possible_obs_locs = [loc for loc in possible_obs_locs if loc % dim != 0 and (loc+1) % dim != 0]

                    # obs cannot exist where other entities are
                    if snake_start_loc in possible_obs_locs:
                        possible_obs_locs.remove(snake_start_loc)
                    if prey_start_loc in possible_obs_locs:
                        possible_obs_locs.remove(prey_start_loc)

                    # pick up to two obstacles
                    obstacle_locs = []
                    for _ in range(3):  # 3 is the max number of obstacles that doesn't risk being unplayable
                        if len(possible_obs_locs) > 0:
                            obstacle_locs.append(random.choice(possible_obs_locs))
                            possible_obs_locs.remove(obstacle_locs[-1])
                        else:
                            break

                    instance['obstacle_locs'] = obstacle_locs

                instances.append(instance)

            experiment['game_instances'] = instances

    def generate(self, variant, filename):
        self.on_generate(variant=variant)
        self.store_file(self.instances, filename, sub_dir="in")


if __name__ == '__main__':
    random.seed(SEED)
    SimpleSnakeInstanceGenerator().generate(None, 'instances.json')
    SimpleSnakeInstanceGenerator().generate('obstacles', 'instances_withobstacles.json')
    SimpleSnakeInstanceGenerator().generate('planning', 'instances_withplanning.json')
