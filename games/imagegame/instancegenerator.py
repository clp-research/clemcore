"""
Randomly generate templates for the private/shared game.

Creates files in ./instances and ./requests
"""

from random import randint

import random

import clemgame
from clemgame.clemgame import GameInstanceGenerator

random.seed(123)
N_INSTANCES = 10

logger = clemgame.get_logger(__name__)
GAME_NAME = "imagegame"

def generate_random_grid(number_of_letters, grid_dimension, fill_row=False, fill_column=False):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    # create an empty grid
    grid = []
    for i in range(0, grid_dimension):

        row = []
        for j in range(0, grid_dimension):
            row.append('▢')
        grid.append(row)

    # randomly initialize the grid
    for i in range(0, min(grid_dimension * grid_dimension, number_of_letters)):
        ## select random row and column location in the grid
        random_row_index = randint(0, grid_dimension - 1)
        random_column_index = randint(0, grid_dimension - 1)

        random_letter_index = randint(0, len(alphabet) - 1)
        random_letter = alphabet[random_letter_index]

        grid[random_row_index][random_column_index] = random_letter

    # select a random row and fill it with a random letter
    if fill_row:
        random_row_index = randint(0, grid_dimension - 1)
        random_letter_index = randint(0, len(alphabet) - 1)
        random_letter = alphabet[random_letter_index]

        row = []
        for i in range(0, grid_dimension):
            row.append(random_letter)

        grid[random_row_index] = row

    if fill_column:
        random_column_index = randint(0, grid_dimension - 1)
        random_letter_index = randint(0, len(alphabet) - 1)
        random_letter = alphabet[random_letter_index]

        for i in range(0, grid_dimension):

            for j in range(0, grid_dimension):
                if j == random_column_index:
                    grid[i][j] = random_letter

    grid_as_string = ''
    for g_row in grid:
        row = ''
        for cell in g_row:
            row += cell + ' '
        grid_as_string += row.strip() + '\n'
    return grid_as_string.strip()

def generate_random_grid(number_of_letters, grid_dimension, letter):
    # create an empty grid
    grid = []
    for i in range(0, grid_dimension):

        row = []
        for j in range(0, grid_dimension):
            row.append('▢')
        grid.append(row)

    for i in range(0, min(grid_dimension * grid_dimension, number_of_letters)):
        ## select random row and column location in the grid
        random_row_index = randint(0, grid_dimension - 1)
        random_column_index = randint(0, grid_dimension - 1)
        grid[random_row_index][random_column_index] = letter


    grid_as_string = ''
    for g_row in grid:
        row = ''
        for cell in g_row:
            row += cell + ' '
        grid_as_string += row.strip() + '\n'
    return grid_as_string.strip()

class ImageGameInstanceGenerator(GameInstanceGenerator):

    def __init__(self):
        super().__init__(GAME_NAME)

    def on_generate(self):

        player_a_prompt_header = self.load_template(f"resources/initial_prompts/player_a_prompt_header.template")
        player_b_prompt_header = self.load_template(f"resources/initial_prompts/player_b_prompt_header.template")
        prompt_question = self.load_template(f"resources/initial_prompts/prompt_question.template")
        initial_grids = self.load_json("resources/grids_v1_5.json")

        compact_grids = []
        random_grids = []
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        for grid in initial_grids:
            random_letter_index = randint(0, len(alphabet) - 1)
            random_letter = alphabet[random_letter_index]

            # at most 10 cells to be filled
            letter_count = min(grid.count('X'), 10)
            # pick a random number to decide on the number of letters to put in the random grid
            random_grid_letter_count = randint(5, letter_count)

            random_grid = generate_random_grid(random_grid_letter_count, 5, random_letter)

            grid = grid.replace('X', random_letter)
            alphabet = alphabet.replace(random_letter, '')

            compact_grids.append(grid)
            random_grids.append(random_grid)

        generated_grids = {'compact_grids': compact_grids, 'random_grids': random_grids}

        for grid_name in generated_grids:

            experiment = self.add_experiment(grid_name)
            grid_dimension = 5

            for grid_index in range(0, len(generated_grids[grid_name])):

                grid = generated_grids[grid_name][grid_index]

                game_instance = self.add_game_instance(experiment, grid_index)

                game_instance["player_1_prompt_header"] = player_a_prompt_header.replace('GRID_DIMENSION', str(grid_dimension) + ' by ' + str(grid_dimension))
                game_instance["player_2_prompt_header"] = player_b_prompt_header.replace('GRID_DIMENSION', str(grid_dimension) + ' by ' + str(grid_dimension))
                game_instance["player_1_question"] = prompt_question
                game_instance['grid_dimension'] = grid_dimension
                game_instance['number_of_letters'] = grid.count('X')
                game_instance['player_1_response_pattern'] = '^instruction: [^\n]+$'
                game_instance['player_1_terminate_pattern'] = '^instruction:\s*(?i)done\b'
                game_instance['player_2_response_pattern'] = '^\n*([A-Z▢]\s){4}[A-Z▢]\n([A-Z▢]\s){4}[A-Z▢]\n([A-Z▢]\s){4}[A-Z▢]\n([A-Z▢]\s){4}[A-Z▢]\n([A-Z▢]\s){4}[A-Z▢]\n*$'
                game_instance['fill_row'] = False
                game_instance['fill_column'] = False
                game_instance['target_grid'] = grid


if __name__ == '__main__':
    ImageGameInstanceGenerator().generate()
