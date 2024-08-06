"""
Generate instances for the referencegame
Version 1.6 (strict regex parsing)

Reads grids_v1.5.json from resources/ (grids don't change in this version)
Creates instances.json in instances/
"""

import random # to create random grids
import Levenshtein # to calculate distance between grids

import clemgame
from clemgame.clemgame import GameInstanceGenerator

random.seed(123)

logger = clemgame.get_logger(__name__)
GAME_NAME = "referencegame"
GRIDS = "resources/grids_v1.5.json"


def generate_samples(grids_name, grids):
    """
    Generate triplets from grids
    :param grids_name: string identifier for grid list
    :param grids: list of string grid representations
    :return: list of triplets where the first is the target and the following two are distractors
    """
    samples = []
    if grids_name == "random_grids":
        grids = create_random_grids(grids)
    # calculate edit distance between grids
    edit_distances = get_distances(grids)
    # select distractors with smallest edit distance
    for target_grid_id in range(len(grids)):
        second_grid_id, third_grid_id = select_distractors(target_grid_id, edit_distances)
        samples.append((grids[target_grid_id], grids[second_grid_id], grids[third_grid_id]))
    return samples


def create_random_grids(grids, fills=10, num_grids=10):
    """
    Create random grids.
    :param grids: list with an empty grid representation to start from
    :param fills: number of Xs in random grid
    :param num_grids: number of grids to create
    :return: list of random grids
    """

    empty_grid = grids[0]
    random_grids = []

    while len(random_grids) != num_grids:
        x_positions = [i for i,char in enumerate(empty_grid) if char == "\u25a2"]
        random_x_positions = random.sample(x_positions, fills)
        # convert grid to list for indexing
        random_grid = list(empty_grid)
        for i in random_x_positions:
            random_grid[i] = "X"
        # convert grid back to string representation
        grid_string = "".join(random_grid)
        if grid_string not in random_grids:
            random_grids.append(grid_string)
    return list(random_grids)


def get_distances(grids):
    """
    Calculate edit distances to select similar distractors
    :param grids: list if string grid representations
    :return: matrix of edit distances with ids corresponding to grids (the full matrix is filled for easier access to distances per grid)
    """
    distances = []
    for i in range(len(grids)):
        i_distances = []
        for j in range(len(grids)):
            if i == j:
                i_distances.append(0)
            else:
                i_distances.append(Levenshtein.distance(grids[i], grids[j]))
        distances.append(i_distances)
    return distances


def select_distractors(target_grid: int, distances: list):
    """
    Select two most similar distractors for the given target
    :param target_grid: id of the target grid in corresponding distance matrix
    :param distances: matrix of distances between grid ids
    :return: ids of two distractor grids
    """
    id1 = None
    id2 = None
    min_distance = 1
    distance_list = distances[target_grid]
    found1 = False
    found2 = False
    while not found2:
        # iterate through list of distances,
        # increasing the distance with every iteration
        # to find the two lowest distances and return their ids
        if min_distance in distance_list:
            if not found1:
                # save id of lowest distance
                id1 = distance_list.index(min_distance)
                found1 = True
            elif id1 != distance_list.index(min_distance):
                # save id of next lowest distance if id1 is already found
                id2 = distance_list.index(min_distance)
                found2 = True
            elif min_distance in distance_list[id1 + 1:]:
                # or save next id of same lowest distance as id1 if it exists
                id2 = distance_list.index(min_distance, id1 + 1)
                found2 = True
            else:
                min_distance += 1
        else:
            min_distance += 1
    return id1, id2


class ReferenceGameInstanceGenerator(GameInstanceGenerator):

    def __init__(self):
        super().__init__(GAME_NAME)

    def on_generate(self):

        player_a_prompt_header = self.load_template(f"resources/initial_prompts/player_a_prompt_header_zero_shot.template")
        player_b_prompt_header = self.load_template(f"resources/initial_prompts/player_b_prompt_header_zero_shot.template")
        grids = self.load_json(GRIDS)

        for grids_group in grids.keys():
            # get triplets
            samples = generate_samples(grids_group, grids[grids_group])
            experiment = self.add_experiment(grids_group)

            game_counter = 0
            for sample in samples:
                # create three instances from each triplet, where the target for player 2 is in
                # one of the three possible positions each (selecting one order for the other two)
                for i in [1,2,3]:
                    target_grid, second_grid, third_grid = sample

                    game_instance = self.add_game_instance(experiment, game_counter)
                    game_instance["player_1_prompt_header"] = player_a_prompt_header.replace('TARGET_GRID', target_grid)\
                                                                                    .replace('SECOND_GRID', second_grid)\
                                                                                    .replace('THIRD_GRID', third_grid)
                    game_instance['player_1_target_grid'] = target_grid
                    game_instance['player_1_second_grid'] = second_grid
                    game_instance['player_1_third_grid'] = third_grid

                    first_grid = ""
                    target_grid_name = []
                    if i == 1:
                        first_grid = target_grid
                        # keep order from player 1 for second and third grid
                        target_grid_name = ["first", "1st", "1"]
                    elif i == 2:
                        first_grid = second_grid
                        second_grid = target_grid
                        # third grid stays third grid
                        target_grid_name = ["second", "2nd", "2"]
                    elif i == 3:
                        first_grid = third_grid
                        # second grid stays second grid
                        third_grid = target_grid
                        target_grid_name = ["third", "3rd", "3"]

                    game_instance["player_2_prompt_header"] = player_b_prompt_header.replace('FIRST_GRID', first_grid)\
                                                                                    .replace('SECOND_GRID', second_grid)\
                                                                                    .replace('THIRD_GRID', third_grid)
                    game_instance['player_2_first_grid'] = first_grid
                    game_instance['player_2_second_grid'] = second_grid
                    game_instance['player_2_third_grid'] = third_grid
                    game_instance['target_grid_name'] = target_grid_name
                    game_instance['player_1_response_pattern'] = '^expression:\s(?P<content>.+)\n*(?P<remainder>.*)'
                    # named groups:
                    # 'content' captures only the generated referring expression
                    # 'remainder' should be empty (if models followed the instructions)
                    game_instance['player_2_response_pattern'] = '^answer:\s(?P<content>first|second|third|1|2|3|1st|2nd|3rd)\n*(?P<remainder>.*)'
                    # 'content' can directly be compared to gold answer
                    # 'remainder' should be empty (if models followed the instructions)

                    # the following two fields are no longer required, but kept for backwards compatibility with previous instance versions
                    game_instance["player_1_response_tag"] = "expression:"
                    game_instance["player_2_response_tag"] = "answer:"

                    game_counter += 1


if __name__ == '__main__':
    ReferenceGameInstanceGenerator().generate(filename="instances.json")
