"""
Generate instances for the referencegame
Version 1.5 (multilingual instances)

Reads grids_v1.5.json from resources/
Creates instances_v1.5_{lang}.json in instances/
"""

import json
import Levenshtein # to calculate distance between grids

import clemgame
from clemgame.clemgame import GameInstanceGenerator

from games.referencegame.resources.localization_utils import MULTILINGUAL_PATTERNS

logger = clemgame.get_logger(__name__)
GAME_NAME = "referencegame"
GRIDS = "resources/grids_v1.5.json"


def generate_samples(grids):
    """
    Generate triplets from grids
    :param grids: list of string grid representations
    :return: list of triplets where the first is the target and the following two are distractors
    """
    samples = []
    # calculate edit distance between grids
    edit_distances = get_distances(grids)
    # select distractors with smallest edit distance
    for target_grid_id in range(len(grids)):
        second_grid_id, third_grid_id = select_distractors(target_grid_id, edit_distances)
        samples.append((grids[target_grid_id], grids[second_grid_id], grids[third_grid_id]))
    return samples


def get_distances(grids):
    """
    Calculate edit distances to select similar distractors
    :param grids: list of string grid representations
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
        self.lang = ""

    def on_generate(self, lang):
        """
        Create instances into self.instances
        (Called by super().generate())
        """
        self.lang = lang

        # load grids
        with open(GRIDS, 'r') as f:
            grids = json.load(f)

        # generate sub experiments
        for grids_group in grids.keys():
            # get triplets
            samples = generate_samples(grids[grids_group])

            player_a_prompt_header = self._load_prompt("player_a_prompt_header.template")
            player_b_prompt_header = self._load_prompt("player_b_prompt_header.template")

            experiment = self.add_experiment(f"{grids_group}")

            game_counter = 0
            for sample in samples:
                # create three instances from each triplet, where the target for player 2 is in
                # one of the three possible positions each (selecting one order for the other two)
                for i in [1, 2, 3]:
                    target_grid, second_grid, third_grid = sample

                    game_instance = self.add_game_instance(experiment, game_counter)
                    game_instance["player_1_prompt_header"] = player_a_prompt_header.replace('TARGET_GRID', target_grid) \
                                                                                    .replace('SECOND_GRID', second_grid) \
                                                                                    .replace('THIRD_GRID', third_grid)
                    game_instance['player_1_target_grid'] = target_grid
                    game_instance['player_1_second_grid'] = second_grid
                    game_instance['player_1_third_grid'] = third_grid

                    # create order of grids for player 2
                    # extract target grid names from localization_utils
                    targets = MULTILINGUAL_PATTERNS[self.lang]["p2_options"].split("|")
                    assert len(targets) == 3

                    first_grid = ""
                    target_grid_name = ""
                    if i == 1:
                        first_grid = target_grid
                        # keep order from player 1 for second and third grid
                        target_grid_name = targets[0] # corresponds to "first"
                    elif i == 2:
                        first_grid = second_grid
                        second_grid = target_grid
                        # third grid stays third grid
                        target_grid_name = targets[1] # corresponds to "second"
                    elif i == 3:
                        first_grid = third_grid
                        # second grid stays second grid
                        third_grid = target_grid
                        target_grid_name = targets[2] # corresponds to "third"

                    game_instance["player_2_prompt_header"] = player_b_prompt_header.replace('FIRST_GRID', first_grid) \
                                                                                    .replace('SECOND_GRID', second_grid) \
                                                                                    .replace('THIRD_GRID', third_grid)
                    game_instance['player_2_first_grid'] = first_grid
                    game_instance['player_2_second_grid'] = second_grid
                    game_instance['player_2_third_grid'] = third_grid
                    game_instance['target_grid_name'] = target_grid_name
                    game_instance['lang'] = self.lang

                    game_counter += 1

    def _load_prompt(self, template):
        """
        Load language specific prompt
        :param lang: language identifier string
        :param template: filename of prompt template
        :return: prompt string
        """
        with open(f"resources/initial_prompts/{self.lang}/{template}", encoding='utf8') as f:
            prompt = f.read()
        return prompt


if __name__ == '__main__':
    # generate language versions
    for language in MULTILINGUAL_PATTERNS.keys():
        ReferenceGameInstanceGenerator().generate(
            filename=f"instances_v1.5_{language}.json", lang=language)
