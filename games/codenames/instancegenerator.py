"""
Generate instances for the game.

Creates files in ./in
"""
from tqdm import tqdm
from clemgame.clemgame import GameInstanceGenerator
import random, copy, argparse
from typing import Set

from games.codenames.constants import *

def generate_random(wordlist, required):
    # sample words for the board
    total = required[TEAM] + required[OPPONENT] + required[INNOCENT] + required[ASSASSIN]
    board = random.sample(wordlist, total)

    # make the assignments for the cluegiver and remove instances from 'unsampled' that were already sampled
    unsampled = copy.copy(board)
    team_words = random.sample(unsampled, required[TEAM])
    unsampled = [word for word in unsampled if word not in team_words]
    opponent_words = random.sample(unsampled, required[OPPONENT])
    unsampled = [word for word in unsampled if word not in opponent_words]
    innocent_words = random.sample(unsampled, required[INNOCENT])
    unsampled = [word for word in unsampled if word not in innocent_words]
    assassin_words = random.sample(unsampled, required[ASSASSIN])
    unsampled = [word for word in unsampled if word not in assassin_words]
    assert len(unsampled) == 0, "Not all words have been assigned to a team!"
    return {
        BOARD: board,
        ASSIGNMENTS: {
            TEAM: team_words,
            OPPONENT: opponent_words,
            INNOCENT: innocent_words,
            ASSASSIN: assassin_words
        }
    }

def generate_similar_within_teams(categories, required):
    total = required[TEAM] + required[OPPONENT] + required[INNOCENT] + required[ASSASSIN]
    board = []
    already_taken_words = set()
    already_taken_categories = set()
    required = {"team": 9, "opponent": 8, "innocent": 7, "assassin": 1}
    assignments = {"team": [], "opponent": [], "innocent": [], "assassin": []}
    for alignment in assignments:
        while len(assignments[alignment]) < required[alignment]:
            remaining = required[alignment] - len(assignments[alignment])
            words = choose_instances_from_random_category(categories, already_taken_words, already_taken_categories, maximum = remaining)
            assignments[alignment].extend(list(words))
            board.extend(list(words))    
    
    # shuffle all alignments within, shuffle board 
    random.shuffle(board)
    for alignment in assignments:
        random.shuffle(assignments[alignment])
    return {"board": board, "assignments": assignments, "private": {"categories": already_taken_categories}}  
    
def choose_instances_from_random_category(categories: Set, already_taken_words: Set, already_taken_categories: Set, maximum = 4):
    remaining_category_names = set(categories.keys()) - already_taken_categories
    category_name = get_random_category(list(remaining_category_names))
    already_taken_categories.add(category_name)

    category_words = set(categories[category_name])
    remaining_words = category_words - already_taken_words
    
    # randomly choose 2-4 words from a category, so that not only one word slot remains
    choices = [2, 3, 4]
    for choice in choices:
        if maximum - choice == 1:
            choices.remove(choice)
            break
    amount = random.choice(choices)
    words = sample_words_from_category(list(remaining_words), min(amount, maximum))
    already_taken_words.update(words)
    return words
    
def sample_words_from_category(category, number_of_words):
    if len(category) < number_of_words:
        raise ValueError(f"The category (with length {len(category)}) does not contain the required amount of words ({number_of_words})!")
    words = []
    for i in range(number_of_words):
        word = random.choice(category)
        words.append(word)
        category.remove(word)
        
    return words
    
def get_random_category(category_list):
    return random.choice(category_list)

def generate_similar_across_teams(categories, required):
    total = required[TEAM] + required[OPPONENT] + required[INNOCENT] + required[ASSASSIN]
    pass

generators={'random': generate_random,
            'easy word assignments': generate_similar_within_teams,
            'difficult word assignments': generate_similar_across_teams}

class CodenamesInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        super().__init__(GAME_NAME)

    def generate(self, keep=False, variable_name=None, experiment_name=None, filename="instances.json"):
        # @overwrite
        self.on_generate(variable_name, experiment_name)
        if keep and (variable_name or experiment_name):
            print(f"Replacing instances for {variable_name}: {experiment_name}.")
            self.replace_instances(variable_name, experiment_name, filename)
        self.store_file(self.instances, filename, sub_dir="in")
        
    def on_generate(self, variable_name = None, experiment_name = None):
        # read experiment config file
        experiment_config = self.load_json("resources/experiments.json")
        defaults = experiment_config["default"]
        variable_experiments = experiment_config["variables"]
        variable_names = variable_experiments.keys()

        if variable_name:
            if variable_name not in variable_names:
                print(f"Variable name {variable_name} not found in experiment config file (only {', '.join(list(variable_names))}).")
                return
            # if the variable_name was set (correctly), we will only generate instances for this experiment suite
            print(f"(Re-)Generate only instances for experiments on {variable_name}.")
            variable_names = [variable_name]
            # otherwise instances for all variables are generated

        for variable_name in variable_names:
            print("Generating instances for variable: ", variable_name)
            experiments = variable_experiments[variable_name]["experiments"]
            experiment_names = experiments.keys()
            if experiment_name:
                if experiment_name not in experiment_names:
                    print(f"Experiment name {experiment_name} not found in experiment config file for {variable_name} (only {', '.join(list(experiment_names))}).")
                    return
                # if the experiment name was set (correctly), we will only generate instances for this specific experiment
                print(f"(Re-)Generate only instances for {experiment_name}.")
                experiment_names = [experiment_name]
                # otherwise instances for all experiments changing this variable are generated

            for experiment_name in experiment_names:
                print("Generating instances for experiment: ", experiment_name)
                experiment = self.add_experiment(experiment_name)
                experiment["variable"] = variable_name
                # set default parameters
                for parameter in defaults:
                    experiment[parameter] = defaults[parameter]
                # set experiment-specific parameters
                for parameter in experiments[experiment_name]:
                    print("Setting experiment parameter: ", parameter)
                    experiment[parameter] = experiments[experiment_name][parameter]

                # load correct wordlist
                wordlist_name = experiment["wordlist"]
                wordlist = self.load_json(f"resources/word_lists/{wordlist_name}")["words"]
                            
                # create game instances
                for game_id in tqdm(range(experiment["number of instances"])):
                    # choose correct generator function
                    assignments = experiment[ASSIGNMENTS]
                    generator = generators[experiment["generator"]]
                    instance = generator(wordlist, assignments)
                    self.test_instance_format(instance, assignments)

                    # Create a game instance
                    game_instance = self.add_game_instance(experiment, game_id)
                    # Add game parameters
                    game_instance[BOARD] = instance[BOARD]
                    game_instance[ASSIGNMENTS] = instance[ASSIGNMENTS]
            
    def test_instance_format(self, board_instance, params):
        # board_instance = {BOARD: [...],
        #                   ASSIGNMENTS: {TEAM: [...], OPPONENT: [...], INNOCENT: [...], ASSASSIN: [...]}}
        
        keys = [TEAM, OPPONENT, INNOCENT, ASSASSIN]
        assert set(params.keys()) == set(keys), f"The params dictionary is missing a key, keys are {params.keys()}, but should be {keys}!"
        
        if not BOARD in board_instance:
            raise KeyError(f"The key '{BOARD}' was not found in the board instance.")
        if not ASSIGNMENTS in board_instance:
            raise KeyError(f"The key '{ASSIGNMENTS}' was not found in the board instance.")
        
        for alignment in params.keys():
            if alignment == TOTAL:
                continue
            if len(board_instance[ASSIGNMENTS][alignment]) != params[alignment]:
                raise ValueError(f"The number of {alignment} on the board ({len(board_instance[ASSIGNMENTS][alignment])}) is unequal to the required number of {alignment} words ({params[alignment]})")
        
        if len(board_instance[BOARD]) != params[TEAM] + params[OPPONENT] + params[INNOCENT] + params[ASSASSIN]:
            raise ValueError(f"The sum of all assignments does not match the total number of words!")
            
        assigned_words = [x for y in board_instance[ASSIGNMENTS] for x in board_instance[ASSIGNMENTS][y]]
        if set(board_instance[BOARD]) != set(assigned_words):
            raise ValueError(f"The words on the board do not match all the assigned words.")
        
    def replace_instances(self, variable_name, experiment_name = None, filename="instances.json", ):
        file = self.load_json(f"in/{filename}")
        if not file:
            print("File does not exist, can be 'overwritten'...")
            return
        old_experiments = file["experiments"]
        # adding all new experiment instances
        new_experiments = self.instances["experiments"]
        for i in range(len(old_experiments)):
            if old_experiments[i]["variable"] == variable_name:
                if experiment_name and not old_experiments[i]["name"] == experiment_name:
                    # experiment name was set, but these old instances belong to a different experiment, so should be kept
                    print(f"Keep {old_experiments[i]['name']}.")
                    new_experiments.append(old_experiments[i])
                else:
                    print(f"Replace {experiment_name}.")
            else:
                # if the variable name is not the same, then these instances should also be kept
                print(f"Keep {old_experiments[i]['name']}.")
                new_experiments.append(old_experiments[i])

        # sort experiments by variable, then by experiment name
        new_experiments.sort(key=lambda k: (k['variable'], k['name']))
        self.instances["experiments"] = new_experiments

if __name__ == '__main__':
    # The resulting instances.json is automatically saved to the "in" directory of the game folder
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--keep", help="Optional flag to keep already generated instances and only replace new instances that will be generated for a specific variable and/or experiment. Otherwise overwrite all old instances.", action="store_true")
    parser.add_argument("-v", "--variable-name", type=str, help="Optional argument to only (re-) generate instances for a specific experiment suite aka variable.")
    parser.add_argument("-e", "--experiment-name", type=str, help="Optional argument to only (re-) generate instances for a specific experiment (variable name must also be set!).")
    args = parser.parse_args()
    if args.experiment_name and not args.variable_name:
        print("Running a specific experiment requires both the experiment name (-e) and the variable name (-v)!")
    else:
        random.seed(SEED)
        CodenamesInstanceGenerator().generate(keep = args.keep, variable_name = args.variable_name, experiment_name = args.experiment_name)
