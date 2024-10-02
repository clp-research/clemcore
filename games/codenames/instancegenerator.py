"""
Generate instances for the game.

Creates files in ./in
"""
from tqdm import tqdm
from clemgame.clemgame import GameInstanceGenerator
import random, copy, argparse, os
from typing import Set
from clemgame.file_utils import file_path

from games.codenames.constants import *

FILENAME = "instances.json"
GENEROUS_FILENAME = "generous_instances.json"
FLAGS = ["IGNORE RAMBLING", "IGNORE FALSE TARGETS OR GUESSES", "REPROMPT ON ERROR", "STRIP WORDS", "IGNORE NUMBER OF TARGETS"]

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

def shuffle_board(board):
    random.shuffle(board)

def shuffle_words_within_assignments(assignments):
    for alignment in assignments:
        random.shuffle(assignments[alignment])

def generate_similar_within_teams(categories, required):
    board = []
    already_taken_words = []
    already_taken_categories = []
    assignments = {"team": [], "opponent": [], "innocent": [], "assassin": []}
    for alignment in assignments:
        while len(assignments[alignment]) < required[alignment]:
            remaining = required[alignment] - len(assignments[alignment])
            words = choose_instances_from_random_category(categories, already_taken_words, already_taken_categories, maximum = remaining)
            assignments[alignment].extend(words)
            board.extend(words)
    
    shuffle_board(board)
    shuffle_words_within_assignments(assignments)
    return {"board": board, "assignments": assignments, "private": {"categories": already_taken_categories}}

def generate_similar_across_teams(categories, required):
    total = required[TEAM] + required[OPPONENT] + required[INNOCENT] + required[ASSASSIN]
    board = []
    already_taken_words = []
    already_taken_categories = []
    assignments = {"team": [], "opponent": [], "innocent": [], "assassin": []}
    while len(board) < total:
        remaining = total - len(board)
        words = choose_instances_from_random_category(categories, already_taken_words, already_taken_categories, maximum = remaining)
        # choose random assignments to distribute words across
        i = 0
        while i < len(words):
            remaining_assignments = [key for key in assignments.keys() if len(assignments[key]) < required[key]]
            assign_to = random.sample(remaining_assignments, min(len(remaining_assignments), len(words)))
            for alignment in assign_to:
                assignments[alignment].append(words[i])
                i += 1
                if i == len(words):
                    break     
        board.extend(words)
    shuffle_board(board)
    shuffle_words_within_assignments(assignments)
    return {"board": board, "assignments": assignments, "private": {"categories": already_taken_categories}}
    
def choose_instances_from_random_category(categories: Set, already_taken_words: Set, already_taken_categories: Set, maximum = 4):
    #total = sum([len(categories[category]) for category in categories if category in remaining_category_names])
    #category_probabilities = [len(categories[category])/total for category in categories if category in remaining_category_names]
    category_name = get_random_category(categories, already_taken_categories, already_taken_words)
    already_taken_categories.append(category_name)

    category_words = categories[category_name]
    remaining_words = [word for word in category_words if word not in already_taken_words]
    
    # randomly choose 2-4 words from a category, so that not only one word slot remains
    choices = [2, 3, 4]
    for choice in choices:
        if maximum - choice == 1:
            choices.remove(choice)
            break
    amount = random.choice(choices)
    words = sample_words_from_category(list(remaining_words), min(amount, maximum))
    already_taken_words.extend(words)
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
    
def get_random_category(categories, already_taken_categories, already_taken_words):
    # remaining_category_names = list(categories.keys() - already_taken_categories)
    remaining_category_names = []
    for category in categories:
        if category in already_taken_categories:
            continue
        remaining_category_size = len([category_name for category_name in categories[category] if category_name not in already_taken_words])
        remaining_category_names.extend([category for i in range(remaining_category_size)])

    return random.choice(remaining_category_names)

    # return np_random_generator.choice(category_list, 1, p=probabilities)

generators={'random': generate_random,
            'easy word assignments': generate_similar_within_teams,
            'difficult word assignments': generate_similar_across_teams}

class CodenamesInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        super().__init__(GAME_NAME)

    def generate(self, keep=False, variable_name=None, experiment_name=None, generous=False):
        # @overwrite
        if generous:
            filename = GENEROUS_FILENAME
        else:
            filename = FILENAME
        if not self.on_generate(variable_name, experiment_name, generous):
            return
        if keep:
            if variable_name and experiment_name:
                print(f"Replacing instances for {variable_name}: {experiment_name}.")
            elif variable_name:
                print(f"Replacing instances for variable {variable_name}.")
            else:
                print(f"Replacing instances for experiment {experiment_name}.")
            self.replace_instances(variable_name, experiment_name, filename)
        self.store_file(self.instances, filename, sub_dir="in")
        
    def on_generate(self, variable_name = None, experiment_name = None, generous=False):
        # read experiment config file
        experiment_config = self.load_json("resources/experiments.json")
        defaults = experiment_config["default"]
        variable_experiments = experiment_config["variables"]
        variable_names = variable_experiments.keys()

        if variable_name:
            if variable_name not in variable_names:
                print(f"Variable name {variable_name} not found in experiment config file (only {', '.join(list(variable_names))}).")
                return False
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
                    return False
                # if the experiment name was set (correctly), we will only generate instances for this specific experiment
                print(f"(Re-)Generate only instances for {experiment_name}.")
                experiment_names = [experiment_name]
                # otherwise instances for all experiments changing this variable are generated

            for name in experiment_names:
                # load correct wordlist
                if "wordlist" in experiments[name].keys():
                    wordlist_name = experiments[name]["wordlist"]
                else:
                    wordlist_name = defaults["wordlist"]
                wordlist_path = f"resources/cleaned_wordlists/{wordlist_name}"
                if not os.path.isfile(file_path(wordlist_path, GAME_NAME)):
                    print(f"> Wordlist {wordlist_name} does not exist, skip {name}.")
                    continue
                wordlist = self.load_json(wordlist_path)["words"]

                print("Generating instances for experiment: ", name)
                experiment = self.add_experiment(name)
                experiment["variable"] = variable_name
                # set default parameters
                for parameter in defaults:
                    experiment[parameter] = defaults[parameter]
                # set experiment-specific parameters
                for parameter in experiments[name]:
                    print("Setting experiment parameter: ", parameter)
                    experiment[parameter] = experiments[name][parameter]
                # set flags
                experiment["flags"] = {}
                for flag in FLAGS:
                    if generous:
                        experiment["flags"][flag] = True
                    else:
                        experiment["flags"][flag] = False

                # FIXME: bad hack to always strip words
                experiment["flags"]["STRIP WORDS"] = True

                # create game instances (also with tqdm possible here)
                for game_id in tqdm(range(experiment["number of instances"])):
                    # choose correct generator function
                    assignments = experiment[ASSIGNMENTS]
                    generator = generators[experiment["generator"]]
                    instance = generator(wordlist, assignments)
                    self.test_instance_format(instance, assignments)

                    # Create a game instance
                    game_instance = self.add_game_instance(experiment, game_id)
                    # Add game parameters
                    for key in instance.keys():
                        game_instance[key] = instance[key]
        return True
            
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
            print("File does not exist, can't be 'overwritten'...")
            return
        old_experiments = file["experiments"]
        # adding all new experiment instances
        new_experiments = self.instances["experiments"]
        for i in range(len(old_experiments)):
            if old_experiments[i]["variable"] == variable_name:
                if experiment_name and not old_experiments[i]["name"] == experiment_name:
                    # experiment name was set, but these old instances belong to a different experiment, so should be kept
                    print(f"Keep {variable_name}: {old_experiments[i]['name']}.")
                    new_experiments.append(old_experiments[i])
                else:
                    print(f"Replace {variable_name}: {old_experiments[i]['name']}.")
            else:
                # if the variable name is not the same, then these instances should also be kept
                print(f"Keep {variable_name}: {old_experiments[i]['name']}.")
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
    parser.add_argument("-g", "--generous", help="Optional flag to generate generous instances where all flags are set to True.", action="store_true")
    args = parser.parse_args()
    if args.experiment_name and not args.variable_name:
        print("Running a specific experiment requires both the experiment name (-e) and the variable name (-v)!")
    else:
        random.seed(SEED)
        CodenamesInstanceGenerator().generate(keep = args.keep, variable_name = args.variable_name, experiment_name = args.experiment_name, generous = args.generous)
