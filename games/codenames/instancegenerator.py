"""
Generate instances for the game.

Creates files in ./in
"""
from tqdm import tqdm
from clemgame.clemgame import GameInstanceGenerator
import random, copy, argparse

from games.codenames.constants import *

def generate_random(wordlist, assignments):
    # sample words for the board
    total = assignments[TEAM] + assignments[OPPONENT] + assignments[INNOCENT] + assignments[ASSASSIN]
    board = random.sample(wordlist, total)

    # make the assignments for the cluegiver and remove instances from 'unsampled' that were already sampled
    unsampled = copy.copy(board)
    team_words = random.sample(unsampled, assignments[TEAM])
    unsampled = [word for word in unsampled if word not in team_words]
    opponent_words = random.sample(unsampled, assignments[OPPONENT])
    unsampled = [word for word in unsampled if word not in opponent_words]
    innocent_words = random.sample(unsampled, assignments[INNOCENT])
    unsampled = [word for word in unsampled if word not in innocent_words]
    assassin_words = random.sample(unsampled, assignments[ASSASSIN])
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

def generate_similar_within_teams(wordlist, assignments):
    pass

def generate_similar_across_teams(wordlist, assignments):
    pass

generators={'random': generate_random,
            'easy word assignments': generate_similar_within_teams,
            'difficult word assignments': generate_similar_across_teams}

class CodenamesInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        super().__init__(GAME_NAME)

    def generate(self, variable_name=None, experiment_name=None, filename="instances.json"):
        # @overwrite
        self.on_generate(variable_name, experiment_name)
        #if experiment_name:
        #    self.replace_instances(experiment_name, filename)
        self.store_file(self.instances, filename, sub_dir="in")
        
    def on_generate(self, variable_name = None, experiment_name = None):
        # read experiment config file
        experiment_config = self.load_json("resources/experiments.json")
        defaults = experiment_config["default"]
        variable_experiments = experiment_config["variables"]
        variable_names = variable_experiments.keys()

        if variable_name:
            # if the variable_name was set, we will only generate instances for this experiment suite
            print(f"(Re-)Generate only instances for experiments on {variable_name}.")
            variable_names = [variable_name]
            # otherwise instances for all variables are generated

        for variable_name in variable_names:
            print("Variable name", variable_name)
            experiments = variable_experiments[variable_name]["experiments"]
            experiment_names = experiments.keys()
            if experiment_name:
                # if the experiment name was set, we will only generate instances for this specific experiment
                print(f"(Re-)Generate only instances for {experiment_name}.")
                experiment_names = [experiment_name]
                # otherwise instances for all experiments changing this variable are generated

            for experiment_name in experiment_names:
                print("Experiment name", experiment_name)
                experiment = self.add_experiment(experiment_name)
                experiment["variable"] = variable_name
                # add variable name
                # add
                # set default parameters
                for parameter in defaults:
                    experiment[parameter] = defaults[parameter]
                # set experiment-specific parameters
                for parameter in experiments[experiment_name]:
                    print("Experiment parameter", parameter)
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
        print(assigned_words)
        if set(board_instance[BOARD]) != set(assigned_words):
            raise ValueError(f"The words on the board do not match all the assigned words.")
        
    def replace_instances(self, experiment_name, filename="instances.json"):
        instances = self.load_json(f"in/{filename}")
        new_instances = {}
        for experiment in instances["experiments"]:
            if experiment["name"] != experiment_name:
                #instance
                #experiment["game_instances"] = self.instances[]
                pass
        instances[experiment_name] = self.instances["experiments"][experiment_name]
        self.instances = instances

if __name__ == '__main__':
    # The resulting instances.json is automatically saved to the "in" directory of the game folder
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--variable-name", type=str, help="Optional argument to only (re-) generate instances for a specific experiment suite aka variable.")
    parser.add_argument("-e", "--experiment-name", type=str, help="Optional argument to only (re-) generate instances for a specific experiment (variable name must also be set!).")
    args = parser.parse_args()
    # check that experiment name is only set when variable name is also set!
    random.seed(SEED)
    CodenamesInstanceGenerator().generate(variable_name = args.variable_name, experiment_name = args.experiment_name)
