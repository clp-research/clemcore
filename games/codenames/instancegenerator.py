"""
Generate instances for the game.

Creates files in ./in
"""
from tqdm import tqdm
from clemgame.clemgame import GameInstanceGenerator
import random, copy
from string import Template

# TODO: these constants should all be experiment parameters, either written in some config file or given as cli parameters
# experiment: {assignments: {...}, wordlists: [...], generator: function() or string identifier}
EXPERIMENT_TYPE = "example"
EXPERIMENT_DIFFICULTY = 1
NUMBER_OF_INSTANCES = 10
CARD_ASSIGNMENTS = {"all": 25, TEAM: 9, OPPONENT: 8, INNOCENT: 7, ASSASSIN: 1}

from games.codenames.constants import *

class CodenamesInstanceGenerator(GameInstanceGenerator):

    def __init__(self):
        super().__init__(GAME_NAME)

    def on_generate(self):
        # Create an experiment
        experiment = self.add_experiment(f"example_instances")
        experiment[TYPE] = EXPERIMENT_TYPE  # experiment parameters
        experiment[OPPONENT_DIFFICULTY] = EXPERIMENT_DIFFICULTY
        # from which words is sampled

        # Load a list of words to choose from
        words = self.load_json("resources/word_lists/original")["words"]

        # We create one instance for the number of required instances
        assert CARD_ASSIGNMENTS["all"] == CARD_ASSIGNMENTS[TEAM] + CARD_ASSIGNMENTS[OPPONENT] + CARD_ASSIGNMENTS[INNOCENT] + CARD_ASSIGNMENTS[ASSASSIN], "Number of words and summed number of assignments to teams does not match"
        for game_id in tqdm(range(NUMBER_OF_INSTANCES)):
            # sample words for the board
            board = random.sample(words, CARD_ASSIGNMENTS["all"])

            # make the assignments for the cluegiver and remove instances from `unsampled`` that were already sampled
            unsampled = copy.copy(board)
            team_words = random.sample(unsampled, CARD_ASSIGNMENTS[TEAM])
            unsampled = [word for word in unsampled if word not in team_words]
            opponent_words = random.sample(unsampled, CARD_ASSIGNMENTS[OPPONENT])
            unsampled = [word for word in unsampled if word not in opponent_words]
            innocent_words = random.sample(unsampled, CARD_ASSIGNMENTS[INNOCENT])
            unsampled = [word for word in unsampled if word not in innocent_words]
            assassin_words = random.sample(unsampled, CARD_ASSIGNMENTS[ASSASSIN])
            unsampled = [word for word in unsampled if word not in assassin_words]
            assert len(unsampled) == 0, "Not all words have been assigned to a team!"

            # Create a game instance
            game_instance = self.add_game_instance(experiment, game_id)
            # Add game parameters
            game_instance[BOARD] = board
            game_instance[ASSIGNMENTS] = {TEAM: team_words, OPPONENT: opponent_words, INNOCENT: innocent_words, ASSASSIN: assassin_words}

    def test_instance_format(self, board_instance, params):
        # board_instance = {BOARD: [...],
        # ASSIGNMENTS: {TEAM: [...], OPPONENT: [...], INNOCENT: [...], ASSASSIN: [...]}}
        
        keys = [TOTAL, TEAM, OPPONENT, INNOCENT, ASSASSIN]
        assert set(params.keys()) == set(keys), f"The params dictionary is missing a key, keys are {params.keys()}, but should be {keys}!"
        
        if not BOARD in board_instance:
            raise KeyError(f"The key '{BOARD}' was not found in the board instance.")
        if not ASSIGNMENTS in board_instance:
            raise KeyError(f"The key '{ASSIGNMENTS}' was not found in the board instance.")
        
        if len(board_instance[BOARD]) != params[TOTAL]:
            raise ValueError(f"The total length of the board {len(board_instance[BOARD])} is unequal to the required board length {params[TOTAL]}.")
        for alignment in params.keys():
            if alignment == TOTAL:
                continue
            if len(board_instance[ASSIGNMENTS][alignment]) != params[alignment]:
                raise ValueError(f"The number of {alignment} on the board ({len(board_instance[ASSIGNMENTS][alignment])}) is unequal to the required number of {alignment} words ({params[alignment]})")
        
        if params[TOTAL] != params[TEAM] + params[OPPONENT] + params[INNOCENT] + params[ASSASSIN]:
            raise ValueError(f"The sum of all assignments does not match the total number of words!")
            
        assigned_words = [x for y in board_instance[ASSIGNMENTS] for x in board_instance[ASSIGNMENTS][y]]
        print(assigned_words)
        if set(board_instance[BOARD]) != set(assigned_words):
            raise ValueError(f"The words on the board do not match all the assigned words.")

    def test_all_instances(self, instances, params):
        for instance in instances:
            self.test_instance_format(instance, params)


if __name__ == '__main__':
    # The resulting instances.json is automatically saved to the "in" directory of the game folder
    random.seed(SEED)
    CodenamesInstanceGenerator().generate()
