"""
Generate instances for the game.

Creates files in ./in
"""
from tqdm import tqdm

#import clemgame
from clemgame.clemgame import GameInstanceGenerator
import random, copy
from string import Template

#logger = clemgame.get_logger(__name__)

GAME_NAME = "codenames"
EXPERIMENT_TYPE = "example"
NUMBER_OF_INSTANCES = 10
CARD_ASSIGNMENTS = {"all": 25, "team": 9, "opponent": 8, "innocent": 7, "assassin": 1}
SEED = 42


# TODO: make number of words, number of team words and so on game or experiment parameters
# TODO: sample from different prepared word lists (or with different similarity metrics) for different experiments
# TODO: filter out clue words that are not single words (e.g. NEW YORK)
# TODO: read from command line or make some experiment config files so that it is easy to implement new experiment instance generations
# experiment: {assignments: {...}, wordlists: [...], generator: function() or string identifier}

class CodenamesInstanceGenerator(GameInstanceGenerator):

    def __init__(self):
        super().__init__(GAME_NAME)

    def on_generate(self):
        # Create an experiment
        experiment = self.add_experiment(f"example_instances")
        experiment["type"] = EXPERIMENT_TYPE  # experiment parameters
        # from which words is sampled

        # Load a list of words to choose from
        words = self.load_json("resources/word_lists/original")["words"]

        # We create one instance for the number of required instances
        assert CARD_ASSIGNMENTS["all"] == CARD_ASSIGNMENTS["team"] + CARD_ASSIGNMENTS["opponent"] + CARD_ASSIGNMENTS["innocent"] + CARD_ASSIGNMENTS["assassin"], "Number of words and summed number of assignments to teams does not match"
        for game_id in tqdm(range(NUMBER_OF_INSTANCES)):
            # sample words for the board
            board = random.sample(words, CARD_ASSIGNMENTS["all"])

            # make the assignments for the cluegiver and remove instances from `unsampled`` that were already sampled
            unsampled = copy.copy(board)
            team_words = random.sample(unsampled, CARD_ASSIGNMENTS["team"])
            unsampled = [word for word in unsampled if word not in team_words]
            opponent_words = random.sample(unsampled, CARD_ASSIGNMENTS["opponent"])
            unsampled = [word for word in unsampled if word not in opponent_words]
            innocent_words = random.sample(unsampled, CARD_ASSIGNMENTS["innocent"])
            unsampled = [word for word in unsampled if word not in innocent_words]
            assassin_words = random.sample(unsampled, CARD_ASSIGNMENTS["assassin"])
            unsampled = [word for word in unsampled if word not in assassin_words]
            assert len(unsampled) == 0, "Not all words have been assigned to a team!"

            # Create a game instance
            game_instance = self.add_game_instance(experiment, game_id)
            # Add game parameters
            game_instance["board"] = board
            game_instance["assignments"] = {"team": team_words, "opponent": opponent_words, "innocent": innocent_words, "assassin": assassin_words}

    def test_instance_format(board_instance, params):
        # board_instance = {"board": [...],
        # "assignments": {"team": [...], "opponent": [...], "innocent": [...], "assassin": [...]}}
        
        keys = ['total', 'team', 'opponent', 'innocent', 'assassin']
        assert set(params.keys()) == set(keys), f"The params dictionary is missing a key, keys are {params.keys()}, but should be {keys}!"
        
        if not "board" in board_instance:
            raise KeyError("The key 'board' was not found in the board instance.")
        if not "assignments" in board_instance:
            raise KeyError("The key 'assignments' was not found in the board instance.")
        
        if len(board_instance["board"]) != params["total"]:
            raise ValueError(f"The total length of the board {len(board_instance['board'])} is unequal to the required board length {total}.")
        for alignment in params.keys():
            if alignment == 'total':
                continue
            if len(board_instance["assignments"][alignment]) != params[alignment]:
                raise ValueError(f"The number of {alignment} on the board ({len(board_instance['assignments'][alignment])}) is unequal to the required number of {alignment} words ({params[alignment]})")
        
        if params['total'] != params['team'] + params['opponent'] + params['innocent'] + params['assassin']:
            raise ValueError(f"The sum of all assignments does not match the total number of words!")
            
        assigned_words = [x for y in board_instance["assignments"] for x in board_instance['assignments'][y]]
        print(assigned_words)
        if set(board_instance["board"]) != set(assigned_words):
            raise ValueError(f"The words on the board do not match all the assigned words.")

    def test_all_instances(instances, params):
        for instance in instances:
            self.test_instance_format(instance, params)


if __name__ == '__main__':
    # The resulting instances.json is automatically saved to the "in" directory of the game folder
    random.seed(SEED)
    CodenamesInstanceGenerator().generate()
