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


if __name__ == '__main__':
    # The resulting instances.json is automatically saved to the "in" directory of the game folder
    random.seed(SEED)
    CodenamesInstanceGenerator().generate()
