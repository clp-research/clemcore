"""
Generate instances for the taboo game.

Creates files in ./instances
"""
import random

from tqdm import tqdm

import clemgame
from clemgame.clemgame import GameInstanceGenerator

N_INSTANCES = 20  # how many different target words; zero means "all"
N_GUESSES = 3  # how many tries the guesser will have
N_REATED_WORDS = 3
LANGUAGE = "en"

logger = clemgame.get_logger(__name__)
GAME_NAME = "taboo"


class TabooGameInstanceGenerator(GameInstanceGenerator):

    def __init__(self):
        super().__init__(GAME_NAME)

    def load_instances(self):
        return self.load_json("in/instances")

    def on_generate(self):
        for frequency in ["high", "medium", "low"]:
            print("Sampling from freq:", frequency)
            # first choose target words based on the difficultly
            fp = f"resources/target_words/{LANGUAGE}/{frequency}_freq_100_v1.5"
            target_words = self.load_file(file_name=fp, file_ending=".txt").split('\n')
            if N_INSTANCES > 0:
                assert len(target_words) >= N_INSTANCES, \
                    f'Fewer words available ({len(target_words)}) than requested ({N_INSTANCES}).'
                target_words = random.sample(target_words, k=N_INSTANCES)

            # use the same target_words for the different player assignments
            experiment = self.add_experiment(f"{frequency}_{LANGUAGE}")
            experiment["max_turns"] = N_GUESSES

            describer_prompt = self.load_template("resources/initial_prompts/initial_describer")
            guesser_prompt = self.load_template("resources/initial_prompts/initial_guesser")
            experiment["describer_initial_prompt"] = describer_prompt
            experiment["guesser_initial_prompt"] = guesser_prompt

            for game_id in tqdm(range(len(target_words))):
                target = target_words[game_id]

                game_instance = self.add_game_instance(experiment, game_id)
                game_instance["target_word"] = target
                game_instance["related_word"] = []  # ps: add manually for now b.c. api doesn't provide ranking

                if len(game_instance["related_word"]) < N_REATED_WORDS:
                    print(f"Found less than {N_REATED_WORDS} related words for: {target}")


if __name__ == '__main__':
    TabooGameInstanceGenerator().generate()
