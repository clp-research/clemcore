"""
Generate instances for the Guess What game.

"""

import os
import sys
from pathlib import Path

print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")

# Adjust the path to include the project root directory
sys.path.append(str(Path(__file__).resolve().parents[2]))

import random
import json
from tqdm import tqdm
import clemgame
from clemgame.clemgame import GameInstanceGenerator

# number of words per episode
num_words = 8

# number of instances per experiment
N_INSTANCES = 10

logger = clemgame.get_logger(__name__)
GAME_NAME = "guesswhat"


class GuessWhatGameInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        super().__init__(GAME_NAME)

    def load_instances(self):
        return self.load_json("in/instances")

    def on_generate(self):
        output_instances = {
            "experiments": []
        }
        self.generate_mix_set(output_instances)
        self.generate_abstract_set(output_instances)

    def generate_mix_set(self, output_instances):
        output_instance_details = {"Level_1": [], "Level_2": [], "Level_3": []}

        category_file_path = os.path.join(os.path.dirname(__file__), "utils", "categories_subcategories.json")
        with open(category_file_path, 'r') as f:
            self.categories = json.load(f)["Categories"]

        for level in [1, 2, 3]:
            experiment_name = f"Level_{level}"
            experiment = self.add_experiment(experiment_name)

            max_turns = num_words
            experiment["max_turns"] = max_turns
            experiment["question_tag"] = "QUESTION:"
            experiment["answer_tag"] = "ANSWER:"
            experiment["guess_tag"] = "GUESS:"
            experiment["answer_variations"] = ["ANSWER: yes", "ANSWER: no", "ANSWER: Yes.", "ANSWER: Yes",
                                               "ANSWER: No.", "ANSWER: No"]
            experiment[
                "letter_based_pattern"] = "(does the target word (start with the letter|contain the letter)|does the target word have the letter\s*[a-z]|is the (first|second|third|fourth|fifth) letter of the target word [a-z])"
            experiment["direct_guess_pattern"] = "^is the target word\s*(['\"])[^'\"]+?\1\s*\?"
            experiment[
                "length_question_pattern"] = "does the target word (have|contain) (more|less|exactly) \d+ (letters|letter)"
            experiment[
                "syllable_question_pattern"] = "does the target word (have|contain) (more than|less than|exactly) (one|two|three|four|five|six|seven|eight|nine|ten|\d+) (syllable|syllables)"
            experiment[
                "pos_question_pattern"] = "^is the target word (a|an)\s+(noun|verb|adjective|adverb|pronoun|preposition|conjunction|interjection)\s*\?"

            answerer_prompt = self.load_template("resources/initial_prompts/answerer_prompt")
            guesser_prompt = self.load_template("resources/initial_prompts/guesser_prompt")

            experiment["answerer_initial_prompt"] = answerer_prompt
            experiment["guesser_initial_prompt"] = guesser_prompt

            used_words = set()
            game_instances = []
            for game_id in tqdm(range(N_INSTANCES)):
                instance, instance_details = self.generate_instance(level, used_words)
                if instance:
                    game_instance = self.add_game_instance(experiment, game_id)
                    game_instance["target_word"] = instance["target"]
                    game_instance["candidate_list"] = instance["items"]
                    game_instances.append(game_instance)

                    output_instance_details[experiment_name].append(instance_details)

            experiment["game_instances"] = game_instances
            output_instances["experiments"].append(experiment)

    def generate_abstract_set(self, output_instances):

        category_file_path = os.path.join(os.path.dirname(__file__), "utils", "abstract_categories_subcategories.json")
        with open(category_file_path, 'r') as f:
            self.categories = json.load(f)["Categories"]

        output_instance_details = {"Abs_Level_1": [], "Abs_Level_2": [], "Abs_Level_3": []}

        for level in [1, 2, 3]:
            experiment_name = f"Abs_Level_{level}"
            experiment = self.add_experiment(experiment_name)

            max_turns = num_words  # max_turns is defined as the number of words in the candidate list
            experiment["max_turns"] = max_turns
            experiment["question_tag"] = "QUESTION:"
            experiment["answer_tag"] = "ANSWER:"
            experiment["guess_tag"] = "GUESS:"
            experiment["answer_variations"] = ["ANSWER: yes", "ANSWER: no", "ANSWER: Yes.", "ANSWER: Yes",
                                               "ANSWER: No.", "ANSWER: No"]
            experiment[
                "letter_based_pattern"] = "(does the target word (start with the letter|contain the letter)|does the target word have the letter\s*[a-z]|is the (first|second|third|fourth|fifth) letter of the target word [a-z])"
            experiment["direct_guess_pattern"] = "^is the target word\s*(['\"])[^'\"]+?\1\s*\?"
            experiment[
                "length_question_pattern"] = "does the target word (have|contain) (more|less|exactly) \d+ (letters|letter)"
            experiment[
                "syllable_question_pattern"] = "does the target word (have|contain) (more than|less than|exactly) (one|two|three|four|five|six|seven|eight|nine|ten|\d+) (syllable|syllables)"
            experiment[
                "pos_question_pattern"] = "^is the target word (a|an)\s+(noun|verb|adjective|adverb|pronoun|preposition|conjunction|interjection)\s*\?"

            answerer_prompt = self.load_template("resources/initial_prompts/answerer_prompt")
            guesser_prompt = self.load_template("resources/initial_prompts/guesser_prompt")


            experiment["answerer_initial_prompt"] = answerer_prompt

            experiment["guesser_initial_prompt"] = guesser_prompt


            used_words = set()
            game_instances = []
            for game_id in tqdm(range(N_INSTANCES)):
                instance, instance_details = self.generate_instance(level, used_words)
                if instance:
                    game_instance = self.add_game_instance(experiment, game_id)
                    game_instance["target_word"] = instance["target"]
                    game_instance["candidate_list"] = instance["items"]
                    game_instances.append(game_instance)

                    output_instance_details[experiment_name].append(instance_details)

            experiment["game_instances"] = game_instances
            output_instances["experiments"].append(experiment)


    def generate_instance(self, level, used_words):
        instance = {"items": [], "target": ""}
        instance_details = {"items": [], "target": ""}
        used_categories = set()

        required_words = num_words

        def find_valid_categories(level):
            if level == 1:
                return [
                    c for c in self.categories if c["Category"] not in used_categories
                    and len([sub for sub in c["Subcategories"] if len(sub["Members"]) >= 1]) >= 2
                ]
            elif level == 2:
                return [
                    c for c in self.categories if c["Category"] not in used_categories
                    and len([sub for sub in c["Subcategories"] if len(sub["Members"]) >= 2]) >= 2
                ]
            elif level == 3:
                return [
                    c for c in self.categories if c["Category"] not in used_categories
                    and len([sub for sub in c["Subcategories"] if len(sub["Members"]) >= 2]) >= 4
                ]
            return []

        def find_valid_subcategories(category, level):
            if level == 1:
                return [sub for sub in category["Subcategories"] if len(sub["Members"]) >= 1]
            elif level == 2:
                return [sub for sub in category["Subcategories"] if len(sub["Members"]) >= 2]
            elif level == 3:
                return [sub for sub in category["Subcategories"] if len(sub["Members"]) >= 2]
            return []


        # Retry limit to prevent infinite loops
        for _ in range(100):
            instance["items"].clear()
            instance_details["items"].clear()

            if level == 1:
                valid_categories = find_valid_categories(level)
                if len(valid_categories) < 4:
                    print("Warning: Not enough valid categories for Level 1. Using available categories.")
                    valid_categories = [c for c in self.categories if c["Category"] not in used_categories]

                selected_categories = random.sample(valid_categories, min(4, len(valid_categories)))
                used_categories.update(cat["Category"] for cat in selected_categories)

                for category in selected_categories:
                    subcategories = find_valid_subcategories(category, level)
                    if len(subcategories) < 2:
                        subcategories = [sub for sub in category["Subcategories"]]
                    selected_subcategories = random.sample(subcategories, min(2, len(subcategories)))
                    for sub in selected_subcategories:
                        available_words = [w for w in sub["Members"] if w not in used_words]
                        if len(available_words) < 1:
                            continue
                        word = random.choice(available_words)
                        used_words.add(word)
                        instance["items"].append(word)
                        instance_details["items"].append({
                            "word": word,
                            "category": category["Category"],
                            "feature": sub["Subcategory"]
                        })

            elif level == 2:
                valid_categories = find_valid_categories(level)
                if len(valid_categories) < 2:
                    valid_categories = [c for c in self.categories if c["Category"] not in used_categories]

                selected_categories = random.sample(valid_categories, min(2, len(valid_categories)))
                used_categories.update(cat["Category"] for cat in selected_categories)

                for category in selected_categories:
                    subcategories = find_valid_subcategories(category, level)
                    if len(subcategories) < 2:
                        subcategories = [sub for sub in category["Subcategories"]]
                    selected_subcategories = random.sample(subcategories, min(2, len(subcategories)))
                    for sub in selected_subcategories:
                        available_words = [w for w in sub["Members"] if w not in used_words]
                        if len(available_words) < 2:
                            continue
                        words = random.sample(available_words, 2)
                        for word in words:
                            used_words.add(word)
                            instance["items"].append(word)
                            instance_details["items"].append({
                                "word": word,
                                "category": category["Category"],
                                "feature": sub["Subcategory"]
                            })

            elif level == 3:
                valid_categories = find_valid_categories(level)
                if len(valid_categories) < 1:
                    valid_categories = [c for c in self.categories if c["Category"] not in used_categories]

                selected_category = random.choice(valid_categories)
                used_categories.add(selected_category["Category"])

                subcategories = find_valid_subcategories(selected_category, level)
                if len(subcategories) < 4:
                    subcategories = [sub for sub in selected_category["Subcategories"]]
                selected_subcategories = random.sample(subcategories, min(4, len(subcategories)))

                for sub in selected_subcategories:
                    available_words = [w for w in sub["Members"] if w not in used_words]
                    if len(available_words) < 2:
                        continue
                    words = random.sample(available_words, 2)
                    for word in words:
                        used_words.add(word)
                        instance["items"].append(word)
                        instance_details["items"].append({
                            "word": word,
                            "category": selected_category["Category"],
                            "feature": sub["Subcategory"]
                        })

            if len(instance["items"]) >= required_words:
                instance["items"] = instance["items"][:required_words]
                instance_details["items"] = instance_details["items"][:required_words]
                instance["target"] = random.choice(instance["items"])
                instance_details["target"] = instance["target"]
                return instance, instance_details

        print("Error: Could not generate a valid instance after several attempts.")
        return None, None


    def save_json(self, data, filepath):
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

if __name__ == '__main__':
    GuessWhatGameInstanceGenerator().generate()


