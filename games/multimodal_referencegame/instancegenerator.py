"""
Generate instances for the referencegame
Version 1.6 (strict regex parsing)

Reads grids_v1.5.json from resources/ (grids don't change in this version)
Creates instances.json in instances/
"""

import random
import clemgame
from clemgame.clemgame import GameInstanceGenerator
import shutil

random.seed(123)

logger = clemgame.get_logger(__name__)
GAME_NAME = "multimodal_referencegame"

MAX_NUMBER_INSTANCES = 30





class ReferenceGameInstanceGenerator(GameInstanceGenerator):

    def __init__(self):
        super().__init__(GAME_NAME)

    def get_aed_dataset(self):
        sequences = self.load_csv(f"resources/sequences.csv")

        aed_dataset = dict()
        for s in sequences:
            line = s[0].split("\t")

            if line[0] == '':
                continue

            image_path = "resources/aed_images/"+line[3].split("/")[-1]
            image_category = line[4]

            if image_category not in aed_dataset:
                aed_dataset[image_category] = [image_path]
            else:
                aed_dataset[image_category].append(image_path)
        return aed_dataset

    def select_random_item(self, images:list):
        random_index = random.randint(0, len(images)-1)
        return images[random_index]

    def generate_grid_instances(self):
        # GRID EXPERIMENT
        player_a_prompt_header = self.load_template(f"resources/initial_prompts/player_a_prompt_images.template")
        player_b_prompt_header = self.load_template(f"resources/initial_prompts/player_b_prompt_images.template")

        grid_dataset = self.load_json('resources/grid_files.json')

        game_counter = 0
        experiment = self.add_experiment('grids')
        while True:

            if game_counter == 30:
                break

            for target_category in grid_dataset:

                target_category_images = grid_dataset[target_category]
                target_image = self.select_random_item(target_category_images)
                target_image_path = f"games/multimodal_referencegame/resources/{target_image}"

                # remove the target image from the list, select another image from the same category
                target_category_images.remove(target_image)
                distractor1 = self.select_random_item(target_category_images)
                distractor1_path = f"games/multimodal_referencegame/resources/{distractor1}"

                # get all image categories, remove the target category
                all_categories = list(grid_dataset.keys())
                all_categories.remove(target_category)
                distractor2_category = self.select_random_item(all_categories)
                distractor2 = self.select_random_item(grid_dataset[distractor2_category])
                distractor2_path = f"games/multimodal_referencegame/resources/{distractor2}"

                for i in [1, 2, 3]:

                    player_a_prompt_header = self.load_template(
                        f"resources/initial_prompts/player_a_prompt_images.template")
                    game_instance = self.add_game_instance(experiment, game_counter)

                    player_1_first_image = ""
                    player_1_second_image = ""
                    player_1_third_image = ""
                    player_2_first_image = ""
                    player_2_second_image = ""
                    player_2_third_image = ""

                    if i == 1:
                        player_1_first_image = target_image_path
                        player_1_second_image = distractor1_path
                        player_1_third_image = distractor2_path

                        player_2_first_image = distractor1_path
                        player_2_second_image = target_image_path
                        player_2_third_image = distractor2_path

                        player_2_target_name = "second"

                        player_a_prompt_header = player_a_prompt_header.replace("FIRST_IMAGE", "the target").replace(
                            "SECOND_IMAGE", "a distractor").replace("THIRD_IMAGE", "a distractor")

                    elif i == 2:
                        player_1_first_image = distractor2_path
                        player_1_second_image = distractor1_path
                        player_1_third_image = target_image_path

                        player_2_first_image = target_image_path
                        player_2_second_image = distractor1_path
                        player_2_third_image = distractor2_path

                        player_2_target_name = "first"

                        player_a_prompt_header = player_a_prompt_header.replace("FIRST_IMAGE", "a distractor").replace(
                            "SECOND_IMAGE", "a distractor").replace("THIRD_IMAGE", "the target")

                    elif i == 3:
                        player_1_first_image = distractor2_path
                        player_1_second_image = target_image_path
                        player_1_third_image = distractor1_path

                        player_2_first_image = distractor2_path
                        player_2_second_image = distractor1_path
                        player_2_third_image = target_image_path

                        player_2_target_name = "third"

                        player_a_prompt_header = player_a_prompt_header.replace("FIRST_IMAGE", "a distractor").replace(
                            "SECOND_IMAGE", "the target").replace("THIRD_IMAGE", "a distractor")

                    game_instance["player_1_prompt_header"] = player_a_prompt_header
                    game_instance['player_1_first_image'] = player_1_first_image
                    game_instance['player_1_second_image'] = player_1_second_image
                    game_instance['player_1_third_image'] = player_1_third_image

                    game_instance["player_2_prompt_header"] = player_b_prompt_header
                    game_instance['player_2_first_image'] = player_2_first_image
                    game_instance['player_2_second_image'] = player_2_second_image
                    game_instance['player_2_third_image'] = player_2_third_image
                    game_instance['target_image_name'] = player_2_target_name
                    game_instance['player_1_response_pattern'] = '^expression:\s(?P<content>.+)\n*(?P<remainder>.*)'
                    # named groups:
                    # 'content' captures only the generated referring expression
                    # 'remainder' should be empty (if models followed the instructions)
                    game_instance[
                        'player_2_response_pattern'] = '^answer:\s(?P<content>first|second|third)\n*(?P<remainder>.*)'
                    # 'content' can directly be compared to gold answer
                    # 'remainder' should be empty (if models followed the instructions)

                    # the following two fields are no longer required, but kept for backwards compatibility with previous instance versions
                    game_instance["player_1_response_tag"] = "expression:"
                    game_instance["player_2_response_tag"] = "answer:"

                    game_counter += 1

                    if game_counter >= MAX_NUMBER_INSTANCES:
                        break

                if game_counter >= MAX_NUMBER_INSTANCES:
                    break

    def generate_scene_instances(self):
        player_a_prompt_header = self.load_template(f"resources/initial_prompts/player_a_prompt_images.template")
        player_b_prompt_header = self.load_template(f"resources/initial_prompts/player_b_prompt_images.template")

        aed_dataset = self.get_aed_dataset()

        game_counter = 0
        image_counter = 1
        experiment = self.add_experiment('scenes')
        for target_category in aed_dataset:

            target_category_images = aed_dataset[target_category]
            target_image = self.select_random_item(target_category_images)
            shutil.copyfile(target_image, f"resources/images/{str(image_counter)}.jpg")
            target_image_path = f"games/multimodal_referencegame/resources/images/{str(image_counter)}.jpg"
            image_counter += 1

            # remove the target image from the list, select another image from the same category
            target_category_images.remove(target_image)
            distractor1 = self.select_random_item(target_category_images)
            shutil.copyfile(distractor1, f"resources/images/{str(image_counter)}.jpg")
            distractor1_path = f"games/multimodal_referencegame/resources/images/{str(image_counter)}.jpg"
            image_counter += 1

            # get all image categories, remove the target category
            all_categories = list(aed_dataset.keys())
            all_categories.remove(target_category)
            distractor2_category = self.select_random_item(all_categories)
            distractor2 = self.select_random_item(aed_dataset[distractor2_category])

            shutil.copyfile(distractor2, f"resources/images/{str(image_counter)}.jpg")
            distractor2_path = f"games/multimodal_referencegame/resources/images/{str(image_counter)}.jpg"
            image_counter += 1

            for i in [1, 2, 3]:

                player_a_prompt_header = self.load_template(
                    f"resources/initial_prompts/player_a_prompt_images.template")
                game_instance = self.add_game_instance(experiment, game_counter)

                player_1_first_image = ""
                player_1_second_image = ""
                player_1_third_image = ""
                player_2_first_image = ""
                player_2_second_image = ""
                player_2_third_image = ""

                if i == 1:
                    player_1_first_image = target_image_path
                    player_1_second_image = distractor1_path
                    player_1_third_image = distractor2_path

                    player_2_first_image = distractor1_path
                    player_2_second_image = target_image_path
                    player_2_third_image = distractor2_path

                    player_2_target_name = "second"

                    player_a_prompt_header = player_a_prompt_header.replace("FIRST_IMAGE", "the target").replace(
                        "SECOND_IMAGE", "a distractor").replace("THIRD_IMAGE", "a distractor")

                elif i == 2:
                    player_1_first_image = distractor2_path
                    player_1_second_image = distractor1_path
                    player_1_third_image = target_image_path

                    player_2_first_image = target_image_path
                    player_2_second_image = distractor1_path
                    player_2_third_image = distractor2_path

                    player_2_target_name = "first"

                    player_a_prompt_header = player_a_prompt_header.replace("FIRST_IMAGE", "a distractor").replace(
                        "SECOND_IMAGE", "a distractor").replace("THIRD_IMAGE", "the target")

                elif i == 3:
                    player_1_first_image = distractor2_path
                    player_1_second_image = target_image_path
                    player_1_third_image = distractor1_path

                    player_2_first_image = distractor2_path
                    player_2_second_image = distractor1_path
                    player_2_third_image = target_image_path

                    player_2_target_name = "third"

                    player_a_prompt_header = player_a_prompt_header.replace("FIRST_IMAGE", "a distractor").replace(
                        "SECOND_IMAGE", "the target").replace("THIRD_IMAGE", "a distractor")

                game_instance["player_1_prompt_header"] = player_a_prompt_header
                game_instance['player_1_first_image'] = player_1_first_image
                game_instance['player_1_second_image'] = player_1_second_image
                game_instance['player_1_third_image'] = player_1_third_image

                game_instance["player_2_prompt_header"] = player_b_prompt_header
                game_instance['player_2_first_image'] = player_2_first_image
                game_instance['player_2_second_image'] = player_2_second_image
                game_instance['player_2_third_image'] = player_2_third_image
                game_instance['target_image_name'] = player_2_target_name
                game_instance['player_1_response_pattern'] = '^expression:\s(?P<content>.+)\n*(?P<remainder>.*)'
                # named groups:
                # 'content' captures only the generated referring expression
                # 'remainder' should be empty (if models followed the instructions)
                game_instance[
                    'player_2_response_pattern'] = '^answer:\s(?P<content>first|second|third)\n*(?P<remainder>.*)'
                # 'content' can directly be compared to gold answer
                # 'remainder' should be empty (if models followed the instructions)

                # the following two fields are no longer required, but kept for backwards compatibility with previous instance versions
                game_instance["player_1_response_tag"] = "expression:"
                game_instance["player_2_response_tag"] = "answer:"

                game_counter += 1

                if game_counter >= MAX_NUMBER_INSTANCES:
                    break

            if game_counter >= MAX_NUMBER_INSTANCES:
                break

    def on_generate(self):
        self.generate_grid_instances()
        self.generate_scene_instances()


if __name__ == '__main__':
    ReferenceGameInstanceGenerator().generate(filename="instances.json")
