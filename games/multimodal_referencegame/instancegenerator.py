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
import matplotlib.pyplot as plt
import json

random.seed(123)

logger = clemgame.get_logger(__name__)
GAME_NAME = "multimodal_referencegame"

MAX_NUMBER_INSTANCES = 30





class ReferenceGameInstanceGenerator(GameInstanceGenerator):

    def __init__(self):
        super().__init__(GAME_NAME)

    def get_ade_dataset(self):
        sequences = self.load_csv(f"resources/sequences.csv")

        aed_dataset = dict()
        for s in sequences:
            line = s[0].split("\t")

            if line[0] == '':
                continue

            image_path = "resources/ade_images/"+line[3].split("/")[-1]
            image_category = line[4]

            if image_category not in aed_dataset:
                aed_dataset[image_category] = [image_path]
            else:
                aed_dataset[image_category].append(image_path)
        return aed_dataset

    def get_docci_dataset(self):

        file = open('resources/docci_dataset/docci_metadata.jsonlines', 'r')

        dataset = dict()
        for s in file.readlines():
            line = json.loads(s)

            image_path = "resources/docci_dataset/images/"+line["example_id"] + ".jpg"

            for ann in line["cloud_vision_api_responses"]["labelAnnotations"]:
                if ann["score"] >= 0.8:
                    image_category = ann["description"].lower()

                    if image_category not in dataset:
                        dataset[image_category] = [image_path]
                    else:
                        if image_path not in dataset[image_category]:
                            dataset[image_category].append(image_path)
        return dataset

    def get_clevr_dataset(self):

        data = self.load_json('resources/CLEVR_v1.0/scenes/CLEVR_val_scenes.json')

        dataset = dict()
        for scene in data['scenes']:

            image_path = 'resources/CLEVR_v1.0/images/val/'+scene['image_filename']
            for object in scene['objects']:
                image_category = object['size']+ ' '+object['color']+' '+object['material']

                if image_category not in dataset:
                    dataset[image_category] = [image_path]
                else:
                    if image_path not in dataset[image_category]:
                        dataset[image_category].append(image_path)

            if len(dataset) == 100:
                break
        return dataset

    def select_random_item(self, images:list):
        random_index = random.randint(0, len(images)-1)
        return images[random_index]

    def plot_grid(self, grid, file_path):
        fig, ax = plt.subplots()
        ax.set_xticks([])
        ax.set_yticks([])

        for y in range(len(grid)):
            for x in range(len(grid[y])):
                if grid[y][x] == "▢":
                    ax.add_patch(plt.Rectangle((x, -y - 1), 1, 1, facecolor="white", edgecolor="black"))
                else:
                    ax.add_patch(plt.Rectangle((x, -y - 1), 1, 1, facecolor="white", edgecolor="black"))
                    ax.text(x + 0.5, -y - 0.5, grid[y][x], ha='center', va='center', color='black', fontsize=12)

        ax.autoscale_view()
        # plt.show()
        plt.savefig(file_path)

    def process_grid(self, saved_grids, grid):

        if grid not in saved_grids:
            ascii_grid = []
            lines = grid.split('\n')
            for l in lines:
                line = l.split(' ')
                ascii_grid.append(line)

            file_path = f"resources/grid_images/{len(saved_grids)}.png"
            self.plot_grid(ascii_grid, file_path)

            saved_grids[grid] = 'games/multimodal_referencegame/'+file_path
        return saved_grids[grid]

    def generate_grid_instances(self):
        # GRID EXPERIMENT
        player_a_prompt_header = self.load_template(f"resources/initial_prompts/player_a_prompt_images.template")
        player_b_prompt_header = self.load_template(f"resources/initial_prompts/player_b_prompt_images.template")

        instances = {}
        saved_grids = {}

        with open('resources/ascii_game_instances.json') as json_file:
            instances = json.load(json_file)

        for exp in instances['experiments']:

            game_counter = 0
            experiment = self.add_experiment(exp['name'])



            for instance in exp['game_instances']:

                player1_target_grid =self.process_grid(saved_grids, instance['player_1_target_grid'])
                player_1_second_grid = self.process_grid(saved_grids, instance['player_1_second_grid'])
                player_1_third_grid = self.process_grid(saved_grids, instance['player_1_third_grid'])

                player_2_first_grid = self.process_grid(saved_grids, instance['player_2_first_grid'])
                player_2_second_grid = self.process_grid(saved_grids, instance['player_2_second_grid'])
                player_2_third_grid = self.process_grid(saved_grids, instance['player_2_third_grid'])

                game_instance = self.add_game_instance(experiment, game_counter)

                game_instance["player_1_prompt_header"] = player_a_prompt_header.replace('FIRST_IMAGE', 'the target').replace('SECOND_IMAGE', 'a distractor').replace('THIRD_IMAGE', 'a distractor')
                game_instance['player_1_first_image'] = player1_target_grid
                game_instance['player_1_second_image'] = player_1_second_grid
                game_instance['player_1_third_image'] = player_1_third_grid

                game_instance["player_2_prompt_header"] = player_b_prompt_header
                game_instance['player_2_first_image'] = player_2_first_grid
                game_instance['player_2_second_image'] = player_2_second_grid
                game_instance['player_2_third_image'] = player_2_third_grid
                game_instance['target_image_name'] = instance['target_grid_name']
                game_instance['player_1_response_pattern'] = '^expression:\s(?P<content>.+)\n*(?P<remainder>.*)'
                # named groups:
                # 'content' captures only the generated referring expression
                # 'remainder' should be empty (if models followed the instructions)
                game_instance[
                    'player_2_response_pattern'] = '^answer:\s(?P<content>first|second|third|1|2|3|1st|2nd|3rd)\n*(?P<remainder>.*)'
                # 'content' can directly be compared to gold answer
                # 'remainder' should be empty (if models followed the instructions)

                # the following two fields are no longer required, but kept for backwards compatibility with previous instance versions
                game_instance["player_1_response_tag"] = "expression:"
                game_instance["player_2_response_tag"] = "answer:"

                game_counter += 1

    def generate_scene_instances(self):
        player_a_prompt_header = self.load_template(f"resources/initial_prompts/player_a_prompt_images.template")
        player_b_prompt_header = self.load_template(f"resources/initial_prompts/player_b_prompt_images.template")

        aed_dataset = self.get_ade_dataset()

        game_counter = 0
        image_counter = 1
        experiment = self.add_experiment('ADE_images')
        for target_category in aed_dataset:

            target_category_images = aed_dataset[target_category]
            target_image = self.select_random_item(target_category_images)
            shutil.copyfile(target_image, f"resources/scene_images/{str(image_counter)}.jpg")
            target_image_path = f"games/multimodal_referencegame/resources/scene_images/{str(image_counter)}.jpg"
            image_counter += 1

            # remove the target image from the list, select another image from the same category
            target_category_images.remove(target_image)
            distractor1 = self.select_random_item(target_category_images)
            shutil.copyfile(distractor1, f"resources/scene_images/{str(image_counter)}.jpg")
            distractor1_path = f"games/multimodal_referencegame/resources/scene_images/{str(image_counter)}.jpg"
            image_counter += 1

            # remove the target image from the list, select another image from the same category
            target_category_images.remove(distractor1)
            distractor2 = self.select_random_item(target_category_images)
            shutil.copyfile(distractor2, f"resources/scene_images/{str(image_counter)}.jpg")
            distractor2_path = f"games/multimodal_referencegame/resources/scene_images/{str(image_counter)}.jpg"
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

                    player_2_target_name = ["second", "2", "2nd"]

                    player_a_prompt_header = player_a_prompt_header.replace("FIRST_IMAGE", "the target").replace(
                        "SECOND_IMAGE", "a distractor").replace("THIRD_IMAGE", "a distractor")

                elif i == 2:
                    player_1_first_image = distractor2_path
                    player_1_second_image = distractor1_path
                    player_1_third_image = target_image_path

                    player_2_first_image = target_image_path
                    player_2_second_image = distractor1_path
                    player_2_third_image = distractor2_path

                    player_2_target_name = ["first", "1", "1st"]

                    player_a_prompt_header = player_a_prompt_header.replace("FIRST_IMAGE", "a distractor").replace(
                        "SECOND_IMAGE", "a distractor").replace("THIRD_IMAGE", "the target")

                elif i == 3:
                    player_1_first_image = distractor2_path
                    player_1_second_image = target_image_path
                    player_1_third_image = distractor1_path

                    player_2_first_image = distractor2_path
                    player_2_second_image = distractor1_path
                    player_2_third_image = target_image_path

                    player_2_target_name = ["third", "3", "3rd"]

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
                    'player_2_response_pattern'] = '^answer:\s(?P<content>first|second|third|1|2|3|1st|2nd|3rd)\n*(?P<remainder>.*)'
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

    def generate_docci_instances(self):
        player_a_prompt_header = self.load_template(f"resources/initial_prompts/player_a_prompt_images.template")
        player_b_prompt_header = self.load_template(f"resources/initial_prompts/player_b_prompt_images.template")

        docci_dataset = self.get_docci_dataset()

        game_counter = 0
        image_counter = 1
        experiment = self.add_experiment('DOCCI_images')
        for target_category in docci_dataset:

            target_category_images = docci_dataset[target_category]
            target_image = self.select_random_item(target_category_images)
            shutil.copyfile(target_image, f"resources/docci_images/{str(image_counter)}.jpg")
            target_image_path = f"games/multimodal_referencegame/resources/docci_images/{str(image_counter)}.jpg"
            image_counter += 1

            # remove the target image from the list, select another image from the same category
            target_category_images.remove(target_image)
            distractor1 = self.select_random_item(target_category_images)
            shutil.copyfile(distractor1, f"resources/docci_images/{str(image_counter)}.jpg")
            distractor1_path = f"games/multimodal_referencegame/resources/docci_images/{str(image_counter)}.jpg"
            image_counter += 1

            # remove the distractor1 image from the list, select another image from the same category
            target_category_images.remove(distractor1)
            distractor2 = self.select_random_item(target_category_images)
            shutil.copyfile(distractor2, f"resources/docci_images/{str(image_counter)}.jpg")
            distractor2_path = f"games/multimodal_referencegame/resources/docci_images/{str(image_counter)}.jpg"
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

                    player_2_target_name = ["second", "2", "2nd"]

                    player_a_prompt_header = player_a_prompt_header.replace("FIRST_IMAGE", "the target").replace(
                        "SECOND_IMAGE", "a distractor").replace("THIRD_IMAGE", "a distractor")

                elif i == 2:
                    player_1_first_image = distractor2_path
                    player_1_second_image = distractor1_path
                    player_1_third_image = target_image_path

                    player_2_first_image = target_image_path
                    player_2_second_image = distractor1_path
                    player_2_third_image = distractor2_path

                    player_2_target_name = ["first", "1", "1st"]

                    player_a_prompt_header = player_a_prompt_header.replace("FIRST_IMAGE", "a distractor").replace(
                        "SECOND_IMAGE", "a distractor").replace("THIRD_IMAGE", "the target")

                elif i == 3:
                    player_1_first_image = distractor2_path
                    player_1_second_image = target_image_path
                    player_1_third_image = distractor1_path

                    player_2_first_image = distractor2_path
                    player_2_second_image = distractor1_path
                    player_2_third_image = target_image_path

                    player_2_target_name = ["third", "3rd", "3"]

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
                    'player_2_response_pattern'] = '^answer:\s(?P<content>first|second|third|1|2|3|1st|2nd|3rd)\n*(?P<remainder>.*)'
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

    def generate_clevr_instances(self):
        player_a_prompt_header = self.load_template(f"resources/initial_prompts/player_a_prompt_images.template")
        player_b_prompt_header = self.load_template(f"resources/initial_prompts/player_b_prompt_images.template")

        clevr_dataset = self.get_clevr_dataset()

        game_counter = 0
        image_counter = 1
        experiment = self.add_experiment('CLEVR_images')
        for target_category in clevr_dataset:

            target_category_images = clevr_dataset[target_category]
            target_image = self.select_random_item(target_category_images)
            shutil.copyfile(target_image, f"resources/clevr_images/{str(image_counter)}.jpg")
            target_image_path = f"games/multimodal_referencegame/resources/clevr_images/{str(image_counter)}.jpg"
            image_counter += 1

            # remove the target image from the list, select another image from the same category
            target_category_images.remove(target_image)
            distractor1 = self.select_random_item(target_category_images)
            shutil.copyfile(distractor1, f"resources/clevr_images/{str(image_counter)}.jpg")
            distractor1_path = f"games/multimodal_referencegame/resources/clevr_images/{str(image_counter)}.jpg"
            image_counter += 1

            # remove the distractor1 image from the list, select another image from the same category
            target_category_images.remove(distractor1)
            distractor2 = self.select_random_item(target_category_images)
            shutil.copyfile(distractor2, f"resources/clevr_images/{str(image_counter)}.jpg")
            distractor2_path = f"games/multimodal_referencegame/resources/clevr_images/{str(image_counter)}.jpg"
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

                    player_2_target_name = ["second", "2", "2nd"]

                    player_a_prompt_header = player_a_prompt_header.replace("FIRST_IMAGE", "the target").replace(
                        "SECOND_IMAGE", "a distractor").replace("THIRD_IMAGE", "a distractor")

                elif i == 2:
                    player_1_first_image = distractor2_path
                    player_1_second_image = distractor1_path
                    player_1_third_image = target_image_path

                    player_2_first_image = target_image_path
                    player_2_second_image = distractor1_path
                    player_2_third_image = distractor2_path

                    player_2_target_name = ["first", "1", "1st"]

                    player_a_prompt_header = player_a_prompt_header.replace("FIRST_IMAGE", "a distractor").replace(
                        "SECOND_IMAGE", "a distractor").replace("THIRD_IMAGE", "the target")

                elif i == 3:
                    player_1_first_image = distractor2_path
                    player_1_second_image = target_image_path
                    player_1_third_image = distractor1_path

                    player_2_first_image = distractor2_path
                    player_2_second_image = distractor1_path
                    player_2_third_image = target_image_path

                    player_2_target_name = ["third", "3rd", "3"]

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
                    'player_2_response_pattern'] = '^answer:\s(?P<content>first|second|third|1|2|3|1st|2nd|3rd)\n*(?P<remainder>.*)'
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
        self.generate_docci_instances()
        self.generate_clevr_instances()


if __name__ == '__main__':
    ReferenceGameInstanceGenerator().generate(filename="instances.json")
