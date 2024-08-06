from clemgame.clemgame import GameInstanceGenerator
import numpy as np
import networkx as nx
from maps import AbstractMap
import os
import random
import json
import shutil


# set the name of the game in the script, as you named the directory
# this name will be used everywhere, including in the table of results
GAME_NAME = 'mm_mapworld_graphs'
NUM_INSTANCES = 10
GRIDS = {"small": (4,4), "medium": (4,4), "large": (4,4)}
SIZES = {"small": 4, "medium": 6, "large": 8} # num_nodes
SEED = 42
RANDOM_PATH = 'random_test_images'
IMAGE_PATH = os.path.join('games', 'mm_mapworld', 'resources', 'images')
DATASET_PATH = os.path.join("games", "mm_mapworld", "resources", "ade_20k", "needed_imgs")
MAPPING_PATH = os.path.join("games", "mm_mapworld", "resources", "ade_20k", "ade_cat_instances.json")
TEMP_IMAGE_PATH = os.path.join("games", "mm_mapworld_graphs", "resources", "images")
MOVE_CONSTRUCTION = "GO: "
STOP_CONSTRUCTION = "DONE"
GRAPH_REGEX = "\"graph\":\s*(\{\s*\"nodes\"\s*:\s*\[.*\]\s*,\s*\"edges\"\s*:\s*\{.*\})\s*\}$"
RESPONSE_REGEX = "^\{[\s]*\"action\":\s*\"([^\{]*?)\"\s*,\s*\"description\":\s*\"([^\{]*?)\"[\s]*,\s*"
RESPONSE_REGEX += GRAPH_REGEX
DONE_REGEX = '^DONE$'
MOVE_REGEX = '^GO:\s*(north|east|west|south)$'



def create_instances(grid_size, graph_size, num_instances = NUM_INSTANCES):
    instances = []
    np.random.seed(SEED)
    random.seed(SEED)
    for i in range(num_instances):
        map = AbstractMap(*grid_size, graph_size)   
        nodes = [str(n) for n in map.G]
        edges = list(map.G.edges())
        rev_edges = [(edge[1], edge[0]) for edge in edges]
        edges.extend(rev_edges)
        img_ref, cat_ref = assign_images(nodes)
        instances.append({
            'nodes': nodes,
            'edges': [str(e) for e in edges],
            'imgs': img_ref,
            'cats': cat_ref,
            'start': random.choice(nodes),
            'use_loop_warning': True,
            'use_turn_limit_warning': True
        })
    return instances

def assign_images(nodes):
    with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    cats = mapping.keys()
    cats_inside = [cat for cat in cats if 'outdoor' not in cat]
    chosen_cats = np.random.choice(cats_inside, size=len(nodes))
    imgs = {}
    cat_mapping = {}
    for i in range(len(nodes)):
        cat_mapping[nodes[i]] = chosen_cats[i].split("/")[1]
        node_img = np.random.choice(mapping[chosen_cats[i]])
        after_copy_path = copy_image(os.path.join(DATASET_PATH, node_img))
        imgs[nodes[i]] = after_copy_path
    return imgs, cat_mapping

def instance_from_args(args, prompts):
    instances = create_instances(
        grid_size=GRIDS[args.get('size', 'large')],
        graph_size=SIZES[args.get('size', 'large')],
        num_instances=args.get('num_instances', NUM_INSTANCES)
    )
    for i in range(len(instances)):
        if args.get('one_shot', 0):
            instances[i]['initial_prompt'] = prompts['initial_one_shot']
        else:
            instances[i]['initial_prompt'] = prompts['initial']
        instances[i]['success_response'] = prompts['later_success']
        instances[i]['invalid_response'] = prompts['later_invalid']
        instances[i]['reprompt'] = args.get('reprompt', False)
        instances[i]['use_images'] = args.get('use_images', True)
        instances[i]["reprompt_format"] = prompts["reprompt_format"]
        instances[i]["limit_warning"] = prompts["limit_warning"]
        instances[i]["loop_warning"] = prompts["loop_warning"]
    return instances

def instances_from_instances(instances, prompts):
    for i in range(len(instances)):
        instances[i]['initial_prompt'] = prompts['initial_one_shot']
        instances[i]['success_response'] = prompts['later_success']
        instances[i]['invalid_response'] = prompts['later_invalid']
        instances[i]["reprompt_format"] = prompts["reprompt_format"]
        instances[i]["limit_warning"] = prompts["limit_warning"]
        instances[i]["loop_warning"] = prompts["loop_warning"]
    return instances

def prep_image_dir():
    if os.path.exists(TEMP_IMAGE_PATH):
        shutil.rmtree(TEMP_IMAGE_PATH)
    os.makedirs(TEMP_IMAGE_PATH)
    
def copy_image(image_path):
    filename = os.path.split(image_path)[1]
    src = image_path
    tgt = os.path.join(TEMP_IMAGE_PATH, filename)
    shutil.copy(src, tgt)
    return tgt
        

class MmMapWorldGraphsInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        # always do this to initialise GameInstanceGenerator
        super().__init__(GAME_NAME)
    def on_generate(self):
        prompts = {
            'initial': self.load_template('resources/initial_prompts/prompt.template'),
            'initial_one_shot': self.load_template('resources/initial_prompts/prompt_one_shot.template'),
            'later_success': self.load_template('resources/later_prompts/successful_move.template'),
            'later_invalid': self.load_template('resources/later_prompts/invalid_move.template'),
            'reprompt_format': self.load_template('resources/reprompts/invalid_format.template'),
            'limit_warning': self.load_template('resources/later_prompts/turn_limit.template'),
            'loop_warning': self.load_template('resources/later_prompts/loop.template'),
        }
        experiments = {
            'small': {"size": "small", "reprompt": False, "one_shot": True},
            'medium': {"size": "medium", "reprompt": False, "one_shot": True},
            'large': {"size": "large", "reprompt": False, "one_shot": True}
        }

        prep_image_dir()
        base_instance_path = os.path.join("games", "mm_mapworld", "in", "instances.json")
        
        if os.path.exists(base_instance_path):
            with open(base_instance_path, 'r') as f:
                base_instances = json.load(f)
            images_path = os.path.join("games", "mm_mapworld", "resources", "images")
            assert os.path.exists(images_path), "run instancegenerator for mm_mapworld to create images directory."
            new_instances = {"experiments": base_instances["experiments"][:3]}
            for i in range(len(new_instances["experiments"])):
                new_instances["experiments"][i]["game_instances"] = instances_from_instances(new_instances["experiments"][i]["game_instances"], prompts)
                for j in range(len(new_instances["experiments"][i]["game_instances"])):
                    new_instances["experiments"][i]["game_instances"][j]["response_regex"] = RESPONSE_REGEX
            new_instance_path = os.path.join("games", "mm_mapworld_graphs", "in", "instances.json")
            with open(new_instance_path, "w", encoding='utf-8') as f:
                json.dump(new_instances, f)
            if os.path.exists(TEMP_IMAGE_PATH):
                shutil.rmtree(TEMP_IMAGE_PATH)
            exit(1)
        
        else:
            for exp in experiments.keys():
                experiment = self.add_experiment(exp)
                game_id = 0
                generated_instances = instance_from_args(experiments[exp], prompts)
                for inst in generated_instances:
                    instance = self.add_game_instance(experiment, game_id)
                    for key, value in inst.items():
                        instance[key] = value
                    instance["move_construction"] = MOVE_CONSTRUCTION
                    instance["stop_construction"] = STOP_CONSTRUCTION
                    instance["response_regex"] = RESPONSE_REGEX
                    instance["done_regex"] = DONE_REGEX
                    instance["move_regex"] = MOVE_REGEX
                    game_id += 1

if __name__ == '__main__':
    # always call this, which will actually generate and save the JSON file
    MmMapWorldGraphsInstanceGenerator().generate()

