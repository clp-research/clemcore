from clemgame.clemgame import GameInstanceGenerator
import numpy as np
from maps import AbstractMap
import os
import random
import json
import networkx as nx
import shutil


# set the name of the game in the script, as you named the directory
# this name will be used everywhere, including in the table of results
GAME_NAME = 'mm_mapworld_specificroom'
NUM_INSTANCES = 10
GRIDS = {"small": (3,3), "medium": (3,4), "large": (4,4)}
SIZES = {"small": 4, "medium": 6, "large": 8}
DISTS = {"on": [0], "close": [1,2], "far": [3,4]}
SEED = 42
RANDOM_PATH = 'random_test_images'
IMAGE_PATH = os.path.join('games', 'mm_mapworld', 'resources', 'images')
# The dataset annotation is in english, making the language agnostic is going to be more challenging
MAPPING_PATH = os.path.join("games", "mm_mapworld", "resources", "ade_20k", "ade_cat_instances.json")
DATASET_PATH = os.path.join("games", "mm_mapworld", "resources", "ade_20k", "needed_imgs")
TEMP_IMAGE_PATH = os.path.join("games", "mm_mapworld_specificroom", "resources", "images")
RESPONSE_REGEX = "^\{[\s]*\"description\":\s*\"([^\{]*?)\"\s*,\s*\"action\":\s*\"([^\{]*?)\"[\s]*\}$"
MOVE_CONSTRUCTION = "GO: "
FOUND_REGEX = "^DONE$"
MOVE_REGEX = "^GO:\s*(north|east|south|west)$"


def create_instances(grid_size = GRIDS['large'], graph_size = SIZES['large'], num_instances = NUM_INSTANCES, goal_dist = DISTS["close"]):
    instances = []
    np.random.seed(SEED)
    random.seed(SEED)
    for i in range(num_instances):
        this_dist = int(np.random.choice(goal_dist))
        start = None
        target = None
        while start is None:
            map = AbstractMap(*grid_size, graph_size)
            dists = dict(nx.all_pairs_shortest_path_length(map.G))
            for node1 in dists:
                for node2 in dists[node1]:
                    if dists[node1][node2] == this_dist:
                        start = str(node1)
                        target = str(node2)
        nodes = [str(n) for n in map.G]
        edges = list(map.G.edges())
        rev_edges = [(edge[1], edge[0]) for edge in edges]
        edges.extend(rev_edges)
        img_ref, cat_ref = assign_images(nodes, target)
        instances.append({
            'nodes': nodes,
            'edges': [str(e) for e in edges],
            'imgs': img_ref,
            'cats': cat_ref,
            'start': start,
            'target': target,
            'target_cat': cat_ref[target],
            'dist': this_dist,
            'use_images': True,
            'reprompt': False,
            'use_loop_warning': True,
            'use_turn_limit_warning': True
        })
    return instances

def assign_images(nodes, target, num_targets = 1):
    with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    cats = mapping.keys()
    cats_inside = [cat for cat in cats if 'outdoor' not in cat]
    target_cat = np.random.choice(cats_inside)
    cats_inside.remove(target_cat)
    target_img = np.random.choice(mapping[target_cat])
    after_copy_path = copy_image(os.path.join(DATASET_PATH, target_img))
    imgs = {target: after_copy_path}
    cat_mapping = {target: target_cat.split("/")[1]}
    for node in nodes:
        if node == target:
            continue
        node_cat = np.random.choice(cats_inside)
        node_img = np.random.choice(mapping[node_cat])
        after_copy_path = copy_image(os.path.join(DATASET_PATH, node_img))
        imgs[node] = after_copy_path
        cat_mapping[node] = node_cat.split("/")[1]
    return imgs, cat_mapping
    

def instance_from_args(args, prompts):
    instances = create_instances(
        grid_size=GRIDS[args.get('size', 'large')],
        graph_size=SIZES[args.get('size', 'large')],
        goal_dist=DISTS[args.get('dist', 'medium')],
        num_instances=args.get('num_instances', NUM_INSTANCES)
    )
    for i in range(len(instances)):
        if args.get('one_shot', 0):
            instances[i]['initial_prompt'] = prompts['initial_one_shot']
        else:
            instances[i]['initial_prompt'] = prompts['initial']
        instances[i]['success_response'] = prompts['later_success']
        instances[i]['invalid_response'] = prompts['later_invalid']
        if args.get('reprompt', 0):
            instances[i]['reprompt'] = True
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
        

class MmMapWorldInstanceGenerator(GameInstanceGenerator):
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
            'on': {"dist": "on", "one_shot": True, "reprompt": False},
            'close': {"dist": "close", "one_shot": True, "reprompt": False},
            'far': {"dist": "far", "one_shot": True, "reprompt": False}
        }

        prep_image_dir()
        for exp in experiments.keys():
             experiment = self.add_experiment(exp)
             game_id = 0
             generated_instances = instance_from_args(experiments[exp], prompts)
             for inst in generated_instances:
                 instance = self.add_game_instance(experiment, game_id)
                 for key, value in inst.items():
                     instance[key] = value
                 instance["move_construction"] = MOVE_CONSTRUCTION
                 instance["done_regex"] = FOUND_REGEX
                 instance["move_regex"] = MOVE_REGEX
                 instance["response_regex"] = RESPONSE_REGEX
                 game_id += 1

if __name__ == '__main__':
    # always call this, which will actually generate and save the JSON file
    MmMapWorldInstanceGenerator().generate()

