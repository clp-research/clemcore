from clemgame.clemgame import GameInstanceGenerator
import numpy as np
from maps import AbstractMap
import os
import random
import json
import networkx as nx
from copy import deepcopy
import shutil


# set the name of the game in the script, as you named the directory
# this name will be used everywhere, including in the table of results
GAME_NAME = 'mm_mapworld_qa'
NUM_INSTANCES = 10
GRIDS = {"small": (3,3), "medium": (3,4), "large": (4,4)}
SIZES = {"small": 4, "medium": 6, "large": 8}
DISTS = {"on": [0], "close": [1,2], "far": [3,4]}
AMBIGUITIES = {"none": [(1,1)], "limited": [(2,2), (1,2)], "strong": [(1,3), (2,3), (3,2)]}
SEED = 42
RANDOM_PATH = 'random_test_images'
IMAGE_PATH = os.path.join('games', 'mm_mapworld', 'resources', 'images')
# The dataset annotation is in english, making the language agnostic is going to be more challenging
MAPPING_PATH = os.path.join("games", "mm_mapworld", "resources", "ade_20k", "ade_cat_instances.json")
DATASET_PATH = os.path.join("games", "mm_mapworld", "resources", "ade_20k", "needed_imgs")
TEMP_IMAGE_PATH = os.path.join("games", "mm_mapworld_qa", "resources", "images")
RESPONSE_REGEX = "^\{[\s]*\"description\":\s*\"([^\{]*?)\"\s*,\s*\"action\":\s*\"([^\{]*?)\"[\s]*\}$"
MOVE_CONSTRUCTION = "GO: "
FOUND_REGEX = "^DONE$"
MOVE_REGEX = "^GO:\s*(north|east|south|west)$"
QA_REGEX = "^Answer:\s*(\d+)$"



def create_instances(grid_size = GRIDS['large'], graph_size = SIZES['medium'], num_instances = NUM_INSTANCES, ambiguity = AMBIGUITIES["limited"]):
    instances = []
    np.random.seed(SEED)
    random.seed(SEED)
    for i in range(num_instances):
        map = AbstractMap(*grid_size, graph_size)
        nodes = [str(n) for n in map.G]
        start = np.random.choice(nodes)
        edges = list(map.G.edges())
        rev_edges = [(edge[1], edge[0]) for edge in edges]
        edges.extend(rev_edges)
        index = np.random.randint(len(ambiguity))
        this_ambiguity = ambiguity[index]
        img_ref, cat_ref, questions = assign_images(nodes, this_ambiguity)
        instances.append({
            'nodes': nodes,
            'edges': [str(e) for e in edges],
            'imgs': img_ref,
            'cats': cat_ref,
            'start': start,
            'use_images': True,
            'reprompt': False,
            'use_loop_warning': True,
            'use_turn_limit_warning': True,
            'questions': questions,
            'ambiguity': this_ambiguity
        })
    return instances

def assign_images(nodes, ambiguity, num_targets = 1):
    # load the category file
    with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    cats = mapping.keys()
    # outdoor images don't make too much sense for rooms in a house
    cats_inside = [cat for cat in cats if 'outdoor' not in cat]
    num_cats_needed = len(nodes) - (ambiguity[0] * (ambiguity[1] - 1))
    chosen_cats = np.random.choice(cats_inside, size = num_cats_needed, replace = False)
    # make sure the decoy category does not exist on the graph
    for c in chosen_cats:
        cats_inside.remove(c)
    chosen_cats = list(chosen_cats)
    # choose it randomly from the rest of the categories
    decoy = np.random.choice(cats_inside)
    imgs = {}
    cat_mapping = {}
    targets = chosen_cats[:2]
    targets.append(decoy)
    l = ambiguity[0] * [ambiguity[1]] 
    while sum(l) < len(nodes):
        l.append(1)
    nodes_per_cat = {chosen_cats[j]: l[j] for j in range(len(chosen_cats))}
    targets = {
        chosen_cats[0].split("/")[1]: nodes_per_cat[chosen_cats[0]],
        chosen_cats[1].split("/")[1]: nodes_per_cat[chosen_cats[1]],
        decoy.split("/")[1]: 0,
    }
    questions = []
    for target in targets:
        questions.append({"q": f"How many different {target.replace('_', ' ')}(s) did we encounter?", "a": str(targets[target])})
    nodes_copy = deepcopy(nodes)
    for c in chosen_cats:
        chosen_nodes = list(np.random.choice(nodes_copy, size=nodes_per_cat[c], replace = False))
        chosen_imgs = np.random.choice(mapping[c], size=nodes_per_cat[c])
        for i in range(len(chosen_nodes)):
            after_copy_path = copy_image(os.path.join(DATASET_PATH, chosen_imgs[i]))
            imgs[chosen_nodes[i]] = after_copy_path
            cat_mapping[chosen_nodes[i]] = c.split("/")[1]
            nodes_copy.remove(chosen_nodes[i])
    return imgs, cat_mapping, questions
    

def instance_from_args(args, prompts):
    instances = create_instances(
        grid_size=GRIDS[args.get('size', 'large')],
        graph_size=SIZES[args.get('size', 'large')],
        ambiguity=AMBIGUITIES[args.get('ambiguity', 'limited')],
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
        instances[i]["qa_init"] = prompts["init_qa"]
        
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

class MmMapWorldQAInstanceGenerator(GameInstanceGenerator):
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
            'init_qa': self.load_template('resources/qa_prompts/init.template'),
            'question_qa': self.load_template('resources/qa_prompts/question.template'),
        }
        experiments = {
            'none': {"ambiguity": "none", "one_shot": True, "reprompt": False},
            'limited': {"ambiguity": "limited", "one_shot": True, "reprompt": False},
            'strong': {"ambiguity": "strong", "one_shot": True, "reprompt": False}
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
                 instance["qa_regex"] = QA_REGEX
                 game_id += 1

if __name__ == '__main__':
    # always call this, which will actually generate and save the JSON file
    MmMapWorldQAInstanceGenerator().generate()

