"""
Creates test instances for MM games and their text only versions
"""

import json

# game_name = "textmapworld_description"

game_names = [
    "referencegame",
    "textmapworld",
    "textmapworld_graphreasoning",
    "textmapworld_questions",
    "textmapworld_specificroom",
    "textmapworld_description",
    "matchit",
    "matchit_1q",
    "matchit_5q",
    "matchit_info",
    "matchit_ascii",
    "matchit_ascii_1q",
    "matchit_ascii_5q",
    "matchit_ascii_info",
    "multimodal_referencegame",
    "mm_mapworld",
    "mm_mapworld_qa",
    "mm_mapworld_specificroom",
    "mm_mapworld_graphs"
]

def gen_test_instances(game_name: str):
    """Generates test instances for a given game.

    Args:
        game_name (str): The name of the game for which to generate test instances.

    This function reads game instance data from JSON files, processes it, and writes the test data to a new JSON file.
    """
    with open(f'games/{game_name}/in/instances.json', 'r') as f:
        json_data = json.load(f)

    pentomino_data = None
    if "matchit" in game_name and "ascii" not in game_name:
        splits = game_name.split("_")
        if len(splits) != 1:
            with open(f'games/{game_name}/in/instances_{splits[-1]}_pentomino.json', 'r') as f:
                pentomino_data = json.load(f)
        else:
            with open(f'games/{game_name}/in/instances_base_pentomino.json', 'r') as f:
                pentomino_data = json.load(f)
            

    test_data = {
        "experiments": [

        ]
    }

    exps = json_data['experiments']
    for e in exps:

        insts = e["game_instances"]
        dict_obj = e         
        dict_obj["game_instances"] = []
        for i in insts:
            if "textmapworld"  in game_name and "description" not in game_name:
                if i["game_id"]%10 == 0: # text_map
                    dict_obj["game_instances"].append(i)
            else:
                if i["game_id"] == 0: # mm_ref
                    dict_obj["game_instances"].append(i)
                

        test_data["experiments"].append(dict_obj)

    if pentomino_data:
        exps = pentomino_data['experiments']
        for e in exps:
            insts = e["game_instances"]
            dict_obj = e         
            dict_obj["game_instances"] = []
            for i in insts:
                if i["game_id"] == 0:
                    dict_obj["game_instances"].append(i)

            test_data["experiments"].append(dict_obj)


    with open(f'games/{game_name}/in/test.json', 'w') as f:
        json.dump(test_data, f, indent=4)

    
for g in game_names:
    gen_test_instances(g)