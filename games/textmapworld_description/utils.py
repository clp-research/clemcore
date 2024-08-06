import numpy as np
from clemgame import string_utils
import random
from os.path import exists
import matplotlib.pyplot as plt
import networkx as nx
import ast

"----------------------------------------------------"
"The functions used in instance_generator.py"
opposite_direction_dict = {'north': 'south', 'south': 'north', 'east': 'west', 'west': 'east'}
dir2delta = {'north': np.array((0, 1)), 'south': np.array((0, -1)), 'east': np.array((1, 0)), 'west': np.array((-1, 0))}

def calculate_directions(nodes, edges):
    directions = {node: [] for node in nodes}
    for node in nodes:
        for direction, delta in dir2delta.items():
            neighbor = tuple(np.array(node) + delta)
            if neighbor in nodes and (node, neighbor) in edges or (neighbor, node) in edges:
                directions[node].append(direction)
    return directions

def generate_moves(directions):
    moves = []
    for node, dirs in directions.items():
        node_moves = [(direction, tuple(np.array(node) + dir2delta[direction])) for direction in dirs]
        moves.append({'node': node, 'node_moves': node_moves})
    return moves


def draw_graph(nodes, edges, cats, path):
    # Create a graph object
    G = nx.Graph()
    # Add nodes to the graph
    G.add_nodes_from(nodes)
    # Add edges to the graph
    G.add_edges_from(edges)
    # Create a layout for the nodes
    pos = {node: node for node in nodes}
    # Draw the graph
    labels = {ast.literal_eval(k): v for k, v in cats.items()}
    plt.figure(figsize=(8, 8))
    nx.set_node_attributes(G, labels, 'label')
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, labels=labels)

    picture_number = random.randint(0, 10000)
    picture_name = "map_" + str(picture_number) + ".png"
    # get the current working directory
    file_exists = exists(path + picture_name)
    if file_exists:
        picture_name = "map_" + str(picture_number + 1) + ".png"
    plt.savefig(path + picture_name)
    plt.clf()

    return picture_name


# Function to generate the complete information
def generate_graph_info(nodes, edges, captions, picture_path):
    dirs = calculate_directions(nodes, edges)
    directions =[(node, dirs[node]) for node in nodes]
    moves = generate_moves(dirs)
    
    renamed_nodes=[]
    for node in list(nodes):
        renamed_nodes.append(captions[str(node)])

    renamed_edges=[]
    for edge in list(edges):
        renamed_edge=(captions[str(edge[0])], captions[str(edge[1])])
        renamed_edges.append(renamed_edge)

    renamed_graph_directions= []
    for path in directions:
        renamed_graph_directions.append((captions[str(path[0])], path[1]))

    renamed_moves_nodes_list=[]
    for move in moves:
        each_move = []
        renamed_node = captions[str(move['node'])]
        for movement in move['node_moves']:
            change = captions[str(movement[1])]
            each_move.append((movement[0], change))
            renamed_moves_nodes_list.append({"node": renamed_node, "node_moves":each_move})
    
    picture_name = draw_graph(nodes, edges, captions, picture_path)

    graph_info = {
        'Graph_Nodes': renamed_nodes,
        'Graph_Edges': renamed_edges,
        'N_edges': len(renamed_edges),
        'Directions': renamed_graph_directions ,
        'Moves': renamed_moves_nodes_list,
        'Picture': picture_name}
    
    return graph_info

def generate_descriptions(imgs,cats,captions):
    """
    Generate descriptions for images based on their captions and categories.
    
    Parameters:
    imgs (dict): A dictionary where keys are labels and values are image paths.
    captions (dict): A dictionary where keys are image paths and values are captions.
    cats (dict): A dictionary mapping labels to categories.
    
    Returns:
    dict: A dictionary mapping categories to descriptions.
    """
    descriptions = {}
    for label, img in imgs.items():
        image = img.split("/")[-1]
        for key, value in captions.items():
            if key.split("/")[-1] == image:
                descriptions[cats[label]] = value
                break  # Exit the loop once a match is found
    return descriptions


"----------------------------------------------------"
"The functions used in master.py"

def get_directions(node, direction_list, saved_node):

    if saved_node != node:
        node = saved_node
    node_directions = None  
    for i in direction_list:
        if i[0]==node:
            node_directions=i[1]
            break
    return node_directions

def string_available_directions(word_list):
    return ', '.join(word_list)

def have_common_element(str1, str2):
    common_elements = ["east", "west", "north", "south"]
    # Convert strings to sets of words
    words1 = set(str1.split())
    words2 = set(str2.split())
    # Check for exact matches
    common_matches = words1.intersection(words2).intersection(common_elements)
    # Return True if there is at least one exact match
    return any(match in common_elements for match in common_matches)



def get_nextnode_label(moves, node, utterance, move_construction):
    next_label=None
    utterance = utterance.strip()
    for move in moves:
        if move["node"]==node:
            moves_node=move['node_moves']
            for step in moves_node:
                move_type = step[0]
                if move_type ==utterance:
                    next_label=step[1]
    return next_label, move_type

def ambiguity_move(old_one, new_one, mapping, moves, move_type):
    
    mapping_dict, label_moves = mapping
    for label, name in mapping_dict.items():
        if name == old_one:
            for move in moves:
                if name == move["node"]:
                    for label_move in label_moves:
                        if label_move["node"] == label:
                            new_labelled_moves = [(m[0], mapping_dict[m[1]]) for m in label_move["node_moves"]]
                            for m in new_labelled_moves:
                                if m[1] == new_one and m[0] == move_type:
                                    for s in label_move["node_moves"]:
                                        if s[0] == move_type:
                                            return label, s[1]
                                        

def loop_identification(visited_rooms, double_cycle):
    flag_loop = False
    if not double_cycle:
        if len(visited_rooms) >= 5:
            if visited_rooms[-1] == visited_rooms[-5] and visited_rooms[-2] == visited_rooms[-4] and visited_rooms[-3] == visited_rooms[-5]:
                flag_loop = True
    elif double_cycle:
        if len(visited_rooms) >= 10:
            l1, l2=np.array_split(visited_rooms[-10:] , 2)
            if all(l1==l2):
                if l1[-1] == l1[-5] and l1[-2] == l1[-4] and l1[-3] == l1[-5]:
                    flag_loop = True
    return flag_loop

def count_word_in_sentence(sentence, word):
    # Split the sentence into words
    words = sentence.split()
    
    # Convert the words and the target word to lowercase for case insensitive comparison
    word = word.lower()
    words = [w.lower() for w in words]
    
    # Count the occurrences of the word
    count = words.count(word)
    return count