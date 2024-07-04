import ast
import os
import numpy as np
import networkx as nx 
from clemgame import  string_utils
from games.textmapworld_questions.graph_generator import GraphGenerator

"----------------------------------------------------"
"The functions used in instance_generator.py"

def generate_filename(game_type, graph_size, cycle_type, ambiguity):

    if cycle_type == "cycle_true":
        cycle = "True"
    elif cycle_type == "cycle_false":
        cycle = "False"
    if graph_size==None:
        filename_parts = [game_type.capitalize().split("_graph")[0], cycle, str(ambiguity)]
        filename = "_".join(filename_parts)
    else:
        filename_parts = [game_type.capitalize().split("_graph")[0], str(graph_size), cycle, str(ambiguity)]
        filename = "_".join(filename_parts) + ".txt"
    return filename


def load_check_graph(file_graphs, instance_number, game_type):
    grids = []
    with open(str(file_graphs), 'r') as file:
        for c, line in enumerate(file):
            line = line.rstrip()
            doc = ast.literal_eval(line)
            nodes = doc.get('Graph_Nodes', [])
            check_set = set()
            if c < instance_number:
                if all(isinstance(item, tuple) for item in nodes):
                    checked_graph_type = "unnamed_graph"
                    check_set.add(checked_graph_type)
                elif all(isinstance(item, str) for item in nodes):
                    checked_graph_type = "named_graph"
                    check_set.add(checked_graph_type)
                grids.append(doc)
        if  check_set != {str(game_type)}:
            raise ValueError("Graph type does not match the specified type")
    return grids



def create_graphs_file(graphs_file_name, num_graphs, graph_type, n, m, rooms, cycle_bool, abiguity):
    
    generated_graphs = 0
    with open(graphs_file_name, "w") as f:
        file_length = os.path.getsize(graphs_file_name)
        while generated_graphs < num_graphs:
            new_instance = GraphGenerator(graph_type, n, m, rooms, cycle_bool, abiguity)
            result = new_instance.generate_instance()
            if result != "No graph generated":
                f.write(str(result) + "\n")
                generated_graphs+=1
    # Check if file is empty
    if os.path.getsize(graphs_file_name) == 0:
        raise ValueError("Generated file is empty")
    return graphs_file_name

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
                                        

def loop_identification(visited_nodes):
    if len(visited_nodes) >= 4:
        if len(set(visited_nodes[-4:])) < 3:
            return True
    return False


def calculate_distance(nodes, edges, node1, node2):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    try:
        distance = nx.shortest_path_length(graph, source=node1, target=node2)
        return distance
    except nx.NetworkXNoPath:
        return "No path exists between the nodes"
    
def count_word_in_sentence(sentence, word):
    # Split the sentence into words
    words = sentence.split()
    
    # Convert the words and the target word to lowercase for case insensitive comparison
    word = word.lower()
    words = [w.lower() for w in words]
    
    # Count the occurrences of the word
    count = words.count(word)
    return count