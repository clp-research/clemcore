import os
import json
import random
from os.path import exists
import networkx as nx
import clemgame
from clemgame.clemgame import GameInstanceGenerator
from games.textmapworld_specificroom.utils import load_check_graph, generate_filename, create_graphs_file, create_graph
logger = clemgame.get_logger(__name__)

"Enter the parameters for the game instance generator"
"-------------------------------------------------------------------------------------------------------------"
"°°°°°°°changeable parameters°°°°°°°"
game_name = "textmapworld_specificroom"
strict = True
create_new_graphs = False # True or False   !if True, the graphs will be created again, threfore pay attention!
size = 8        #"large"
n = 4
m = 4
instance_number = 10
game_type = "named_graph" #"named_graph" or "unnamed_graph"
cycle_type="cycle_false" #"cycle_true" or "cycle_false"
ambiguity= None #(repetition_rooms, repetition_times) or None
if strict:
    DONE_REGEX = '^DONE$'
    MOVE_REGEX = '^GO:\s*(north|east|west|south)$'
else:
    DONE_REGEX = 'DONE'
    MOVE_REGEX = 'GO:\s*(north|east|west|south)'
loop_reminder = False
max_turns_reminder = False
distances = {"on": [0], "close": [1,2], "far": [3,4]}

"°°°°°°°imported parameters°°°°°°°"
prompt_file_name = 'PromptNamedGame.template' if game_type == "named_graph" else 'PromptUnnamedGame.template'
prompt_file_name = os.path.join('resources', 'initial_prompts', prompt_file_name)
current_directory = os.getcwd().replace("\instance_generator", "")
with open(os.path.join(current_directory, "games", "textmapworld_specificroom", 'resources', 'initial_prompts', "answers.json")) as json_file:
    answers_file = json.load(json_file)
with open(os.path.join(current_directory, "games", "textmapworld_specificroom", 'resources', 'initial_prompts', "reminders.json")) as json_file:
    reminders_file = json.load(json_file)
"-------------------------------------------------------------------------------------------------------------"

class GraphGameInstanceGenerator(GameInstanceGenerator):

    def __init__(self,  ):
        super().__init__(game_name)

    def on_generate(self):

        created_name= generate_filename(game_type, size, cycle_type, ambiguity)
        file_graphs = os.path.join("games", "textmapworld_specificroom", 'files', created_name)
        if not create_new_graphs:
            if not os.path.exists(file_graphs):
                raise ValueError("New graphs are not created, but the file does not exist. Please set create_new_graphs to True.")
        else:
            if os.path.exists(file_graphs):
                raise ValueError("The file already exists, please set create_new_graphs to False.")
            create_graphs_file(file_graphs, instance_number, game_type, n, m, size, cycle_type, ambiguity)
        game_id = 0
        player_a_prompt_header =  self.load_template(prompt_file_name)
        Player2_positive_answer = answers_file["PositiveAnswerNamedGame"] 
        Player2_negative_answer = answers_file["NegativeAnswerNamedGame"]
        for key, value in distances.items():
            experiment = self.add_experiment(key)
            if os.path.exists(file_graphs):
                grids = load_check_graph(file_graphs, instance_number, game_type)
                for grid in grids:
                    game_instance = self.add_game_instance(experiment, game_id)
                    game_id += 1
                    game_instance["Prompt"] = player_a_prompt_header
                    game_instance["Player2_positive_answer"] = Player2_positive_answer
                    game_instance["Player2_negative_answer"] = Player2_negative_answer
                    game_instance["Move_Construction"] = MOVE_REGEX
                    game_instance["Stop_Construction"] = DONE_REGEX
                    game_instance["Grid_Dimension"] = str(grid["Grid_Dimension"])
                    game_instance['Graph_Nodes'] = str(grid['Graph_Nodes'])
                    game_instance['Graph_Edges'] = str(grid['Graph_Edges'])
                    game_instance['Current_Position'] = str(grid['Initial_Position'])
                    game_instance['Picture_Name'] = grid['Picture_Name']
                    game_instance["Directions"] = str(grid["Directions"])
                    game_instance["Moves"] = str(grid["Moves"])
                    game_instance['Cycle'] = grid['Cycle']
                    game_instance['Ambiguity'] = grid['Ambiguity']
                    game_instance['Game_Type'] = game_type
                    game_instance["Loop_Reminder"] = loop_reminder
                    game_instance["Loop_Reminder_Text"] = reminders_file["loop_reminder"]
                    game_instance["Max_Turns_Reminder"] = max_turns_reminder
                    game_instance["Max_Turns_Reminder_Text"] = reminders_file["max_turns_reminder"]
                    game_instance["Mapping"] = str(grid["Mapping"])
                    game_instance["Strict"] = strict
                    generated_graph = create_graph(grid["Graph_Nodes"], grid["Graph_Edges"])
                    dists = dict(nx.all_pairs_shortest_path_length(generated_graph))
                    random_distance = random.choice(value)
                    distance_found = False
                    for k,v in dists.items():
                        if k == grid["Initial_Position"]:
                            for neighbor, distance in v.items():
                                if distance == random_distance:
                                    game_instance["Specific_Room"] = neighbor
                                    game_instance["Specific_Room_Distance"] = str(random_distance)
                                    distance_found = True
                                    break
                    if not distance_found:
                        for room, val in dists.items():
                            for neighbor, distance in val.items():
                                if distance == random_distance:
                                    game_instance['Current_Position'] = room
                                    game_instance["Specific_Room"] = neighbor
                                    game_instance["Specific_Room_Distance"] = str(random_distance)
                                    break
                            
                        

if __name__ == '__main__':
    # always call this, which will actually generate and save the JSON file
    GraphGameInstanceGenerator().generate()