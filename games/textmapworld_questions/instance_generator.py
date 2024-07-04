import os
import json
from os.path import exists
import random
import clemgame
from clemgame.clemgame import GameInstanceGenerator
from games.textmapworld_questions.utils import load_check_graph, generate_filename, create_graphs_file 
logger = clemgame.get_logger(__name__)

"Enter the parameters for the game instance generator"
"-------------------------------------------------------------------------------------------------------------"
"°°°°°°°changeable parameters°°°°°°°"
game_name = "textmapworld_questions"
size = 8
create_new_graphs = False # True or False   !if True, the graphs will be created again, threfore pay attention!
n = 4
m = 4
instance_number = 10
game_type = "named_graph" #"named_graph" or "unnamed_graph"
cycle_type="cycle_false" #"cycle_true" or "cycle_false"
strict = True 
if strict:
    DONE_REGEX = '^DONE$'
    MOVE_REGEX = '^GO:\s*(north|east|west|south)$'
    QA_REGEX = '^Answer:\s*(\d+)\s*$'
else:
    DONE_REGEX = 'DONE'
    MOVE_REGEX = 'GO:\s*(north|east|west|south)'
    QA_REGEX = "Answer:\s*(\d+)"
loop_reminder = False
max_turns_reminder = False
ambiguity_types =  {"none": [None], "limited": [(2,2), (1,2)], "strong": [(1,3), (2,3), (3,2)]}

"°°°°°°°imported parameters°°°°°°°"
prompt_file_name = 'PromptNamedGame.template'
prompt_file_name = os.path.join('resources', 'initial_prompts', prompt_file_name)
current_directory = os.getcwd().replace("\instance_generator", "")
with open(os.path.join(current_directory, "games", "textmapworld_questions", 'resources', 'initial_prompts', "answers.json")) as json_file:
    answers_file = json.load(json_file)
with open(os.path.join(current_directory, "games", "textmapworld_questions", 'resources', 'initial_prompts', "reminders.json")) as json_file:
    reminders_file = json.load(json_file)
"-------------------------------------------------------------------------------------------------------------"

class GraphGameInstanceGenerator(GameInstanceGenerator):

    def __init__(self,  ):
        super().__init__(game_name)

    def on_generate(self):
        rooms = ["Bedroom", "Living room", "Kitchen", "Bathroom", "Dining room", "Study room", "Guest room", "Game room", "Home office", "Laundry room", "Pantry", "Attic", 
                                "Basement", "Garage", "Sunroom", "Mudroom", "Closet", "Library", "Foyer", "Nursery", "Home gym", "Media room", "Home theater", "Playroom", "Utility room", "Workshop", 
                                "Conservatory", "Craft room", "Music room", "Gallery", "Sauna", "Billiard room", "Bar", "Wine cellar", "Cellar", "Exercise room", "Studio", "Recreation room", "Solarium", "Balcony"]
        
        player_a_prompt_header =  self.load_template(prompt_file_name)
        Player2_positive_answer = answers_file["PositiveAnswerNamedGame"] 
        Player2_negative_answer = answers_file["NegativeAnswerNamedGame"]
        game_id = 0
        for key, value in ambiguity_types.items():
            ambiguity = random.choice(value)
            experiment = self.add_experiment(key)
            created_name= generate_filename(game_type, size, cycle_type, key)
            file_graphs = os.path.join("games", "textmapworld_questions", 'files', created_name)
            if not create_new_graphs:
                if not os.path.exists(file_graphs):
                    raise ValueError("New graphs are not created, but the file does not exist. Please set create_new_graphs to True.")
            else:
                if os.path.exists(file_graphs):
                    raise ValueError("The file already exists, please set create_new_graphs to False.")
                create_graphs_file(file_graphs, instance_number, game_type, n, m, size, cycle_type, value)
                
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
                    game_instance["QA_Construction"] = QA_REGEX
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
                    game_instance["Question"] = reminders_file["question"]
                    new_nodes_list = [s.split("_")[0] for s in grid['Graph_Nodes']]
                    count_nodes = {item: new_nodes_list.count(item) for item in set(new_nodes_list)}
                    sorted_items = sorted(count_nodes.items(), key=lambda x: x[1], reverse=True)
                    available_choices = [item for item in rooms if item not in new_nodes_list]
                    random_choice = random.choice(available_choices)
                    game_instance["First_Question_Answer"] = str((sorted_items[0][0],sorted_items[0][1]))
                    game_instance["Second_Question_Answer"] = str((sorted_items[1][0],sorted_items[1][1]))
                    game_instance["Third_Question_Answer"] = str((random_choice ,0))
                    game_instance["Question_reprompt"] = str(reminders_file["question_rule"])
                    game_instance["Mapping"] = str(grid["Mapping"])
                    game_instance["Strict"] = strict
                    
                        

if __name__ == '__main__':
    # always call this, which will actually generate and save the JSON file
    GraphGameInstanceGenerator().generate()