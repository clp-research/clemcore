import os
import json
from os.path import exists
import clemgame
import ast
from clemgame.clemgame import GameInstanceGenerator
from games.textmapworld_description.utils import generate_graph_info, generate_descriptions
logger = clemgame.get_logger(__name__)

"Enter the parameters for the game instance generator"
"-------------------------------------------------------------------------------------------------------------"
"°°°°°°°changeable parameters°°°°°°°"
game_name = "textmapworld_description"
strict = True 
if strict:
    DONE_REGEX = '^DONE$'
    MOVE_REGEX = '^GO:\s*(north|east|west|south)$'
else:
    DONE_REGEX = 'DONE'
    MOVE_REGEX = 'GO:\s*(north|east|west|south)'
    
create_new_graphs = False 
n = 4
m = 4
instance_number = 10
loop_reminder = False
max_turns_reminder = False
experiments_details = {"small": (4,"cycle_false"), "medium": (6, "cycle_false"), "large": (8, "cycle_false"), "medium_cycle": (6, "cycle_true"), "large_cycle": (8, "cycle_true")}

"°°°°°°°imported parameters°°°°°°°"
prompt_file_name = 'PromptNamedGame.template'
prompt_file_name = os.path.join('resources', 'initial_prompts', prompt_file_name)
current_directory = os.getcwd().replace("\instance_generator", "")
with open(os.path.join(current_directory, "games", "textmapworld_description", 'resources', 'multimodal_instances.json')) as json_file:
    multimodal_instances = json.load(json_file)
with open(os.path.join(current_directory, "games", "textmapworld_description", 'resources', 'initial_prompts', "answers.json")) as json_file:
    answers_file = json.load(json_file)
with open(os.path.join(current_directory, "games", "textmapworld", 'resources', 'initial_prompts', "reminders.json")) as json_file:
    reminders_file = json.load(json_file)
with open(os.path.join(current_directory, "games", "textmapworld_description", 'resources', 'captions.json')) as json_file:
    captions = json.load(json_file)
image_path = current_directory + "/games/textmapworld_description/resources/images/"
"-------------------------------------------------------------------------------------------------------------"

class GraphGameInstanceGenerator(GameInstanceGenerator):

    def __init__(self,  ):
        super().__init__(game_name)

    def on_generate(self):
        player_a_prompt_header =  self.load_template(prompt_file_name)
        Player2_positive_answer = answers_file["PositiveAnswerNamedGame"] 
        Player2_negative_answer = answers_file["NegativeAnswerNamedGame"]
        experiment_names = [s['name'] for s in multimodal_instances['experiments']]
        for name in experiment_names:
            experiment = self.add_experiment(name)
            size, cycle= experiments_details[name]
            data = next((e for e in multimodal_instances['experiments'] if e['name'] == name), None)
            for instance in data['game_instances']:
                game_id = instance['game_id']
                game_instance = self.add_game_instance(experiment, game_id)
                nodes = [ast.literal_eval(t) for t in instance['nodes']]
                edges= [ast.literal_eval(t) for t in  instance['edges']]
                changed_graph = generate_graph_info(nodes, edges, instance['cats'], image_path)
                descriptions = generate_descriptions(instance['imgs'], instance['cats'], captions)
                game_instance["Prompt"] = player_a_prompt_header
                game_instance["Player2_positive_answer"] = Player2_positive_answer
                game_instance["Player2_negative_answer"] = Player2_negative_answer
                game_instance["Move_Construction"] = MOVE_REGEX
                game_instance["Stop_Construction"] = DONE_REGEX
                game_instance["Grid_Dimension"] = "4"
                game_instance['Graph_Nodes'] = str(changed_graph['Graph_Nodes'])
                game_instance['Graph_Edges'] = str(changed_graph['Graph_Edges'])
                game_instance['Current_Position'] = instance['cats'][instance["start"]]
                game_instance['Picture_Name'] = changed_graph['Picture']
                game_instance["Directions"] = str(changed_graph['Directions'])
                game_instance["Moves"] = str(changed_graph['Moves'])
                game_instance['Cycle'] = cycle
                game_instance['Ambiguity'] = None 
                game_instance['Game_Type'] = "named_graph"
                game_instance["Descriptions"] = descriptions
                game_instance["Loop_Reminder"] = loop_reminder
                game_instance["Loop_Reminder_Text"] = reminders_file["loop_reminder"]
                game_instance["Max_Turns_Reminder"] = max_turns_reminder
                game_instance["Max_Turns_Reminder_Text"] = reminders_file["max_turns_reminder"]
                game_instance["Mapping"]= str(instance['cats'])
                game_instance["Strict"] = strict



if __name__ == '__main__':
    # always call this, which will actually generate and save the JSON file
    GraphGameInstanceGenerator().generate()