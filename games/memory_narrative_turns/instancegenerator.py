import random
import sys
import os

from tqdm import tqdm

import clemgame
from clemgame.clemgame import GameInstanceGenerator

N_INSTANCES = int(os.getenv('NINSTANCES')) or 5  

attrs = ['firstnames',
         'lastnames',
         'companies',
         'emails',
         'hobbies',
         'traits',
         'clothing'
         ]

LANGUAGE = "en"

logger = clemgame.get_logger(__name__)
GAME_NAME = "memory_narrative_turns"

class MemoryNarrativeTurnsGameInstaceGenerator(GameInstanceGenerator):


    def __init__(self):
        super().__init__(GAME_NAME)

    def load_instances(self):
        return self.load_json("in/instances")
    
    def read_files(self):
        data = {}
        for attr in attrs:
            data[attr] = self.load_file(f"resources/data/{attr}", file_ending=".txt").split("\n")
        return data
    
    def getrandomremove(self, attr):
        '''
        ensure that a randomly chosen item is not chosen again for uniquness to an individual
        '''
        l = self.data[attr]
        random.shuffle(l)
        return l.pop()
    
    def getrandom(self, attr):
        '''
        randomly pick an item and return it, do not remove from list to allow duplicates
        '''
        return random.choice(self.data[attr])
    
    def coin_toss(self):
        return random.choice([0,1])==1

    def generate_email(self, fistname, lastname):
        domain = self.getrandom('emails')
        if self.coin_toss():
            if self.coin_toss():
                return f'{fistname}.{lastname}@{domain}'
            else:
                return f'{lastname}.{fistname}@{domain}'

        return f'{fistname}{lastname}@{domain}'
    
    def generate_question(self, attr, seed, value):
        if seed == 'clothing':
            a = "" if value[-1] == 's' else "a "
            return f"What is the {attr} of the person who wears {a}{value.lower()}?"
        if seed == 'hobby':
            return f"What is the {attr} of the person who likes {value.lower()}?"
        if seed == 'trait':
            return f"What is the {attr} of the person who has {value.lower()}?"
        if seed == 'work':
            return f"What is the {attr} of the person works for {value}?"

    def on_generate(self):

        initial_prompt = self.load_template("resources/initial_prompts/prompt")
        self.data = self.read_files()
        experiment = self.add_experiment(f"contact_memory_{LANGUAGE}")
        experiment["language"] = LANGUAGE  # experiment parameters
        experiment['initial_prompt'] = initial_prompt

        questions = []
        answers = []
        prompts = []
        
        for game_id in tqdm(range(N_INSTANCES)):
            prompt = ''
            if game_id == 0:
                prompt += 'Here is some information about a person that I need you to remember:'
            else:
                prompt += '\nHere is some information about another person that I need you to remember:'
            firstname = self.getrandom('firstnames')
            lastname = self.getrandom('lastnames')
            email = self.generate_email(firstname.lower(), lastname.lower())
            prompt += '\n\n'
            prompt += f'name is {firstname} {lastname}, email is {email},'
            work = self.getrandomremove('companies')
            prompt += f' works for {work}'
            attr_map = {'first name': firstname, 
                        'last name':lastname
                        # 'full name': f'{firstname} {lastname}',
                        # 'email': email
                        }
            target_seed_options = {'work': work}
            # always include the hobby as a fallback
            hobby = self.getrandomremove('hobbies')
            target_seed_options['hobby'] = hobby
            prompt += f', likes {hobby}'
            # also include clothing and trait at random
            if self.coin_toss(): 
                clothing = self.getrandomremove('clothing')
                target_seed_options['clothing'] = clothing
                prompt += f', wears {clothing}'
            if self.coin_toss(): 
                trait = self.getrandomremove('traits')
                target_seed_options['trait'] = trait
                prompt += f', has {trait}'
            prompt += '.\n'
            
            target_attr = random.choice(list(attr_map.keys()))
            target_seed = random.choice(list(target_seed_options.keys()))
            question = self.generate_question(target_attr, target_seed, target_seed_options[target_seed])
            prompts.append(prompt)
            questions.append(question)
            answers.append(attr_map[target_attr])


        # experiment['initial_prompt'] = initial_prompt.replace('$PEOPLE$', prompt)
        game_ids = list(range(N_INSTANCES))
        for game_id in tqdm(range(N_INSTANCES)):
            game_instance = self.add_game_instance(experiment, game_id)
            game_instance['prompt'] = prompts[game_id]
            # make sure it only asks questions about people that have already been inputted
            ask_id = random.choice(game_ids)
            while ask_id > game_id:
                ask_id = random.choice(game_ids)
            game_instance['question'] = questions[ask_id]
            game_instance['answer'] = answers[ask_id]
            game_instance['qa_game_id'] = ask_id

if __name__ == '__main__':
    MemoryNarrativeTurnsGameInstaceGenerator().generate()
