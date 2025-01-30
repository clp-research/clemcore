"""
Generate instances for the memory game.

Creates files in ./in
"""
import random
import sys
import os

from tqdm import tqdm

import clemgame
from clemgame.clemgame import GameInstanceGenerator

N = os.getenv('NINSTANCES')
if N is not None:   
    N_INSTANCES = int(N)
else:
    N_INSTANCES = 5  

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
GAME_NAME = "askmissing"

class AskMissingInstaceGenerator(GameInstanceGenerator):


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

    def on_generate(self):

        initial_prompt = self.load_template("resources/initial_prompts/prompt")
        self.data = self.read_files()
        experiment = self.add_experiment(f"contact_memory_{LANGUAGE}")
        experiment["language"] = LANGUAGE  # experiment parameters

        experiment['initial_prompt'] = initial_prompt

        for game_id in tqdm(range(N_INSTANCES)):

            prompt = ''
            skip = random.choice(['first', 'last', 'email', 'work', 'attr'])
            firstname = self.getrandom('firstnames')
            lastname = self.getrandom('lastnames')
            email = self.generate_email(firstname.lower(), lastname.lower())
            prompt += '\n\n'
            if skip != 'first':
                prompt += f'First Name: {firstname}\n'
            if skip != 'last':
                prompt += f'Last Name: {lastname}\n'
            if skip != 'email':
                prompt += f'Email: {email}\n'
            if skip != 'work':
                work = self.getrandomremove('companies')
                prompt += f'Work: {work}\n'

            if skip != 'attr':
                random_attr = random.choice(['hobby', 'clothing', 'physical'])
                if random_attr == 'hobby':
                    hobby = self.getrandomremove('hobbies')
                    prompt += f'Hobby: {hobby}\n'
                if random_attr == 'clothing':
                    clothing = self.getrandomremove('clothing')
                    prompt += f'Clothing: {clothing}\n'
                if random_attr == 'physical':
                    trait = self.getrandomremove('traits')
                    prompt += f'Physical Traits: {trait}\n'
                
            game_instance = self.add_game_instance(experiment, game_id)
            game_instance['info'] = prompt
            game_instance['skip'] = skip

if __name__ == '__main__':
    AskMissingInstaceGenerator().generate()