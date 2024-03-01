"""
Randomly generate templates for the private/shared game in English.

Creates files in ./instances and ./requests
"""
import random
from typing import Tuple, Dict, List

from tqdm import tqdm

import clemgame
from clemgame.clemgame import GameInstanceGenerator
from games.privateshared.constants import (
    PROBES_PATH, REQUESTS_PATH, SLOT_PATH, PROMPT_PATH, WORDS_PATH,
    GAME_NAME, EXPERIMENTS)

ID = 1
LANG = 'en'
SEED = 2102
N_INSTANCES = 10

what_slot = {'travel-booking': 'Travel',
             'job-interview': 'Job Application',
             'restaurant': 'Restaurant',
             'things-places': 'Things at places',
             'letter-number': 'Numbered letters'}

logger = clemgame.get_logger(__name__)


def sample_instance(slot_values: dict, what_value: str) -> Tuple[dict, str]:
    """Create an instance with randomly chosen values for each slot."""
    instance_str = f'WHAT: {what_value}\n'
    instance_dic = {}
    for key, values in slot_values.items():
        rand_v = random.choice(values)
        instance_dic[key] = rand_v
        instance_str += f'{key.upper()}: {rand_v}\n'
    return instance_dic, instance_str.strip('\n')


def sample_request_order(request_strings: dict) -> List[str]:
    """Create an order for the values' requests."""
    n_slots = len(request_strings)
    return random.sample(list(request_strings.keys()), n_slots)


def sample_request_texts(request_strings: dict) -> Dict[str, int]:
    """Select types for each request."""
    requests = {}
    for key, question_types in request_strings.items():
        n = len(question_types)
        r_index = random.choice(range(n))
        requests[key] = r_index
    return requests


def sample_probes(probing_questions: dict, n_slots: int) -> Dict[int, dict]:
    """Select types for each probing question."""
    probes = {}
    for turn in range(n_slots + 1):
        probes[turn] = {}
        for key, question_types in probing_questions.items():
            n = len(question_types)
            r_index = random.choice(range(n))
            probes[turn][key] = r_index
    return probes


class PrivateSharedGameInstanceGenerator(GameInstanceGenerator):
    """Generator of instances for all experiments."""

    def __init__(self):
        super().__init__(GAME_NAME)
        words = self.load_json(WORDS_PATH.format(LANG))
        self.tags = words['tags']
        self.answer = words["ANSWER"]
        self.aside = words["ASIDE"]
        self.me = words["ME"]

    def on_generate(self):
        """Generate configuration of all experiments."""

        for exp_name in EXPERIMENTS:
            # load all necessary contents
            probes = self.load_json(PROBES_PATH.format(exp_name))
            requests = self.load_json(REQUESTS_PATH.format(exp_name))
            slot_values = self.load_json(SLOT_PATH.format(exp_name))
            experiment = self.add_experiment(exp_name)
            prompt = self.load_template(PROMPT_PATH.format(exp_name, ID))

            for game_id in tqdm(range(N_INSTANCES)):
                game_instance = self.add_game_instance(experiment, game_id)
                what_value = what_slot[exp_name]
                inst_dic, inst_str = sample_instance(slot_values, what_value)
                n_slots = len(inst_dic)
                # create the instance
                game_instance['tag'] = self.tags[exp_name]
                game_instance['slots'] = inst_dic
                game_instance['initial_prompt'] = self.create_prompt(
                    prompt, inst_str, self.tags[exp_name])
                game_instance['request_order'] = sample_request_order(requests)
                game_instance['requests'] = sample_request_texts(requests)
                game_instance['probes'] = sample_probes(probes, n_slots)
                game_instance['lang'] = LANG

    def create_prompt(self, prompt: str, instance: str, tag: str) -> str:
        """Fill in the initial prompt variables."""
        text = prompt.replace('$INSTANCE$', instance)
        text = text.replace('$QUESTIONER$', tag)
        text = text.replace('$ANSWER$', self.answer)
        text = text.replace('$ASIDE$', self.aside)
        text = text.replace('$ME$', self.me)
        return text


if __name__ == '__main__':
    random.seed(SEED)
    PrivateSharedGameInstanceGenerator().generate()
