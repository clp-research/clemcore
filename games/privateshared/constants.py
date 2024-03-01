"""
Constants used in the privateshared game and instance generator.

To add a new experiment, append its config to EXPERIMENTS, what_slot and tags.
"""

import numpy as np

GAME_NAME = 'privateshared'
EXPERIMENTS = ['travel-booking', 'job-interview', 'restaurant',
               'things-places', 'letter-number']

# paths to game resources
PROBES_PATH = 'resources/texts/{}/probing_questions.json'
RETRIES_PATH = 'resources/texts/reprompts.json'
REQUESTS_PATH = 'resources/texts/{}/requests.json'
SLOT_PATH = 'resources/texts/{}/slot_values.json'
PROMPT_PATH = 'resources/initial_prompts/{}_{}'
WORDS_PATH = 'resources/{}_words.json'

# labels
INVALID = 'NA'
INVALID_LABEL = 2

# standard messages
UPDATE = 'Value for {} anticipated; ground truth turn updated from {} to {}.'
NOT_SUCCESS = 'Answer for {} invalid after max attempts.'
SUCCESS = 'Answer for {} valid after {} tries.'
RESULT = 'Answer is {}correct.'
NOT_PARSED = 'Answer could not be parsed!'
