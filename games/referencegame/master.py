from typing import List, Dict

import numpy as np

from backends import Model
from clemgame import file_utils
from clemgame import metrics
from clemgame.clemgame import GameMaster, GameBenchmark, GameScorer
from clemgame import get_logger
from games.referencegame.game import ReferenceGame
from games.referencegame.resources.localization_utils import MULTILINGUAL_PATTERNS
import re

GAME_NAME = "referencegame"
logger = get_logger(__name__)


class ReferenceGameMaster(GameMaster):

    def __init__(self, experiment: Dict, player_models: List[Model]):
        super().__init__(GAME_NAME, experiment, player_models)
        self.experiment = experiment
        self.game = None
        self.game_instance = None
        self.player_1_response_pattern = ""
        self.player_2_response_pattern = ""
        #TODO make mode a command line parameter
        self.mode = "strict" # "liberal" #


    def setup(self, **game_instance):
        self.game_instance = game_instance

        self.game = ReferenceGame(self.game_instance, self.player_models)

        self.log_players({
            "GM": "Game master for referencegame",
            "Player_1": self.player_models[0].get_name(),
            "Player_2": self.player_models[1].get_name()}
        )

        self.player_1_response_pattern = self.generate_regex("p1")
        self.player_2_response_pattern = self.generate_regex("p2")

    def generate_regex(self, player: str):
        """
        Combine language specific content with regex pattern
        The regex uses 4 named groups: tag, response, content and remainder
        - model is instructed to start its answer with "tag"
        - the following \s is optional (as asian languages don't use spaces)
        - "response" combines "content" and "remainder" (in case everything should be passed on to the next player)
        - "content" is defined a the first produced paragraph (up to a newline)
        - "remainder" collects all following content
        Strict parsing mode:
        - Model response has to start with "tag" (not optional)
        - "remainder" should be "" (has to be implemented in response checking)
        Liberal parsing mode:
        - "tag" doesn't have to be at the beginning but only somewhere in the reply
        - "remainder" doesn't have to be empty (has to be implemented in response checking)

        :param player: string identifier of player ("p1" or "p2")
        :return: liberal regex pattern for given player in the current language
        """
        tag = content = ""
        if player == "p1":
            tag = MULTILINGUAL_PATTERNS[self.game.lang]["p1_tag"]
            content = ".+"
        elif player == "p2":
            tag = MULTILINGUAL_PATTERNS[self.game.lang]["p2_tag"]
            # collect answer options and also allow models to output numbers, as we don't do prompt engineering in all languages
            content = MULTILINGUAL_PATTERNS[self.game.lang]["p2_options"] + "|1|2|3"
        if self.mode == "strict":
            return f'^(?P<tag>{tag})\s*(?P<response>(?P<content>{content})\n*(?P<remainder>(.|\n)*))'
        else:
            return f'(?P<tag>{tag})\s*(?P<response>(?P<content>{content})\n*(?P<remainder>(.|\n)*))'


    @classmethod
    def applies_to(cls, game_name: str) -> bool:
        return game_name == GAME_NAME

    def play(self) -> None:
        logger.info("Game turn: %d", self.game.turn_count)
        self.turn()

    def turn(self):

        self.log_next_turn()

        # generate referring expression - Player 1 side

        self.game.given_instruction.add_user_message(self.game.player_1_prompt_header)

        # log the game master to player 1
        action = {'type': 'send message', 'content': self.game.given_instruction.user_messages[-1]}
        self.log_event(from_="GM", to="Player 1", action=action)

        player_1_prompt, player_1_response, player_1_response_text = self.game.instruction_giver(
            self.game.given_instruction, None)

        # log the retrieved utterance
        action = {'type': 'get message', 'content': player_1_response_text}
        self.log_event(from_="Player 1", to="GM", action=action, call=(player_1_prompt, player_1_response))

        self.game.given_instruction.add_system_message(player_1_response_text)

        player_1_pattern = re.compile(self.player_1_response_pattern, re.IGNORECASE)
        p1_match = re.match(player_1_pattern, player_1_response_text)
        match = False
        if p1_match:
            if self.mode == "liberal":
                # we don't care how much the model generates
                action = {'type': 'parse', 'content': player_1_response_text,
                          'expression': p1_match.group('response')}
                self.log_event(from_="GM", to="GM", action=action)
                match = True
            elif self.mode == "strict":
                # the model should not produce more than one paragraph
                if p1_match.group('remainder') == "":
                    # we only pass on the expression
                    action = {'type': 'parse', 'content': player_1_response_text,
                              'expression': p1_match.group('content')}
                    self.log_event(from_="GM", to="GM", action=action)
                    match = True

        if not match:
            # if the Player 1 message doesn't match the rule in the given mode
            # log the message and abort the game
            action = {'type': 'invalid format', 'content': 'Invalid generated expression',
                      'original_content': player_1_response_text}
            self.log_event(from_="GM", to="GM", action=action)

            return

        # guess the grid - Player 2 side

        self.game.followed_instruction.add_user_message(
            self.game.player_2_prompt_header.replace('TARGET_EXPRESSION', player_1_response_text))

        # log the game master to player 2
        action = {'type': 'send message', 'content': self.game.followed_instruction.user_messages[-1]}
        self.log_event(from_="GM", to="Player 2", action=action)

        player_2_prompt, player_2_response, player_2_response_text = self.game.instruction_follower(
            self.game.followed_instruction, None)

        self.game.followed_instruction.add_system_message(player_2_response_text)

        self.game.turn_count += 1

        # log the retrieved utterance
        action = {'type': 'get message', 'content': player_2_response_text}
        self.log_event(from_="Player 2", to="GM", action=action, call=(player_2_prompt, player_2_response))

        # check if the Player 2 message matches the rule => start with "Answer: " and generate only the label
        player_2_pattern = re.compile(self.player_2_response_pattern, re.IGNORECASE)
        p2_match = re.match(player_2_pattern, player_2_response_text)
        match = False
        # set liberal parsinf mode (i.e., ignore the rest if the model produced more than the answer option
        # (like punctuation or "grid" in the multilingual version)
        self.mode == "liberal"
        if p2_match:
            if self.mode == "liberal":
                # we don't care how much more the model generates,
                # but we collect only the given option
                action = {'type': 'parse', 'content': player_2_response_text,
                          'answer': p2_match.group('content')}
                self.log_event(from_="GM", to="GM", action=action)
                match = True
            elif self.mode == "strict":
                # the model should not produce more than the answer
                if p2_match.group('remainder') == "":
                    action = {'type': 'parse', 'content': player_2_response_text,
                              'answer': p2_match.group('content')}
                    self.log_event(from_="GM", to="GM", action=action)
                    match = True

        if not match:
            # abort the game if the output doesn't match the rule
            action = {'type': 'invalid format', 'content': 'Invalid generated choice',
                      'original_content': player_2_response_text}
            self.log_event(from_="GM", to="GM", action=action)


class ReferenceGameScorer(GameScorer):

    def __init__(self, experiment: Dict, game_instance: Dict):
        super().__init__(GAME_NAME, experiment, game_instance)
        self.target_grid_name = self.extend_target(game_instance["target_grid_name"], game_instance["lang"])

    def extend_target(self, target, lang):
        """
        Extend the target to contain numerals as well as strings (if not already done by the instancegenerator)
        :param target: the string representation of the target position or a list of options
        :return: extended list of text and numeral version of target
        """
        if isinstance(target, list):
            # the extension was already done in the instancegenerator
            return target
        options = MULTILINGUAL_PATTERNS[lang]["p2_options"].split("|")
        assert target in options
        target_index = options.index(target)
        return [target, str(target_index + 1)]

    def compute_scores(self, episode_interactions: Dict) -> None:
        '''
        Compute and log scores for one episode of referencegame.
        :param episode_interactions: the game episode interactions log
        '''

        # For referencegame, there is just one turn (one exchange of p1-p2 is logged as one turn)
        turn = episode_interactions["turns"][0]
        turn_index = 0

        aborted = False

        turn_request_count = 0
        turn_parsed_request_count = 0
        episode_request_count = 0
        episode_parsed_request_count = 0

        success = 0

        # evaluate Player 1
        turn_request_count += 1
        episode_request_count += 1
        # check if the Player 1 message followed the rule
        # (true if third interaction (GM to GM) has type "parse")
        if turn[2]['action']['type'] == "parse":
            turn_parsed_request_count += 1
            episode_parsed_request_count += 1

            # log the Player 1 - message length
            p1_expression = turn[2]['action']['expression']
            expression_length = len(p1_expression)
            self.log_turn_score(turn_index, 'Generated Expression Length', expression_length)
            # as there is just one turn, this is the same as episode scores
            self.log_episode_score('Generated Expression Length', expression_length)

            # log the Player 1 - number of tokens in the generated expression
            number_of_tokens = len(p1_expression.split(' '))
            self.log_turn_score(turn_index, 'Generated Expression Number of Tokens', number_of_tokens)
            # as there is just one turn, this is the same as episode scores
            self.log_episode_score('Generated Expression Number of Tokens', number_of_tokens)

            # evaluate Player 2 (only if Player 1's response was valid)
            turn_request_count += 1
            episode_request_count += 1
            # check if the Player 2 message matched the rule
            # (true if sixth interaction (GM to GM) has type "parse")

            # HOTFIX to allow for backward compatible, more liberal player 2 parsing for existing game runs
            # (in the future already handled by ReferenceGameMaster above)
            p2_match = False
            if turn[5]['action']['type'] == "invalid format":
                # parse again with additionally allowing numbers as answers (and ignore anything that follows (like "grid" or punctuation))
                tag = MULTILINGUAL_PATTERNS[self.lang]["p2_tag"]
                content = MULTILINGUAL_PATTERNS[self.lang]["p2_options"] + "|1|2|3"
                p2_pattern_extended = f'^(?P<tag>{tag})\s*(?P<content>{content})'
                player_2_pattern = re.compile(p2_pattern_extended, re.IGNORECASE)
                p2_match = re.match(player_2_pattern, turn[5]['action']['original_content'])

            if turn[5]['action']['type'] == "parse" or p2_match:
                turn_parsed_request_count += 1
                episode_parsed_request_count += 1

                # check if the target grid number matches the output from Player 2
                player_2_answer = ""
                if p2_match:
                    player_2_answer = p2_match.group('content')
                elif turn[5]['action']['type'] == "parse":
                    player_2_answer = turn[5]['action']['answer']

                if player_2_answer.lower() in self.target_grid_name:
                    success = 1

                self.log_episode_score('Aborted at Player 1', 0)
                self.log_episode_score('Aborted at Player 2', 0)
            else:
                self.log_episode_score('Aborted at Player 1', 0)
                self.log_episode_score('Aborted at Player 2', 1)
                aborted = True
        else:
            self.log_episode_score('Aborted at Player 1', 1)
            self.log_episode_score('Aborted at Player 2', 0)
            self.log_turn_score(turn_index, 'Generated Expression Length', np.nan)
            self.log_episode_score('Generated Expression Length', np.nan)
            self.log_turn_score(turn_index, 'Generated Expression Number of Tokens', np.nan)
            self.log_episode_score('Generated Expression Number of Tokens', np.nan)
            aborted = True

        # log the turn request count, parsed & violated request counts
        self.log_turn_score(turn_index, metrics.METRIC_REQUEST_COUNT, turn_request_count)
        self.log_turn_score(turn_index, metrics.METRIC_REQUEST_COUNT_PARSED, turn_parsed_request_count)
        self.log_turn_score(turn_index, metrics.METRIC_REQUEST_COUNT_VIOLATED,
                            turn_request_count - turn_parsed_request_count)
        self.log_turn_score(turn_index, metrics.METRIC_SUCCESS, success)

        # log the episode request count, parsed & violated request counts
        self.log_episode_score(metrics.METRIC_REQUEST_COUNT, episode_request_count)
        self.log_episode_score(metrics.METRIC_REQUEST_COUNT_PARSED, episode_parsed_request_count)
        self.log_episode_score(metrics.METRIC_REQUEST_COUNT_VIOLATED,
                               episode_request_count - episode_parsed_request_count)

        self.log_episode_score(metrics.METRIC_ABORTED, int(aborted))
        success = success if not aborted else np.nan
        self.log_episode_score(metrics.METRIC_SUCCESS, success)
        loose = 1 - success if not aborted else np.nan
        self.log_episode_score(metrics.METRIC_LOSE, loose)

        bench_score = success * 100 if not aborted else np.nan
        self.log_episode_score(metrics.BENCH_SCORE, bench_score)

        request_success_ratio = round(episode_parsed_request_count / float(episode_request_count), 4)
        self.log_episode_score(metrics.METRIC_REQUEST_SUCCESS, request_success_ratio)


class ReferenceGameBenchmark(GameBenchmark):

    def __init__(self):
        super().__init__(GAME_NAME)

    def get_description(self):
        return "Reference Game between two agents " \
               "where one has to describe one of three grids " \
               "and the other has to guess which one it is."

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return ReferenceGameMaster(experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return ReferenceGameScorer(experiment, game_instance)


def main():
    # select one instance
    experiments = file_utils.load_json("in/instances_v1.5_en.json", "referencegame")
    instance = experiments["experiments"][0]["game_instances"][0]
    master = ReferenceGameMaster(instance, ["fsc-openchat-3.5-0106", "fsc-openchat-3.5-0106"])
    master.setup(**instance)
    master.play()


if __name__ == '__main__':
    main()