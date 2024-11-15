from typing import List, Tuple, Dict

from backends import Model
from clemgame import metrics
from clemgame.clemgame import GameMaster, GameBenchmark, GameScorer
from games.imagegame.game import ImageGame
from games.imagegame.evaluator import evaluate, calculate_flipped_pixels
from clemgame import get_logger
from games.imagegame.resources.localization_utils import MULTILINGUAL_PATTERNS
import re
import math

GAME_NAME = "imagegame"

logger = get_logger(__name__)


class ImageGameMaster(GameMaster):

    def __init__(self, experiment: Dict, player_models: List[Model]):
        super().__init__(GAME_NAME, experiment, player_models)
        self.experiment = experiment
        self.game = None
        self.request_count = 0
        self.parsed_request_count = 0
        self.violated_request_count = 0
        self.aborted_ratio = 0
        self.player_1_response_pattern = ""
        self.player_1_terminate_pattern = ""
        self.player_2_response_pattern = ""
        # TODO make mode a command line parameter
        self.mode = "strict"  # "liberal"

        self.turn_request_stats = {}

    def get_description(self) -> str:
        return "Image Game simulation"

    def _on_setup(self, **game_instance):
        self.game_instance = game_instance

        self.game = ImageGame(self.game_instance, self.player_models)

        self.log_players({
            "GM": "Game master for imagegame",
            "Player_1": self.player_models[0].get_name(),
            "Player_2": self.player_models[1].get_name()}
        )

        self.player_1_response_pattern = self.generate_regex("p1_response")
        self.player_1_terminate_pattern = self.generate_regex("p1_terminate")
        self.player_2_response_pattern = self.generate_regex("p2_response")

    def setup(self, **kwargs):
        self._on_setup(**kwargs)

    def generate_regex(self, kind: str):
        """
        Combine language specific content with regex pattern
        The player 1 regex uses 5 named groups: head, body, tag, instruction and tail
        - model is instructed to start answer with "tag"
        - "head" is everything before the first occurrence of "tag"
        - "body" combines "tag" and "instruction"
        - "instruction" is everything after the first occurrence of "tag" up to a newline character
        - "tail" collects all following content
        - "head" and "tail" might consist of only whitespaces
        Strict parsing mode:
        - Model response has to start with "tag" (not optional)
        - "head" and "tail" should be ""
        Liberal parsing mode:
        - "tag" doesn't have to be at the beginning but only somewhere in the reply
        - "head" and "tail" don't have to be empty
        - In case the model generates multiple instructions in one answer
          the first instruction is extracted and others are ignored.

        The player 2 regex matches with a 5 by 5 grid.
        It has 3 named groups: head, grid and tail.
        "head": is everything before the grid.
        "grid": the grid. Must start at line beginning.
        "tail": is everything after the grid.
        Strict parsing mode:
        - "head" and "tail" can consist of newline characters
        Liberal parsing mode:
        - "head": if not empty:
            - the line above the grid must be empty or whitespaces
            - above the empty line, everything is allowed
            - must end with newline character, so grid starts at line beginning
        - "tail": if not empty:
            - whitespaces are allowed after the grid
            - the line under the grid must be empty or whitespaces
            - after the empty line, everything is allowed

        :param kind: string identifier for the type of regex ("p1_response", "p1_terminate", "p2_response")
        :return: regex pattern of given kind in the current language and the current parsing mode
        """
        tag = MULTILINGUAL_PATTERNS[self.game.lang]["tag"]

        assert self.mode in {"strict", "liberal"}, f"The mode '{self.mode}' is unknown"
        assert kind in {"p1_response", "p1_terminate", "p2_response"}, f"'{kind}' is invalid value for param kind"

        if kind == "p2_response":
            grid = "([A-Z▢]\s){4}[A-Z▢]\n([A-Z▢]\s){4}[A-Z▢]\n([A-Z▢]\s){4}[A-Z▢]\n([A-Z▢]\s){4}[A-Z▢]\n([A-Z▢]\s){4}[A-Z▢]"
            if self.mode == "liberal":
                return f"^(?P<head>([\S\s]*\n\s*\n)?(\s*\n+)?)(?P<grid>{grid})(?P<tail>\s*(\n\s*\n[\S\s]*)?)$"
            elif self.mode == "strict":
                return f"^(?P<head>\n*)(?P<grid>{grid})(?P<tail>\n*)$"

        elif kind == "p1_response":
            instruction = ".+"
            if self.mode == "liberal":
                return f"^(?P<head>[\S\s]*?)(?P<body>(?P<tag>{tag})\s*(?P<instruction>{instruction}))(?P<tail>[\S\s]*)$"
            elif self.mode == "strict":
                return f"^(?P<head>)(?P<body>(?P<tag>{tag})[ \t]*(?P<instruction>{instruction}))(?P<tail>)$"

        elif kind == "p1_terminate":
            instruction = MULTILINGUAL_PATTERNS[self.game.lang]["terminate_token"]
            if self.mode == "liberal":
                return f"^(?P<head>((?!{tag})[\S\s])*)(?P<body>(?P<tag>{tag})\s*(?P<instruction>{instruction}))(?P<tail>[\S\s]*)$"
            elif self.mode == "strict":
                return f"^(?P<head>\s*)(?P<body>(?P<tag>{tag})\s*(?P<instruction>{instruction}))(?P<tail>\s*)$"

    @classmethod
    def applies_to(cls, game_name: str) -> bool:
        return game_name == GAME_NAME

    def play(self) -> None:
        while self.game.proceeds():
            logger.info("Game turn: %d", self.game.current_turn)
            self.turn()

    def turn(self):
        # instruction giving - A side
        self.log_next_turn()
        self.turn_request_stats[self.game.current_turn] = {'request_count': 0, 'parsed_count': 0, 'violated_count': 0}

        if self.game.next_turn_message != '':
            self.game.given_instruction.add_user_message(self.game.next_turn_message)
        self.game.next_turn_message = self.game.player_1_question

        # log the game master to player 1
        action = {'type': 'send message', 'content': self.game.given_instruction.user_messages[-1]}
        self.log_event(from_="GM", to="Player 1", action=action)

        player_1_prompt, player_1_response, player_1_response_text = self.game.instruction_giver(self.game.given_instruction,
                                                                                            self.game.current_turn)

        self.request_count += 1
        self.turn_request_stats[self.game.current_turn]['request_count'] += 1

        # log the retrieved utterance
        action = {'type': 'get message', 'content': player_1_response_text}
        self.log_event(from_="Player 1", to="GM", action=action, call=(player_1_prompt, player_1_response))

        # add the message to Player 1
        self.game.given_instruction.add_system_message(player_1_response_text)

        # check if it reached the end on 1 side
        # Note: In case the model generates the terminate pattern as first instruction, the game is ended (not aborted).
        #   TODO: game should be aborted in this case because player 2 didn't have a chance to generate grid?
        match = re.compile(self.player_1_terminate_pattern, re.IGNORECASE).match(player_1_response_text)
        if match:
            # Player 1 message matched with the terminate pattern in the given mode
            self.parsed_request_count += 1
            self.turn_request_stats[self.game.current_turn]['parsed_count'] += 1
            self.game.terminate = True
            return

        # continue if the Player didn't say -> Instruction: DONE
        # check if Player 1 message follows the rule => start with "Instruction:"
        match = re.compile(self.player_1_response_pattern, re.IGNORECASE).match(player_1_response_text)
        if match:
            self.parsed_request_count += 1
            self.turn_request_stats[self.game.current_turn]['parsed_count'] += 1
            action = {'type': 'parse', 'content': match.group('body').strip(),
                      'instruction': match.group('instruction').strip(),
                      'original_content': player_1_response_text}
            self.log_event(from_="GM", to="GM", action=action)
        else:
            # log invalid format for the Player 1
            action = {'type': 'invalid format', 'content': 'Invalid instruction format',
                      'original_content': player_1_response_text}
            self.log_event(from_="GM", to="GM", action=action)
            self.turn_request_stats[self.game.current_turn]['violated_count'] += 1

            self.violated_request_count += 1
            self.aborted_ratio += 1
            # terminate the game play when the message doesn't follow the rule
            self.game.terminate = True
            return


        # instruction following - 2 side
        if self.game.current_turn == 0:
            self.game.followed_instruction.add_user_message(
                self.game.player_2_prompt_header + '\n' + match.group('body').strip())
        else:
            self.game.followed_instruction.add_user_message(match.group('body').strip())


        # log the game master to player 2
        action = {'type': 'send message', 'content': self.game.followed_instruction.user_messages[-1]}
        self.log_event(from_="GM", to="Player 2", action=action)

        player_2_prompt, player_2_response, player_2_response_text = self.game.instruction_follower(
            self.game.followed_instruction, self.game.current_turn)

        # log the retrieved utterance
        action = {'type': 'get message', 'content': player_2_response_text}
        self.log_event(from_="Player 2", to="GM", action=action, call=(player_2_prompt, player_2_response))
        self.game.followed_instruction.add_system_message(player_2_response_text)

        # increase the request count
        self.turn_request_stats[self.game.current_turn]['request_count'] += 1
        self.request_count += 1

        # check if Player 2 message has the required format: grid
        match = re.compile(self.player_2_response_pattern).match(player_2_response_text)
        if match:
            self.parsed_request_count += 1
            self.turn_request_stats[self.game.current_turn]['parsed_count'] += 1

            action = {'type': 'parse', 'content': match.group('grid'),
                      'original_content': player_2_response_text}
            self.log_event(from_="GM", to="GM", action=action)

        else:
            # log the invalid format
            action = {'type': 'invalid format', 'content': 'Invalid grid format',
                      'original_content': player_2_response_text}
            self.log_event(from_="GM", to="GM", action=action)
            self.turn_request_stats[self.game.current_turn]['violated_count'] += 1
            self.violated_request_count += 1
            self.aborted_ratio += 1
            # terminate the game play when the message doesn't follow the rule
            self.game.terminate = True
            return

        self.game.current_turn += 1


class ImageGameScorer(GameScorer):

    def __init__(self, experiment: Dict, game_instance: Dict):
        super().__init__(GAME_NAME, experiment, game_instance)
        self.target_grid = game_instance["target_grid"]

    def compute_scores(self, episode_interactions: Dict) -> None:

        precision, recall, f1 = 0, 0, 0

        previous_turn_grid = '▢ ▢ ▢ ▢ ▢\n▢ ▢ ▢ ▢ ▢\n▢ ▢ ▢ ▢ ▢\n▢ ▢ ▢ ▢ ▢\n▢ ▢ ▢ ▢ ▢'
        flipped_count_sum = 0
        expression_length_sum = 0
        expression_number_of_tokens = 0
        current_turn_grid = ''

        episode_request_count = 0
        episode_parsed_request_count = 0
        episode_violated_request_count = 0

        aborted = False
        aborted_at = None
        number_of_turns = 0

        # loop over each turn and calculate the metrics for both Player 1 and 2.

        for t_index, turn in enumerate(episode_interactions["turns"]):

            turn_request_count = 0
            turn_parsed_request_count = 0
            turn_violated_request_count = 0

            # Player generates "DONE"
            # (true if turn has length 2)
            if len(turn) == 2:
                break

            turn_request_count += 1
            episode_request_count += 1

            # check the Player 1 message if it matches the rule
            # (true if third interaction (GM to GM) has type "parse")
            if turn[2]['action']['type'] == 'parse':
                turn_parsed_request_count += 1
                episode_parsed_request_count += 1
            else:
                turn_violated_request_count += 1
                episode_violated_request_count += 1
                aborted = True
                aborted_at = 1
                # do not continue processing the rest of the turn when the game is aborted
                break

            # Player 2's turn

            turn_request_count += 1
            episode_request_count += 1

            # check Player 2 message if it matches the instruction => grid
            # (true if sixth interaction (GM to GM) has type "parse")
            if turn[5]['action']['type'] == 'parse':
                turn_parsed_request_count += 1
                episode_parsed_request_count += 1
            else:
                turn_violated_request_count += 1
                episode_violated_request_count += 1
                aborted = True
                aborted_at = 2
                break

            # calculate player-specific and turn-specific metrics

            # Player 2 message
            player_2_message = turn[4]['action']['content']

            try:
                precision, recall, f1 = evaluate(self.target_grid, player_2_message)
            except:
                pass

            # number of turns other
            number_of_turns += 1

            # Player 1 message
            player_1_message = turn[2]['action']['instruction']

            # Player 1 - message length
            expression_length = len(player_1_message)
            self.log_turn_score(t_index, 'Generated Expression Length', expression_length)
            expression_length_sum += expression_length

            # Player 1 - number of tokens in the generated expression
            number_of_tokens = len(player_1_message.split(' '))
            self.log_turn_score(t_index, 'Generated Expression Number of Tokens', number_of_tokens)
            expression_number_of_tokens += number_of_tokens

            self.log_turn_score(t_index, 'Precision', precision)
            self.log_turn_score(t_index, 'Recall', recall)
            self.log_turn_score(t_index, 'F1', f1)

            # calculate flipped pixel counts
            flipped_count = 0
            try:
                current_turn_grid = player_2_message
                flipped_count = calculate_flipped_pixels(previous_turn_grid, current_turn_grid)
            except:
                pass

            flipped_count_sum += flipped_count
            previous_turn_grid = current_turn_grid
            self.log_turn_score(t_index, 'Changed Cell Count', flipped_count)

            # request count, parsed & violated request counts
            self.log_turn_score(t_index, metrics.METRIC_REQUEST_COUNT,
                                turn_request_count)
            self.log_turn_score(t_index, metrics.METRIC_REQUEST_COUNT_PARSED,
                                turn_parsed_request_count)
            self.log_turn_score(t_index, metrics.METRIC_REQUEST_COUNT_VIOLATED,
                                turn_violated_request_count)


        # quick fix for ZeroDevisionError when player 1 generated terminate token in first answer:
        # treat as if game was aborted
        # should be handled in GameMaster (see todo in self.turn())
        if number_of_turns == 0:
            aborted = True
            aborted_at = 1

        # Episode level logging
        if aborted:
            # if aborted give NaN value to all metrics
            self.log_episode_score('Precision', math.nan)
            self.log_episode_score('Recall', math.nan)
            self.log_episode_score('F1', math.nan)
            self.log_episode_score(metrics.BENCH_SCORE, math.nan)

            # average of flipped pixel counts
            self.log_episode_score('Average Changed Cell Count', math.nan)

            # average of expression length
            self.log_episode_score('Average Generated Instruction Length', math.nan)

            # average of number of tokens in generated expression
            self.log_episode_score('Average Generated Expression Number of Tokens', math.nan)

            # the last turn scores are also the scores for the episode
            self.log_episode_score(metrics.METRIC_SUCCESS, 0)

            # lose ratio
            self.log_episode_score(metrics.METRIC_LOSE, 0)

            # aborted ratio
            self.log_episode_score(metrics.METRIC_ABORTED, 1)
        else:
            # the last turn scores are also the scores for the episode
            self.log_episode_score('Precision', precision)
            self.log_episode_score('Recall', recall)
            self.log_episode_score('F1', f1)
            self.log_episode_score(metrics.BENCH_SCORE, f1)

            # average of flipped pixel counts
            flipped_count_sum = round(flipped_count_sum / float(number_of_turns), 4)
            self.log_episode_score('Average Changed Cell Count', flipped_count_sum)

            # average of expression length
            expression_length_sum = round(expression_length_sum / float(number_of_turns), 4)
            self.log_episode_score('Average Generated Instruction Length', expression_length_sum)

            # average of number of tokens in generated expression
            expression_number_of_tokens = round(expression_number_of_tokens / float(number_of_turns), 4)
            self.log_episode_score('Average Generated Expression Number of Tokens', expression_number_of_tokens)

            # the last turn scores are also the scores for the episode
            self.log_episode_score(metrics.METRIC_SUCCESS, 1 if f1 >= 99 else 0)

            # lose ratio
            self.log_episode_score(metrics.METRIC_LOSE, 0 if f1 >= 99 else 1)

            # aborted ratio
            self.log_episode_score(metrics.METRIC_ABORTED, 0)

        # aborted at player x
        self.log_episode_score("Aborted at Player 1", 1 if aborted_at == 1 else 0)
        self.log_episode_score("Aborted at Player 2", 1 if aborted_at == 2 else 0)

        # request count, parsed & violated request counts
        self.log_episode_score(metrics.METRIC_REQUEST_COUNT, episode_request_count)
        self.log_episode_score(metrics.METRIC_REQUEST_COUNT_VIOLATED, episode_violated_request_count)
        self.log_episode_score(metrics.METRIC_REQUEST_COUNT_PARSED, episode_parsed_request_count)

        # request success ratio
        if episode_request_count == 0:
            self.log_episode_score(metrics.METRIC_REQUEST_SUCCESS, 0)
        else:
            request_success_ratio = round(episode_parsed_request_count / float(episode_request_count), 4)
            self.log_episode_score(metrics.METRIC_REQUEST_SUCCESS, request_success_ratio)


class ImageGameBenchmark(GameBenchmark):

    def __init__(self):
        super().__init__(GAME_NAME)

    def get_description(self):
        return "Image Game simulation to generate referring expressions and fill a grid accordingly"

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return ImageGameMaster(experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return ImageGameScorer(experiment, game_instance)