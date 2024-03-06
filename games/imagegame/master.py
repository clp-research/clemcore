from typing import List, Tuple, Dict

from backends import Model
from clemgame import metrics
from clemgame.clemgame import GameMaster, GameBenchmark, GameScorer
from games.imagegame.game import ImageGame
from games.imagegame.evaluator import evaluate, calculate_flipped_pixels
from clemgame import get_logger
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

    def setup(self, **kwargs):
        self._on_setup(**kwargs)

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
        match = re.compile(self.game.player_1_terminate_pattern, re.IGNORECASE).match(player_1_response_text)
        if match:
            print('Parsed output: {} '.format(True))
            self.parsed_request_count += 1
            self.turn_request_stats[self.game.current_turn]['parsed_count'] += 1
            self.game.terminate = True
            return
        else:
            # continue if the Player didn't say -> Instruction: DONE
            # check if Player 1 message follows the rule => start with "Instruction:"
            player_1_message_matched = re.compile(self.game.player_1_response_pattern, re.IGNORECASE).match(player_1_response_text)
            if player_1_message_matched:
                parsed_instruction = ''
                if '\n' in player_1_response_text:
                    parsed_instruction = player_1_response_text.split('\n')[0]
                else:
                    parsed_instruction = player_1_response_text

                self.parsed_request_count += 1
                self.turn_request_stats[self.game.current_turn]['parsed_count'] += 1
                action = {'type': 'parse', 'content': parsed_instruction,
                          'original_content': player_1_response_text}
                self.log_event(from_="GM", to="GM", action=action)
                player_1_response_text = parsed_instruction
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
                    self.game.player_2_prompt_header + '\n' + player_1_response_text)
            else:
                self.game.followed_instruction.add_user_message(player_1_response_text)


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
            match = re.compile(self.game.player_2_response_pattern).match(player_2_response_text)
            if match:
                self.parsed_request_count += 1
                self.turn_request_stats[self.game.current_turn]['parsed_count'] += 1

                action = {'type': 'parse', 'content': player_2_response_text,
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
        self.player1_response_pattern = r'{}'.format(game_instance["player_1_response_pattern"])
        self.player2_response_pattern = r'{}'.format(game_instance["player_2_response_pattern"])
        self.player1_terminate_pattern = r'{}'.format(game_instance["player_1_terminate_pattern"])

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
        number_of_turns = 0

        # loop over each turn and calculate the metrics for both Player 1 and 2.

        for t_index, turn in enumerate(episode_interactions["turns"]):

            turn_request_count = 0
            turn_parsed_request_count = 0
            turn_violated_request_count = 0

            # Player 1 message
            player_1_message = turn[1]['action']['content']

            # Player generates "DONE"
            match = re.compile(self.player1_terminate_pattern, re.IGNORECASE).match(player_1_message)
            if match:
                break

            turn_request_count += 1
            episode_request_count += 1

            # check the Player 1 message if it matches the rule
            player_1_message_matched = re.compile(self.player1_response_pattern, re.IGNORECASE).match(player_1_message)
            if player_1_message_matched:
                if '\n' in player_1_message:
                    parsed_instruction = player_1_message.split('\n')[0]
                    player_1_message = parsed_instruction

                turn_parsed_request_count += 1
                episode_parsed_request_count += 1
            else:
                turn_violated_request_count += 1
                episode_violated_request_count += 1
                aborted = True
                # do not continue processing the rest of the turn when the game is aborted
                break

            # check if the turn includes the Player 2 message
            # in case the turn doesn't include an item and index position 4, it means the game has been aborted
            if len(turn) < 4:
                aborted = True
                break

            # Player 2 message
            player_2_message = turn[4]['action']['content']
            turn_request_count += 1
            episode_request_count += 1

            # check Player 2 message if it matches the instruction => grid
            match = re.compile(self.player2_response_pattern).match(player_2_message)
            if match:
                turn_parsed_request_count += 1
                episode_parsed_request_count += 1
            else:
                turn_violated_request_count += 1
                episode_violated_request_count += 1
                aborted = True
                break

            # calculate player-specific and turn-specific metrics

            try:
                precision, recall, f1 = evaluate(self.target_grid, player_2_message)
            except:
                pass

            # number of turns other
            number_of_turns += 1

            # Player 1 - message length
            expression_length = len(player_1_message.replace('Instruction:', '').strip())
            self.log_turn_score(t_index, 'Generated Expression Length', expression_length)
            expression_length_sum += expression_length

            # Player 1 - number of tokens in the generated expression
            number_of_tokens = len(player_1_message.replace('Instruction:', '').strip().split(' '))
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