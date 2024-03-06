from typing import List, Tuple, Dict

from backends import Model
from clemgame import file_utils
from clemgame import metrics
from clemgame.clemgame import GameMaster, GameBenchmark, GameScorer
from clemgame import get_logger
from games.referencegame.game import ReferenceGame
import re
import math

GAME_NAME = "referencegame"
logger = get_logger(__name__)


class ReferenceGameMaster(GameMaster):

    def __init__(self, experiment: Dict, player_models: List[Model]):
        super().__init__(GAME_NAME, experiment, player_models)
        self.experiment = experiment
        self.game = None
        self.request_count = 0
        self.parsed_request_count = 0
        self.violated_request_count = 0
        self.aborted_ratio = 0

    def get_description(self) -> str:
        return "Reference Game simulation with GPT-3.5 model"


    def _on_setup(self, **game_instance):
        self.game_instance = game_instance

        self.game = ReferenceGame(self.game_instance, self.player_models)

        self.log_players({
            "GM": "Game master for referencegame",
            "Player_1": self.player_models[0].get_name(),
            "Player_2": self.player_models[1].get_name()}
        )

    def setup(self, **kwargs):
        self._on_setup(**kwargs)

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

        player_1_prompt, player_1_response, player_1_response_text = self.game.instruction_giver(self.game.given_instruction, None)

        # log the retrieved utterance
        action = {'type': 'get message', 'content': player_1_response_text}
        self.log_event(from_="Player 1", to="GM", action=action, call=(player_1_prompt, player_1_response))

        self.game.given_instruction.add_system_message(player_1_response_text)

        self.request_count += 1

        if re.match(self.game.player_1_response_pattern, player_1_response_text.lower()):
            parsed_instruction = ''
            if '\n' in player_1_response_text:
                parsed_instruction = player_1_response_text.split('\n')[0]
            else:
                parsed_instruction = player_1_response_text

            action = {'type': 'parse', 'content': parsed_instruction,
                      'original_content': player_1_response_text}
            self.log_event(from_="GM", to="GM", action=action)
            self.parsed_request_count += 1
            player_1_response_text = parsed_instruction
        else:
            # if the Player 1 message don't match the rule => start with "Expression: "
            # log the message and abort the game
            action = {'type': 'invalid format', 'content': 'Invalid generated expression',
                      'original_content': player_1_response_text}
            self.log_event(from_="GM", to="GM", action=action)

            self.violated_request_count += 1
            self.aborted_ratio = 1
            return

        # guess the grid - Player 2 side
        self.game.followed_instruction.add_user_message(self.game.player_2_prompt_header.replace('TARGET_EXPRESSION', player_1_response_text))

        # log the game master to player 2
        action = {'type': 'send message', 'content': self.game.followed_instruction.user_messages[-1]}
        self.log_event(from_="GM", to="Player 2", action=action)

        player_2_prompt, player_2_response, player_2_response_text = self.game.instruction_follower(self.game.followed_instruction, None)

        self.game.followed_instruction.add_system_message(player_2_response_text)

        self.game.turn_count += 1

        # log the retrieved utterance
        action = {'type': 'get message', 'content': player_2_response_text}
        self.log_event(from_="Player 2", to="GM", action=action, call=(player_2_prompt, player_2_response))
        self.request_count += 1

        # check if the Player 2 message matches the rule => grid
        if re.match(self.game.player_2_response_pattern, player_2_response_text.lower()):
            self.parsed_request_count += 1

            action = {'type': 'parse', 'content': player_2_response_text,
                      'original_content': player_2_response_text}
            self.log_event(from_="GM", to="GM", action=action)

        else:
            # abort the game if the output doesn't match the rule
            action = {'type': 'invalid format', 'content': 'Invalid generated choice',
                      'original_content': player_2_response_text}
            self.log_event(from_="GM", to="GM", action=action)

            self.violated_request_count += 1
            self.aborted_ratio = 1


class ReferenceGameScorer(GameScorer):

    def __init__(self, experiment: Dict, game_instance: Dict):
        super().__init__(GAME_NAME, experiment, game_instance)
        self.target_grid_name = game_instance["target_grid_name"]
        self.player_1_response_pattern = r'{}'.format(game_instance["player_1_response_pattern"])
        self.player_2_response_pattern = r'{}'.format(game_instance["player_2_response_pattern"])
        self.player_2_response_tag = game_instance["player_2_response_tag"]
        self.player_1_response_tag = game_instance["player_1_response_tag"]

    def compute_scores(self, episode_interactions: Dict) -> None:

        success = 0
        lost_count = 0
        expression_length_sum = 0
        expression_number_of_tokens = 0

        episode_request_count = 0
        episode_parsed_request_count = 0
        episode_violated_request_count = 0
        aborted = False
        number_of_turns = 0

        # loop over each turn and compute turn-specific scores for the metrics
        for t_index, turn in enumerate(episode_interactions["turns"]):

            turn_request_count = 0
            turn_parsed_request_count = 0
            turn_violated_request_count = 0

            # Player 1 message
            player_1_message = turn[1]['action']['content']

            turn_request_count += 1
            episode_request_count += 1

            # check if the Player 1 message follows the rule
            if re.match(self.player_1_response_pattern, player_1_message.lower()):
                turn_parsed_request_count += 1
                episode_parsed_request_count += 1
            else:
                turn_violated_request_count += 1
                episode_violated_request_count += 1
                aborted = True
                break

            number_of_turns += 1

            # Player 2 message
            player_2_message = turn[4]['action']['content']
            turn_request_count += 1
            episode_request_count += 1

            # check if the Player 2 message matches the rule -> start "Answer: ..."
            if re.match(self.player_2_response_pattern, player_2_message.lower()):
                turn_parsed_request_count += 1
                episode_parsed_request_count += 1

                # check if the target grid number matches the output from Player 2
                if self.target_grid_name.lower() in player_2_message.lower().replace(self.player_2_response_tag, '').strip():
                    success = 1
                else:
                    lost_count = 1
            else:
                turn_violated_request_count += 1
                episode_violated_request_count += 1
                aborted = True
                break

            # log the Player 1 - message length
            expression_length = len(player_1_message.lower().replace(self.player_1_response_tag, '').strip())
            self.log_turn_score(t_index, 'Generated Expression Length', expression_length)
            expression_length_sum += expression_length

            # log the Player 1 - number of tokens in the generated expression
            number_of_tokens = len(player_1_message.lower().replace(self.player_1_response_tag, '').strip().split(' '))
            self.log_turn_score(t_index, 'Generated Expression Number of Tokens', number_of_tokens)
            expression_number_of_tokens += number_of_tokens

            # log the request count, parsed & violated request counts
            self.log_turn_score(t_index, metrics.METRIC_REQUEST_COUNT, turn_request_count)
            self.log_turn_score(t_index, metrics.METRIC_REQUEST_COUNT_VIOLATED, turn_violated_request_count)
            self.log_turn_score(t_index, metrics.METRIC_REQUEST_COUNT_PARSED, turn_parsed_request_count)

            self.log_turn_score(t_index, metrics.METRIC_SUCCESS, success)

        if aborted:
            # if aborted all metrics get the value NaN
            self.log_episode_score('Average Generated Expression Length', math.nan)

            # average of number of tokens in generated expression
            self.log_episode_score('Average Generated Expression Number of Tokens', math.nan)

            # the last turn scores are also the scores for the episode
            # no need to calculate it again
            self.log_episode_score(metrics.METRIC_SUCCESS, 0)

            # lose ratio
            self.log_episode_score(metrics.METRIC_LOSE, 0)

            # aborted ratio
            self.log_episode_score(metrics.METRIC_ABORTED, 1)

            # benchmark score
            self.log_episode_score(metrics.BENCH_SCORE, math.nan)
        else:
            # average of expression length
            expression_length_sum = round(expression_length_sum / float(number_of_turns), 4)
            self.log_episode_score('Average Generated Expression Length', expression_length_sum)

            # average of number of tokens in generated expression
            expression_number_of_tokens = round(expression_number_of_tokens / float(number_of_turns), 4)
            self.log_episode_score('Average Generated Expression Number of Tokens', expression_number_of_tokens)

            # the last turn scores are also the scores for the episode
            # no need to calculate it again
            self.log_episode_score(metrics.METRIC_SUCCESS, success)

            # lose ratio
            self.log_episode_score(metrics.METRIC_LOSE, lost_count)

            # aborted ratio
            self.log_episode_score(metrics.METRIC_ABORTED, 0)

            # benchmark score
            self.log_episode_score(metrics.BENCH_SCORE, success * 100)

        # request count, parsed & violated request counts
        self.log_episode_score(metrics.METRIC_REQUEST_COUNT, episode_request_count)
        self.log_episode_score(metrics.METRIC_REQUEST_COUNT_VIOLATED, episode_violated_request_count)
        self.log_episode_score(metrics.METRIC_REQUEST_COUNT_PARSED, episode_parsed_request_count)

        # request success ratio
        if not aborted:
            request_success_ratio = round(episode_parsed_request_count / float(episode_request_count), 4)
            self.log_episode_score(metrics.METRIC_REQUEST_SUCCESS, request_success_ratio)
        else:
            self.log_episode_score(metrics.METRIC_REQUEST_SUCCESS, 0)


class ReferenceGameBenchmark(GameBenchmark):

    def __init__(self):
        super().__init__(GAME_NAME)

    def get_description(self):
        return "Reference Game simulation to generate referring expressions and guess the grid"

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return ReferenceGameMaster(experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return ReferenceGameScorer(experiment, game_instance)

def main():
    # select one instance
    experiments = file_utils.load_json("in/instances.json", "referencegame")
    instance = experiments["experiments"][0]["game_instances"][0]
    master = ReferenceGameMaster(instance, ("gpt-3.5-turbo", "gpt-3.5-turbo"))
    master.setup(**instance)
    master.play()


if __name__ == '__main__':
    main()
