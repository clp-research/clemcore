from typing import Dict, List
import numpy as np
from backends import Model
from clemgame.clemgame import GameMaster, GameBenchmark, Player, DialogueGameMaster, GameScorer
from clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, METRIC_REQUEST_COUNT, \
    METRIC_REQUEST_COUNT_VIOLATED, METRIC_REQUEST_COUNT_PARSED, METRIC_REQUEST_SUCCESS, BENCH_SCORE
from clemgame import get_logger
from clemgame import file_utils, string_utils
import math
import re

GAME_NAME = "guesswhat"

logger = get_logger(__name__)


class Guesser(Player):
    def __init__(self, model: Model, max_turns):
        super().__init__(model)
        self.max_turns = max_turns

    def _custom_response(self, messages, turn_idx):
        # mock response
        if turn_idx < self.max_turns-1: 
            return "QUESTION: Is it a mammal?"
        else: 
            return "GUESS: Table"

class Answerer(Player):
    def __init__(self, model: Model, max_turns):
        super().__init__(model)
        self.max_turns = max_turns
        
    def _custom_response(self, messeges, turn_idx):
        # mock response
        if turn_idx < self.max_turns:
            return "ANSWER: No."
        elif turn_idx >= self.max_turns:
            raise Exception("We should not be here...")


class GuessWhat(DialogueGameMaster):
    """
    This class implements a "Guess What?" game in which player A (the Guesser) asks a
    question or makes a guess, and player B (the Answerer) responds with "yes" or "no".
    """

    def __init__(self, experiment: Dict, player_models: List[Model]):
        super().__init__(GAME_NAME, experiment, player_models)

        self.max_turns: int = experiment["max_turns"]
        self.question_tag = experiment["question_tag"]
        self.answer_tag = experiment["answer_tag"]
        self.guess_tag = experiment["guess_tag"]
        self.answer_variations = experiment["answer_variations"]
        self.guesser_initial_prompt = experiment["guesser_initial_prompt"]
        self.answerer_initial_prompt = experiment["answerer_initial_prompt"]
        self.letter_based_pattern = experiment["letter_based_pattern"]
        self.direct_guess_pattern = experiment["direct_guess_pattern"]
        self.length_question_pattern = experiment["length_question_pattern"]
        self.syllable_question_pattern = experiment["syllable_question_pattern"]
        self.pos_question_pattern = experiment["pos_question_pattern"]
        self.incorrect_guess = False
        self.correct_guess = False


    def check_question(self, question: str, candidate_list: List[str]) -> List[Dict]:

        """
        Checks the questions content to see if they follow the rules of the game. Returns a list of content errors found.
        """
        errors = []

        question_text = question.replace(self.question_tag, "").strip().lower()
        letter_based_pattern = re.compile(r'{}'.format(self.letter_based_pattern, re.IGNORECASE))
        direct_guess_pattern = re.compile(r'{}'.format(self.direct_guess_pattern, re.IGNORECASE))
        length_question_pattern = re.compile(r'{}'.format(self.length_question_pattern, re.IGNORECASE))
        syllable_question_pattern = re.compile(r'{}'.format(self.syllable_question_pattern, re.IGNORECASE))
        pos_question_pattern = re.compile(r'{}'.format(self.pos_question_pattern, re.IGNORECASE))


        if letter_based_pattern.search(question_text):
            errors.append({
                "message": "Invalid question. Asking about specific letters or their positions is not allowed.",
                "type": 1
            })

        direct_guess_match = direct_guess_pattern.match(question_text)
        if direct_guess_match:
            errors.append({
                "message": "Invalid question. Guessing without 'GUESS:' format is not allowed.",
                "type": 2
            })

        if length_question_pattern.search(question_text):
            errors.append({
                "message": "Invalid question. Asking about the length of the target word is not allowed.",
                "type": 3
            })

        if syllable_question_pattern.search(question_text):
            errors.append({
               "message": "Invalid question. Asking about the number of syllables is not allowed.",
                "type": 4
            })

        if pos_question_pattern.match(question_text):
            errors.append({
                "message": "Invalid question. Asking about the part of speech (POS) of the target word is not allowed.",
                "type": 5
        })

        return errors

    def _on_setup(self, **game_instance):
        logger.info("_on_setup")
        self.game_instance = game_instance

        self.target_word = game_instance["target_word"]
        self.candidate_list = game_instance["candidate_list"]

        self.guesser_initial_prompt = self.guesser_initial_prompt.replace("$LIST$", str(self.candidate_list)).replace("$N$", str(self.max_turns-1))
        self.answerer_initial_prompt = self.answerer_initial_prompt.replace("$TARGET WORD$", str(self.target_word))
        
        self.guesser = Guesser(self.player_models[0], self.max_turns)
        self.answerer = Answerer(self.player_models[1], self.max_turns)

        self.add_player(self.guesser)
        self.add_player(self.answerer)

         # Two different variables for the errors 
        self.invalid_format = False
        self.invalid_content = False

        self.guess_word = None


    def _on_before_game(self):
        self.add_user_message(self.guesser, self.guesser_initial_prompt)

    def _does_game_proceed(self):
        if self.invalid_format:
            self.log_to_self("invalid format", "abort game")
            return False
        if self.invalid_content:
            self.log_to_self("invalid content", "abort game")
            return False
        if self.correct_guess:
            self.log_to_self("correct guess", "end game")
            return False
        if self.incorrect_guess:
            self.log_to_self("incorrect guess", "end game")
            return False
        if self.current_turn >= self.max_turns:
            self.log_to_self("max turns reached", str(self.max_turns))
            return False
        return True

    def _validate_player_response(self, player: Player, utterance: str) -> bool:

        self.invalid_format = False  # Reset the flags at the beginning of validation
        self.invalid_content = False 

        if player == self.guesser:

            # Check if the response is neither a valid question nor a valid guess format
            if not (utterance.startswith(self.question_tag) or utterance.startswith(self.guess_tag)):
                self.log_to_self("invalid format", "Invalid format. Guesser must use the form 'QUESTION: ' or 'GUESS: '.")
                self.invalid_format = True

                return False

            # Validate the question format
            if utterance.startswith(self.question_tag):
                question_text = utterance[len(self.question_tag):].strip()

                # Check for multiple "QUESTION:" occurrences
                if utterance.count(self.question_tag) > 1:
                    self.log_to_self("invalid format", "Multiple questions detected in a single turn.")
                    self.invalid_format = True
                    return False

                # Check if there is text after the question mark
                if "?" in question_text:
                    parts = question_text.split("?")
                    if len(parts) > 2 or parts[1].strip() != "":
                        self.log_to_self("invalid format", "Invalid format. Question must stop after the question mark.")
                        self.invalid_format = True
                        return False
                    
                # Check for specific content-related errors by calling check_question
                errors = self.check_question(utterance, self.candidate_list)
                if errors:
                    # Log all errors as invalid content and return False
                    for error in errors:
                        self.log_to_self("invalid content", error["message"])
                        self.invalid_content = True
                    return False

            # Validate the guess format
            elif utterance.startswith(self.guess_tag):

                guess_word = utterance[len(self.guess_tag):].strip().lower()
                guess_word = string_utils.remove_punctuation(guess_word)
                self.guess_word = guess_word

                # Check if the guess contains more than one word
                if len(guess_word.split()) > 1:
                    self.invalid_format = True
                    return False

                # Check correct and incorrect guess
                if guess_word == self.target_word.lower():
                    self.correct_guess = True
                    self.log_to_self("correct guess", guess_word)
                else:
                    self.incorrect_guess = True
                    self.log_to_self("incorrect guess", guess_word)

                # If guess format is valid, allow it
                return True

        elif player == self.answerer:
            if utterance not in self.answer_variations:
                self.invalid_format = True
                return False
        return True

    def _after_add_player_response(self, player: Player, utterance: str):
        if player == self.guesser:
            if self.current_turn == 0:
                # Include first question in the prompt
                prompt_with_first_question = f"{self.answerer_initial_prompt}\n\n{utterance}"
                self.add_user_message(self.answerer, prompt_with_first_question)
            else:
                self.add_user_message(self.answerer, utterance)
        if player == self.answerer:
            if not self.incorrect_guess and not self.correct_guess:  # Check if a guess has not been made
                self.add_user_message(self.guesser, utterance)


class GuessWhatScorer(GameScorer):

    def __init__(self, experiment: Dict, game_instance: Dict):
        super().__init__(GAME_NAME, experiment, game_instance)

    def compute_scores(self, episode_interactions: Dict) -> None:
        turn_scores = []

        invalid_format_guesser_count = 0
        invalid_format_answerer_count = 0
        invalid_content_guesser_count = 0
        invalid_content_answerer_count = 0
        
        guesser_won = False
        max_turns = self.experiment["max_turns"]
        
        speed_score = 0

        # Set lower_bound_turns based on the level to calculate speed
        game_level = self.experiment["name"]

        num_categories_1 = 4
        num_features_3 = 4
        
        if game_level == "Level_1" or "Abs_Level_1":
            lower_bound_turns = num_categories_1 + 1  
        elif game_level == "Level_2" or "Abs_Level_2":
            lower_bound_turns = math.log2(max_turns) + 1
        elif game_level == "Level_3" or "Abs_Level_3":
            lower_bound_turns = num_features_3 + 1


        for turn_idx, turn in enumerate(episode_interactions["turns"]):
            turn_score = {"request_count": 1}
            # Track invalid responses during this turn 
            invalid_format_in_turn = False
            invalid_content_in_turn = False

            for event_idx, event in enumerate(turn):
                action = event["action"]

                # Handle invalid format and content per player by looking at the previous event's "from" field
                if action["type"] == "invalid format":
                    if event_idx - 1 >= 0:  # Check if there is a previous event
                        previous_event = turn[event_idx - 1]
                        if previous_event["from"] == "Player 1":  # Guesser
                            invalid_format_guesser_count += 1
                        elif previous_event["from"] == "Player 2":  # Answerer
                            invalid_format_answerer_count += 1
                    invalid_format_in_turn = True

                if action["type"] == "invalid content":
                    if event_idx - 1 >= 0:  # Check if there is a previous event
                        previous_event = turn[event_idx - 1]
                        if previous_event["from"] == "Player 1":  # Guesser
                            invalid_content_guesser_count += 1
                        elif previous_event["from"] == "Player 2":  # Answerer
                            invalid_content_answerer_count += 1
                    invalid_content_in_turn = True

                if action["type"] == "correct guess":
                    guesser_won = True
    

            if invalid_format_in_turn or invalid_content_in_turn:
                turn_score["violated_request_count"] = 1
                turn_score["parsed_request_count"] = 0
            else:
                turn_score["violated_request_count"] = 0
                turn_score["parsed_request_count"] = 1

            self.log_turn_score(turn_idx, 'Accuracy', 1 if guesser_won else 0)
            self.log_turn_score(turn_idx, METRIC_REQUEST_COUNT_VIOLATED, turn_score["violated_request_count"])
            self.log_turn_score(turn_idx, METRIC_REQUEST_COUNT_PARSED, turn_score["parsed_request_count"])
            self.log_turn_score(turn_idx, METRIC_REQUEST_COUNT,turn_score["request_count"])
            turn_scores.append(turn_score)

        # Sum up turn scores
        violated_request_count = sum(turn["violated_request_count"] for turn in turn_scores)
        self.log_episode_score(METRIC_REQUEST_COUNT_VIOLATED, violated_request_count)
        
        parsed_request_count = sum(turn["parsed_request_count"] for turn in turn_scores)
        self.log_episode_score(METRIC_REQUEST_COUNT_PARSED, parsed_request_count)

        request_count = sum(turn["request_count"] for turn in turn_scores)
        self.log_episode_score(METRIC_REQUEST_COUNT, request_count)

        # # Log the overall scores
        # self.log_episode_score(METRIC_REQUEST_COUNT_VIOLATED, violated_request_count)
        # self.log_episode_score(METRIC_REQUEST_COUNT_PARSED, parsed_request_count)
        # self.log_episode_score(METRIC_REQUEST_COUNT, request_count)

        # Compute the request success ratio
        if request_count != 0:
            self.log_episode_score(METRIC_REQUEST_SUCCESS, parsed_request_count / request_count)
        else:
            self.log_episode_score(METRIC_REQUEST_SUCCESS, 0)

        # If any violation occurred, mark the game as aborted and don't compute BENCH_SCORE
        if invalid_format_in_turn or invalid_content_in_turn:
            self.log_episode_score(METRIC_ABORTED, 1)
            self.log_episode_score(BENCH_SCORE, np.nan)
        else:
            # No abort, continue with normal scoring
            self.log_episode_score(METRIC_ABORTED, 0)

            if guesser_won:
                self.log_episode_score(METRIC_SUCCESS, 1)
                self.log_episode_score(METRIC_LOSE, 0)

                # The maximum speed will be reached if the guesser wins the game in the average minimum turns calculated for each level and 
                # decreases as a consistent rate as the number of turns increases 
                if request_count <= lower_bound_turns:
                    speed_score = 100
                    self.log_episode_score("Speed", 100)
                else:
                    speed_score = 100 * (max_turns - request_count) / (max_turns - lower_bound_turns)
                self.log_episode_score("Speed", max(0, speed_score))

                bench_score = max(0, speed_score)
                self.log_episode_score(BENCH_SCORE, bench_score)

            else:
                self.log_episode_score(METRIC_SUCCESS, 0)
                self.log_episode_score(METRIC_LOSE, 1)
                self.log_episode_score(BENCH_SCORE, 0)
                

        # Log invalid response counts for both players
        self.log_episode_score("Invalid format guesser response", invalid_format_guesser_count)
        self.log_episode_score("Invalid format answerer response", invalid_format_answerer_count)
        self.log_episode_score("Invalid content guesser response", invalid_content_guesser_count)
        self.log_episode_score("Invalid content answerer response", invalid_content_answerer_count)


class GuessWhatGameBenchmark(GameBenchmark):
    def __init__(self):
        super().__init__(GAME_NAME)

    def get_description(self):
        return "Guess What? game between two agents where one asks questions to guess the target word from list of candidates and the other answers with 'yes' or 'no'."

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return GuessWhat(experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return GuessWhatScorer(experiment, game_instance)


def main():
    # select one experiment and instance
    experiments = file_utils.load_json("in/instances.json", GAME_NAME)
    experiment_1 = experiments["experiments"][0]
    game_1 = experiment_1["game_instances"][0]
    master = GuessWhat(experiment_1, ["mock", "mock"])
    master.setup(**game_1)
    master.play()


if __name__ == '__main__':
    main()
