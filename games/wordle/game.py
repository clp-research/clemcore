from typing import Dict, List

from backends import Model
from clemgame import get_logger
from games.wordle.utils.guesser import Guesser
from games.wordle.utils.critic import Critic
from games.wordle.utils.promptgenerator import PromptGenerator


logger = get_logger(__name__)


class WordleGame:
    def __init__(
        self,
        prompt_generator_config: Dict,
        max_attempts_per_game: int,
        max_retry_per_error: int,
        max_retry_invalid_word: int,
        max_word_length: int,
        use_critic: bool,
        max_critic_opinion_count: int,
        english_words_list: str,
        models: List[Model],
        response_format_keywords: Dict,
    ):
        self.max_attempts = max_attempts_per_game
        self.max_retry = max_retry_per_error
        self.max_retry_invalid_word = max_retry_invalid_word
        self.max_word_length = max_word_length
        self.use_critic = use_critic
        self.max_critic_opinion_count = max_critic_opinion_count
        self.english_words_list = english_words_list
        self.models = models
        self.response_format_keywords = response_format_keywords

        self.guesser = Guesser(self.models[0], self.response_format_keywords)
        self.guesser_mode = self.models[0].get_name()

        if len(self.models) > 1:
            if self.models[0] == self.models[1]:
                # Both Guesser and Critic using same model
                self.guess_critic = Critic(
                    self.models[0], self.response_format_keywords
                )
                self.guess_critic_mode = self.models[0].get_name()
            else:
                self.guess_critic = Critic(
                    self.models[1], self.response_format_keywords
                )
                self.guess_critic_mode = self.models[1].get_name()
        else:
            # Both Guesser and Critic using same model
            self.guess_critic = Critic(self.models[0], self.response_format_keywords)
            self.guess_critic_mode = self.models[0].get_name()

        self.guesser_prompt = []
        self.critic_prompt = []
        self.guesser_retry = 0
        self.guesser_retry_invalid_word = 0
        self.critic_retry = 0
        self.terminate = False
        self.guesser_error = None
        self.critic_error = None
        self.guesser_req_count = 0
        self.critic_req_count = 0
        self.guesser_parsed_req_count = 0
        self.critic_parsed_req_count = 0

        # While testing with clue - if no clue found, don't proceed
        self.use_clue = prompt_generator_config["use_clue"]
        if prompt_generator_config["use_clue"]:
            if not prompt_generator_config["target_word_clue"]:
                self.guesser_error = "NO_CLUE_FOUND"

        self.attempts = 0
        self.promptgen = PromptGenerator(**prompt_generator_config)

    def proceeds(self) -> bool:
        if self.terminate:
            if "human" in self.guesser_mode or (
                self.use_critic and "human" in self.guess_critic_mode
            ):
                logger.info(f"You Won!")
            return False

        if self.use_critic:
            if self.critic_error:
                if self.critic_retry < self.max_retry:
                    return True
                return False

        if self.guesser_error:
            if self.guesser_error == "NO_CLUE_FOUND":
                return False
            if self.guesser_error == "NOT_VALID_ENGLISH_WORD":
                if self.guesser_retry_invalid_word < self.max_retry_invalid_word:
                    return True
                return False
            if self.guesser_retry < self.max_retry:
                return True
            return False

        if self.attempts < self.max_attempts:
            return True

        if "human" in self.guesser_mode or (
            self.use_critic and "human" in self.guess_critic_mode
        ):
            logger.info(f"You Lost!")

        return False

    def turn(
        self,
        for_critic,
        guess,
        explanation,
        guess_feedback,
        agreement,
        agreement_explanation,
        error,
    ):
        if for_critic:
            if not error:
                utterance = self.promptgen.create_critic_prompt(
                    guess,
                    explanation,
                    guess_feedback,
                    self.critic_prompt,
                    agreement,
                    agreement_explanation,
                )
            else:
                self.critic_retry += 1
                utterance = self.promptgen.recreate(
                    error,
                    guess,
                    explanation,
                    self.critic_prompt,
                    agreement,
                    agreement_explanation,
                    True,
                )

            self.critic_req_count += 1
            self.promptgen.tailor_prompt(self.critic_prompt, self.guess_critic_mode)
            send_prompt = self.critic_prompt.copy()

            send_prompt, message, response = self.guess_critic(
                send_prompt, self.attempts
            )
            response_keyword = self.response_format_keywords["agreement"]  # "agreement"
            result = {
                response_keyword: "",
                self.response_format_keywords["explanation"]: "",
            }
            self.find_guess_explanation(
                self.guess_critic_mode, response, response_keyword, result
            )
            error = self.check_for_errors(result[response_keyword], False)
        else:
            if not error:
                if (
                    guess
                    and guess_feedback
                    and (
                        agreement is None
                        or agreement
                        in self.response_format_keywords["agreement_match_keywords"]
                    )
                ):
                    logger.debug(
                        f"Attempt [{self.attempts}] | Guess_Status [{guess_feedback}]"
                    )
                utterance = self.promptgen.create(
                    guess,
                    explanation,
                    guess_feedback,
                    self.guesser_prompt,
                    agreement,
                    agreement_explanation,
                )
            else:
                if error != "NOT_VALID_ENGLISH_WORD":
                    # Retry count is incremented only if the guess doesn't
                    # stick to rules (english word, only 5 letters, only
                    # alphabets)
                    # Since the models don't know what the game treats as
                    # valid words, this should not be treated as an error
                    self.guesser_retry += 1
                else:
                    # Some Models might not be able to generate a valid word
                    # in the first attempt. So, we are keeping a count of 5 tries
                    # to generate a valid word
                    self.guesser_retry_invalid_word += 1

                utterance = self.promptgen.recreate(
                    error,
                    guess,
                    explanation,
                    self.guesser_prompt,
                    agreement,
                    agreement_explanation,
                )

            self.guesser_req_count += 1

            self.promptgen.tailor_prompt(self.guesser_prompt, self.guesser_mode)
            send_prompt = self.guesser_prompt.copy()

            send_prompt, message, response = self.guesser(send_prompt, self.attempts)

            response_keyword = self.response_format_keywords["guess"]
            result = {
                response_keyword: "",
                self.response_format_keywords["explanation"]: "",
            }
            self.find_guess_explanation(
                self.guesser_mode, response, response_keyword, result
            )
            error = self.check_for_errors(result[response_keyword])

        if for_critic:
            self.critic_error = error
            if not self.critic_error:
                self.critic_retry = 0
                self.critic_parsed_req_count += 1
        else:
            self.guesser_error = error

            if not self.guesser_error:
                self.guesser_retry = 0
                self.guesser_retry_invalid_word = 0
                self.guesser_parsed_req_count += 1

            else:
                if self.guesser_error == "NOT_VALID_ENGLISH_WORD":
                    self.guesser_retry = 0
                    self.guesser_parsed_req_count += 1

                if "human" in self.guesser_mode:
                    logger.error("Some error in the guess: ", self.guesser_error)

        return utterance, send_prompt, message, response, result, error

    def increment_attempt(self):
        self.attempts += 1

    def get_game_status(self):
        if self.use_critic:
            if self.critic_error:
                return self.critic_error

        if self.guesser_error:
            return self.guesser_error
        else:
            if self.terminate:
                return "SUCCESS"
            if self.attempts == self.max_attempts:
                return "MAX_ATTEMPTS_REACHED"

    def check_guess_status(self, guess_feedback):
        letters = []
        colors = []
        for letter_color in guess_feedback.split(" "):
            letter, color = letter_color.split("<")
            letters.append(letter)
            colors.append(color)
        if all("green" in color for color in colors):
            self.terminate = True

    def find_keyword_match(self, text, keyword):
        # response contains the data in the format of guess:.. explanation:..
        # hence adding a colon to the keyword to find the exact match
        keyword = f"{keyword}:"
        start_index = text.find(keyword)
        end_index = text.find("\n", start_index)
        if start_index == -1 or end_index == -1:
            return "INVALID_FORMAT"
        start_index += len(keyword)
        word = text[start_index:end_index].strip()
        return word

    def find_guess_explanation(
        self, player_mode: bool, response: str, keyword: str, result: Dict
    ):
        if player_mode == "human":
            result[keyword] = response.split(" ")[-1].strip()

        else:
            # Adding additional new line to extract the text for each keyword until the next line
            response = response + "\n"
            if keyword == self.response_format_keywords["agreement"]:
                match_keyword = self.find_keyword_match(response, keyword)
                if (
                    match_keyword
                    not in self.response_format_keywords["agreement_match_keywords"]
                ):
                    result[keyword] = "INVALID_FORMAT"
                else:
                    result[keyword] = match_keyword
            else:
                result[keyword] = self.find_keyword_match(response, keyword)
            result["explanation"] = self.find_keyword_match(
                response, self.response_format_keywords["explanation"]
            )
        # return result

    def check_for_errors(self, word: str, for_guesser: bool = True):
        if word == "INVALID_FORMAT":
            return "INVALID_FORMAT"
        if for_guesser:
            if not word.isalpha():
                return "INVALID_WORD"
            if len(word) != self.max_word_length:
                return "INVALID_WORD_LENGTH"
            if word not in self.english_words_list:
                return "NOT_VALID_ENGLISH_WORD"

    def colorcode(self, guess):
        color_word = ""
        for letter_color in guess.split(" "):
            letter, color = letter_color.split("<")

            if "red" in color:
                color = "31"
            if "green" in color:
                color = "32"
            if "yellow" in color:
                color = "33"

            color_word += f"\033[{color};40m{letter}\033[0m"
        color_word = color_word.strip()
        return color_word
