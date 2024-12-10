from typing import Dict, Tuple, List, Union

import re
import copy
import numpy as np

from backends import Model, HumanModel
from clemgame.clemgame import GameMaster, GameBenchmark, GameScorer
from clemgame import get_logger
import clemgame.metrics as metrics


from games.wordle.utils.guessvalidator import GuessValidator
from games.wordle.utils.guesser import Guesser
from games.wordle.utils.critic import Critic
from games.wordle.utils.compute_metrics import ComputeMetrics

GAME_NAME = "wordle"

logger = get_logger(__name__)


class WordleGameMaster(GameMaster):
    def __init__(self, game_name: str, experiment: Dict, player_models: List[Model]):
        super().__init__(game_name, experiment, player_models)
        self.config = experiment

        self.model_a = player_models[0]
        if len(player_models) > 1:
            self.model_b = player_models[1]

        elif len(player_models) == 1 and self.config["use_critic"]:
            self.model_b = player_models[0]

        else:
            self.model_b = None

        # initialise attributes that will be used for the evaluation scores
        self.aborted: bool = False
        self.lose: bool = False
        self.success: bool = False
        self.complete_turns: int = 0

    def setup(self, game_id, target_word, target_word_clue, target_word_difficulty):
        self.game_id = game_id
        self.target_word = target_word.strip()
        self.target_word_clue = target_word_clue.strip()
        self.use_clue = self.config["use_clue"]
        self.use_critic = self.config["use_critic"]

        if self.use_clue:
            if isinstance(self.player_models[0], HumanModel):
                logger.info(f"Target word clue: {self.target_word_clue}")
        self.target_word_difficulty = target_word_difficulty.strip()

        self.guessvalidator = GuessValidator(self.target_word)
        self.guess_feedback = {}

        # instantiate both players
        self.player_a = Guesser(self.model_a, self.config["lang_keywords"])
        if self.model_b:
            self.player_b = Critic(self.model_b, self.config["lang_keywords"])
            player2_details = f"Word Guesser Critic ({self.model_b})"
        else:
            self.player_b = None
            player2_details = f"Player B: ObjectID Evaluator (Programmatic)"

        self.n_turns = self.config["common_config"]["n_turns"]
        # initialise game variables
        self.current_turn: int = 0
        self.reprompt = False
        self.reprompt_error = {key: None for key in range(1, self.n_turns + 1)}
        self.reprompt_answer = {key: None for key in range(1, self.n_turns + 1)}

        self.max_retry_per_error = self.config["common_config"]["max_retry_per_error"]
        self.cur_retry_per_error = {key: 0 for key in self.max_retry_per_error.keys()}

        # initialise common metrics
        self.request_counts = {
            key: 0 for key in range(1, self.n_turns + 1)
        }  # [0] * (self.n_turns + 1)
        self.parsed_request_counts = {
            key: 0 for key in range(1, self.n_turns + 1)
        }  # [0] * (self.n_turns + 1)
        self.violated_request_counts = {
            key: 0 for key in range(1, self.n_turns + 1)
        }  # [0] * (self.n_turns + 1)

        self.game_result = {
            "target_word": self.target_word,
            "target_word_clue": self.target_word_clue,
            "target_word_difficulty": self.target_word_difficulty,
            "use_clue": self.use_clue,
            "use_critic": self.use_critic,
            "guess": [],
            "guess_explanation": [],
            "critic_feedback": {},
        }
        self.before_critic = False

        # add initial prompts to each player's messages
        self.initiate(
            self.config["guesser_prompt"], self.config["guesser_critic_prompt"]
        )

        # always log the details of the players in this format (see logdoc)
        self.log_players(
            {
                "GM": "Game master for wordle",
                "Player 1": f"Word Guesser ({self.model_a})",
                "Player 2": player2_details,
            }
        )

        # log any additional keys that will be relevant for evaluation
        self.log_key("n_turns", self.n_turns)

    def initiate(self, prompt_player_a: str, prompt_player_b: str = None) -> None:
        """Initialise the dialogue history (firstlast specific)."""
        # append the initial message of each player to their history
        # the value user means the message is from an interlocutor of the model
        self.player_a.history.append({"role": "user", "content": prompt_player_a})

        if self.player_b:
            self.player_b.history.append({"role": "user", "content": prompt_player_b})

    def play(self) -> None:
        """Play the game until the end (mandatory)."""
        # play the game
        while self.proceed():
            self.current_turn += 1
            # always call log_next_turn when a new turn starts
            self.log_next_turn()
            self.turn()

            if self.success:
                action = {"type": "correct guess", "content": "game_result = WIN"}
                self.log_event(from_="GM", to="GM", action=action)
                break

        if not self.success:
            if self.aborted:
                action = {"type": "invalid format", "content": "game_result = ABORT"}
                self.log_event(from_="GM", to="GM", action=action)
            else:
                self.lose = True
                action = {"type": "incorrect guess", "content": "game_result = LOSS"}
                self.log_event(from_="GM", to="GM", action=action)

        # log all temporary game variables that are needed for evaluation
        self.log_eval_assets()

    def proceed(self) -> None:
        """Check if the game loop should continue (firstlast specific)."""
        return (
            self.current_turn < self.n_turns
            and not self.aborted
            and not self.lose
            and not self.success
        )

    def _parse_response(self, pattern, response):
        matches = pattern.findall(response)
        if len(matches) != 1:
            return None
        else:
            return matches[0].strip()

    def parse(self, player: str, response: str) -> dict:
        assert player in ("a", "b")

        if not response:
            return None, None

        if player == "a":
            if not response.startswith(self.config["lang_keywords"]["guess_lang"]):
                return None, None

            guess_keyword = self.config["lang_keywords"]["guess_lang"]

        else:
            if not response.startswith(self.config["lang_keywords"]["agreement_lang"]):
                return None, None

            guess_keyword = self.config["lang_keywords"]["agreement_lang"]

        explanation_keyword = self.config["lang_keywords"]["explanation_lang"]

        response = response.strip()
        additional_response = response.split("\n")
        if len(additional_response) > 2:
            return None, "UNKNOWN_TAGS"

        # Guess/Agreement should only be one word
        guess_pattern = re.compile(rf"{guess_keyword}([^\n]*)", re.IGNORECASE)
        # Explanation can spread across multiple lines
        explanation_pattern = re.compile(
            rf"{explanation_keyword}([^\n]*)", re.IGNORECASE)
        

        guess = self._parse_response(guess_pattern, response)
        if guess:
            guess = guess.lower().strip()

        explanation = self._parse_response(explanation_pattern, response)
        if explanation:
            explanation = explanation.strip()

        if not guess:
            return None, "MORE_THAN_ONE_GUESS"

        if player == "a":
            return {"guess": guess, "explanation": explanation}, None
        else:
            return {"agreement": guess, "explanation": explanation}, None

    def check_correctness(self, player: str, answer: str) -> str:
        """
        Check if the utterance conforms to rules
        The guess should be a single word
        The word should be a valid English word
        The guess word length should be exactly 5
        The word should not contain any characters other than letters
        """
        assert player in ("a", "b")

        if player == "a":
            to_check = answer["guess"]
        else:
            to_check = answer["agreement"]

        if to_check.count(" ") > 0 or not to_check.isalpha():
            return "INVALID_FORMAT"

        if player == "a":
            if len(to_check) != self.config["lang_keywords"]["max_word_length"]:
                return "INVALID_WORD_LENGTH"

            if to_check not in self.config["lang_keywords"]["official_words_list"]:
                return "NOT_VALID_WORD_FOR_GAME"

        else:
            if (
                to_check
                not in self.config["lang_keywords"]["agreement_match_keywords_lang"]
            ):
                return "NOT_VALID_CRITIC_WORD"

        return "VALID_GUESS"

    def _get_content_to_log_for_each_turn(self, player, answer, status):
        assert player in ("a", "b")

        content = ""
        if answer:
            if player == "a":
                content = f"attempts = {self.current_turn}\ntarget_word = {self.target_word}\n"
                if self.use_clue:
                    content += f"target_word_clue = {self.target_word_clue}\n"

                content += f"guess = {answer['guess']}\n"
                if status == "progress":
                    if not self.use_critic:
                        use_key = "no_critic"
                    else:
                        if self.before_critic:
                            use_key = "before_critic"
                        else:
                            use_key = "after_critic"

                    content += f"guess_feedback = {self.guess_feedback[self.current_turn][use_key]}"

            else:
                if status == "progress":
                    # content = f"agreement = {answer['agreement']}\nexplanation = {answer['explanation']}"
                    if answer["agreement"] == "yes":
                        content = "Critic agrees with the Guesser -- Sharing the critic's explanation with the guesser"
                    else:
                        content = "Critic does not agree with the Guesser -- Sharing the critic's explanation with the guesser"

        return content

    def _get_error_to_log(self, player, error):
        assert player in ("a", "b")
        if player == "a":
            if error == "INVALID_START_WORD":
                return "The response should always start with the keyword 'guess:'"
            elif error == "INVALID_FORMAT":
                return "The guess should be a single word and should only contain letters."
            elif error == "INVALID_WORD_LENGTH":
                return f"The length of the guessed word is not {self.config['lang_keywords']['max_word_length']}."
            elif error == "NOT_VALID_WORD_FOR_GAME":
                return "The guessed word is not a valid word for this game."
            elif error == "MORE_THAN_ONE_GUESS":
                return "The response should contain the 'guess:' keyword only once."
            elif error == "UNKNOWN_TAGS":
                return "The response should contain only the 'guess:' and 'explanation:' keywords and associated information."
        else:
            if error == "INVALID_START_WORD":
                return "The response should always start with the keyword 'agreement:'"
            elif error == "INVALID_FORMAT":
                return "The agreement should be a single word and should only be yes or no."
            elif error == "NOT_VALID_CRITIC_WORD":
                return f"The agreement should be one of the following: {self.config['lang_keywords']['agreement_match_keywords_lang']}"
            elif error == "MORE_THAN_ONE_GUESS":
                return "The response should contain the 'agreement:' keyword only once."
            elif error == "UNKNOWN_TAGS":
                return "The response should contain only the 'agreement:' and 'explanation:' keywords and associated information."

    def _handle_reprompt(self, player):
        assert player in ("a")

        if self.reprompt:
            # increase the counter of requests that conform to form rules
            self.parsed_request_counts[self.current_turn] += 1

            is_correct_reply = self.reprompt_error[self.current_turn]

            while (
                self.cur_retry_per_error[is_correct_reply]
                < self.max_retry_per_error[is_correct_reply]
            ):
                self.cur_retry_per_error[is_correct_reply] += 1

                content_to_log = f"Guesser Error: {is_correct_reply} while parsing Player 1's (model: {self.model_a}) response"
                action = {"type": "metadata", "content": content_to_log}
                self.log_event(from_="GM", to="GM", action=action)

                content = (
                    self.config["lang_keywords"]["error_prompt_text"][is_correct_reply]
                    + " "
                    + self.config["lang_keywords"]["error_prompt_text"]["RETRY"]
                    + "\n\n"
                )
                content += self._get_short_prompt(player)
                content = content.strip()

                self._append_utterance(content, player, "user")

                # also add the reply to the transcript
                action = {"type": "send message", "content": content}
                self.log_event(from_="GM", to="Player 1", action=action)

                self.reprompt = False
                answer = self._get_model_response(player, reprompt=True)

                is_valid_turn = self._check_validity(player, answer)
                if is_valid_turn:
                    self.cur_retry_per_error[is_correct_reply] = 0
                    self.reprompt = False
                    return True, answer
                
                # increase the counter of requests that conform to form rules
                self.parsed_request_counts[self.current_turn] += 1

                if self.reprompt:
                    is_correct_reply = self.reprompt_error[self.current_turn]
                else:
                    break

            logger.error(f"Current Turn: {self.current_turn}, Error in response from player a: {self.reprompt_answer[self.current_turn]}, is_correct_reply: {is_correct_reply}")
            self.reprompt = False
            if self.cur_retry_per_error[is_correct_reply] == self.max_retry_per_error[is_correct_reply]:
                self._handle_abort(
                    player, self.reprompt_answer[self.current_turn], is_correct_reply
                )
            return False, None

    def _handle_abort(self, player, answer, is_correct_reply):
        assert player in ("a", "b")

        self.aborted = True
        # log the abortion event
        content_to_log = ""#self._get_content_to_log_for_each_turn(player, answer, "abort")
        
        if player == "a":
            content_to_log += "\nGuess "
        else:
            content_to_log += "\nAgreement "

        content_to_log += f"does not conform to the format rules"
        if is_correct_reply:
            error_log = self._get_error_to_log(player, is_correct_reply)
            content_to_log += f"\nError: {error_log}"

        action = {"type": "metadata", "content": content_to_log.strip()}
        self.log_event(from_="GM", to="GM", action=action)
        # increase the counter of requests that violate form rules
        self.violated_request_counts[self.current_turn] += 1

    def set_reprompt(self, is_correct_reply, answer):
        if (
            self.cur_retry_per_error[is_correct_reply]
            < self.max_retry_per_error[is_correct_reply]
        ):
            self.reprompt = True
            self.reprompt_error[self.current_turn] = is_correct_reply
            self.reprompt_answer[self.current_turn] = answer
            # self.cur_retry_per_error[is_correct_reply] += 1

    def _check_validity(self, player: str, answer: str) -> bool:
        """Check if answer is valid and correct"""
        assert player in ("a", "b")
        # parse answer
        answer_parse, addl_error = self.parse(player, answer)
        if answer_parse is None:
            if addl_error:
                error = addl_error
            else:
                error = "INVALID_START_WORD"
            self._handle_abort(player, answer_parse, error)
            return False

        # if correct characters, check correctness wrt game rules
        is_correct_reply = self.check_correctness(player, answer_parse)
        if is_correct_reply != "VALID_GUESS":
            if is_correct_reply in self.config["common_config"]["retry_error_type"]:
                self.set_reprompt(is_correct_reply, answer_parse)
                return False

            self._handle_abort(player, answer_parse, is_correct_reply)
            return False

        # increase the counter of requests that conform to form rules
        self.parsed_request_counts[self.current_turn] += 1

        if not self.use_critic:
            use_key = "no_critic"
        else:
            if self.before_critic:
                use_key = "before_critic"
            else:
                use_key = "after_critic"

        if player == "a":
            val_guess = self.guessvalidator.validate(answer_parse["guess"])

            self.guess_feedback[self.current_turn][use_key] = val_guess
            if not self.use_critic or (self.use_critic and not self.before_critic):
                self.game_result["guess"].append((answer_parse["guess"], val_guess))
                self.game_result["guess_explanation"].append(
                    answer_parse["explanation"]
                )

            if self.use_critic:
                if self.current_turn not in self.game_result["critic_feedback"]:
                    self.game_result["critic_feedback"][self.current_turn] = {}

                self.game_result["critic_feedback"][self.current_turn][
                    use_key
                ] = answer_parse


            if use_key in ["no_critic", "after_critic"]:
                if self.is_guess_correct(val_guess):
                    logger.error(f"Guess is correct: {answer_parse['guess']}, setting success flag to True")
                    self.success = True

        else:
            if self.use_critic:
                self.game_result["critic_feedback"][self.current_turn][
                    "critic_response"
                ] = answer_parse

        # log the fact that the answer was correct
        if player == "a" and self.use_critic and not self.before_critic:
            guess_before_critic = self.game_result["critic_feedback"][
                self.current_turn
            ]["before_critic"]["guess"]
            guess_after_critic = self.game_result["critic_feedback"][self.current_turn][
                "after_critic"
            ]["guess"]
            critic_agreement = self.game_result["critic_feedback"][self.current_turn][
                "critic_response"
            ]["agreement"]
            if guess_before_critic != guess_after_critic:
                guess_response = "Change in player1's guess."
                logger.debug(
                    "Player1 changed the guess word after sharing the critic's opinion"
                )
                self.game_result["critic_feedback"][self.current_turn][
                    "guess_changed_after_critic_response"
                ] = True
            else:
                guess_response = "No change in player1's guess."
                logger.debug(
                    "Player1 did not change the guess word after sharing the critic's opinion"
                )
                self.game_result["critic_feedback"][self.current_turn][
                    "guess_changed_after_critic_response"
                ] = False

            guess_response += f"\nguess_before_critic_opinion: {guess_before_critic}\n\
                                critic_agreement: {critic_agreement}\nguess_after_critic_opinion: {guess_after_critic}\n\
                                                Proceeding with guess validation"

            action = {"type": "metadata", "content": guess_response}
            self.log_event(from_="GM", to="GM", action=action)

        if not self.use_critic or (self.use_critic and not self.before_critic):
            content_to_log = self._get_content_to_log_for_each_turn(
                player, answer_parse, "progress"
            )
            action = {"type": "metadata", "content": content_to_log}
            self.log_event(from_="GM", to="GM", action=action)

        return True

    def _prepare_playera_reply_to_playerb(self) -> str:
        content = (
            "\n\n"
            + self.config["lang_keywords"]["clue_lang"]
            + " "
            + self.target_word_clue
            + "\n"
        )

        if self.use_critic:
            guess_val = self.game_result["critic_feedback"][self.current_turn][
                "before_critic"
            ]["guess"]
            explanation_val = self.game_result["critic_feedback"][self.current_turn][
                "before_critic"
            ]["explanation"]
        else:
            guess_val = self.game_result["guess"][self.current_turn - 1]
            explanation_val = self.game_result["explanation"][self.current_turn - 1]

        content += self.config["lang_keywords"]["guess_lang"] + " " + guess_val + "\n"
        content += (
            self.config["lang_keywords"]["explanation_lang"] + " " + explanation_val
        )

        return content

    def _prepare_playerb_reply_to_playera(self, answer_b: str) -> str:
        content = (
            self.config["lang_keywords"]["clue_lang"]
            + " "
            + self.target_word_clue
            + "\n"
        )
        content += (
            self.config["lang_keywords"]["guess_agreement_lang"]
            + " "
            + answer_b["agreement"]
            + "\n"
        )
        content += (
            self.config["lang_keywords"]["agreement_explanation_lang"]
            + " "
            + answer_b["explanation"]
        )

        return content

    def _append_utterance(self, utterance: str, player: str, role: str) -> None:
        """Add an utterance to the history of a player (firstlast specific)."""
        assert player in ("a", "b")
        if player == "a":
            self.player_a.history.append({"role": role, "content": utterance})
        else:
            self.player_b.history.append({"role": role, "content": utterance})

    def _get_short_prompt(self, player: str) -> str:
        """Get short prompt for a player"""
        assert player in ("a", "b")

        content = (
            self.config["lang_keywords"]["error_prompt_text"]["INVALID_FORMAT"]
            + "\n"
        )

        if player == "a":
            guess_lang = self.config["lang_keywords"]["guess_lang"]
            guess_word_lang = self.config["lang_keywords"]["guess_word_lang"]
        else:
            guess_lang = self.config["lang_keywords"]["agreement_lang"]
            guess_word_lang = self.config["lang_keywords"]["agreement_word_lang"]


        content += guess_lang + " " + guess_word_lang + "\n"

        content += (
            self.config["lang_keywords"]["explanation_lang"]
            + " "
            + self.config["lang_keywords"]["explanataion_details_lang"]
        )
        return content
        
    def _add_guess_feedback_in_next_turns(self, player: str, use_key: str) -> str:        
        assert player in ("a")

        return (self.config["lang_keywords"]["guess_feedback_lang"]
                + " "
                + self.guess_feedback[self.current_turn - 1][use_key]
               )


    def _add_clue_to_initial_prompt(self, player: str) -> str:
        assert player in ("a")

        content = (
            "\n\n"
            + self.config["lang_keywords"]["clue_lang"]
            + " "
            + self.target_word_clue
            + "\n"
        )
        self.player_a.history[-1]["content"] += content
        return content


    def _prepare_player_query(self, player: str) -> str:
        assert player in ("a", "b")

        content = ""
        if player == "a":
            if self.use_critic:
                if not self.before_critic:
                    content = self._prepare_playerb_reply_to_playera(
                        self.game_result["critic_feedback"][self.current_turn][
                            "critic_response"
                        ]
                    )
                    if self.current_turn == 1:
                        content += "\n\n" + self._get_short_prompt(player)
                        content = content.strip()
                        self._append_utterance(content, player, "user")

                else:
                    if self.current_turn == 1:
                        content = self._add_clue_to_initial_prompt(player)
                    else:
                        content = self._add_guess_feedback_in_next_turns(player, "after_critic")
            else:
                if self.current_turn == 1:
                    if self.use_clue:
                        content = self._add_clue_to_initial_prompt(player)
                    
                else:
                    content = self._add_guess_feedback_in_next_turns(player, "no_critic")

        else:
            content = self._prepare_playera_reply_to_playerb()
            if self.current_turn == 1:
                self.player_b.history[-1]["content"] += content

        if self.current_turn > 1:
            content += "\n\n" + self._get_short_prompt(player)
            content = content.strip()
            self._append_utterance(content, player, "user")

        return content

    def _get_model_response(self, player: str, reprompt=False) -> str:
        assert player in ("a", "b")

        if player == "a":
            use_player = self.player_a
            use_from = "Player 1"

        else:
            use_player = self.player_b
            use_from = "Player 2"

        if not reprompt:
            content = self._prepare_player_query(player)

            # also add the reply to the transcript
            content_to_log = (
                content if self.current_turn > 1 else use_player.history[-1]["content"]
            )

            action = {"type": "send message", "content": content_to_log}
            self.log_event(from_="GM", to=use_from, action=action)

        # make an API call (or get a programmatic response) from player a
        prompt, raw_answer, answer = use_player(use_player.history, self.current_turn)
        # add API call to the records
        action = {"type": "get message", "content": answer}
        self.log_event(
            from_=use_from,
            to="GM",
            action=action,
            call=(copy.deepcopy(prompt), raw_answer),
        )
        # add reply to its own memory
        self._append_utterance(answer, player, "assistant")

        # increase the number of API requests
        self.request_counts[self.current_turn] += 1
        return answer

    def is_guess_correct(self, guess_feedback):
        letters = []
        colors = []
        for letter_color in guess_feedback.split(" "):
            letter, color = letter_color.split("<")
            letters.append(letter)
            colors.append(color)
        if all("green" in color for color in colors):
            return True
        return False
    
    def _handle_playera_response(self, answer_a: str) -> str:
        model_response = answer_a
        # check if the game should be aborted or lost
        is_valid_turn = self._check_validity("a", answer_a)
        if not is_valid_turn:
            if self.reprompt:
                # go ahead with reprompt
                logger.error(f"Current Turn: {self.current_turn}, INVALID_WORD in response; Reprompting player a")
                is_valid_turn, model_response = self._handle_reprompt("a")
                if not is_valid_turn:
                    # stop game
                    return None
            else:
                # stop game
                return None        
            
        return model_response

    def turn(self) -> None:
        """Perform a game turn, utterances by A and B"""
        self.before_critic = True
        self.guess_feedback[self.current_turn] = {}

        # get player A's reply and add it to its history
        answer_a = self._get_model_response("a")

        # check if the game should be aborted or lost
        answer_player_a = self._handle_playera_response(answer_a)
        logger.error(f"Current Turn: {self.current_turn}, Received response from player a: {answer_player_a}, self.success: {self.success}")


        if answer_player_a is None:
            return None

        # add A's reply to B's history
        if self.use_critic:
            logger.error(f"Current turn: {self.current_turn}, use_critic: {self.use_critic}, self.success: {self.success} calling player b")
            self.before_critic = False
            answer_b = self._get_model_response("b")
            # check if the game should be aborted or lost
            is_valid_turn = self._check_validity("b", answer_b)
            if not is_valid_turn:
                # stop game
                return None
            
            logger.error(f"Valid answer {answer_b} from player b, self.success: {self.success} calling player a")

            # get player A's reply and add it to its history
            answer_a = self._get_model_response("a")

            logger.error(f"Received answer {answer_a} from player a")
            # check if the game should be aborted or lost
            answer_player_a = self._handle_playera_response(answer_a)
            logger.error(f"After parsing, answer from player a is: {answer_player_a}, self.success: {self.success}")
            if answer_player_a is None:
                logger.error(f"Answer from player a is None, returning, self.success: {self.success}")
                return None
        self.complete_turns += 1

    def log_eval_assets(self) -> None:
        """Aux to log variables needed for scoring (firstlast specific)"""
        self.log_key("Played turns", self.current_turn)
        self.log_key("Complete turns", self.complete_turns)
        self.log_key(metrics.METRIC_ABORTED, self.aborted)
        self.log_key(metrics.METRIC_LOSE, self.lose)
        # self.log_key(metrics.METRIC_SUCCESS, self.success)
        log_req_counts = list(self.request_counts.values())
        log_req_counts = log_req_counts[: self.current_turn]
        self.log_key(metrics.METRIC_REQUEST_COUNT, log_req_counts)

        log_req_counts = list(self.parsed_request_counts.values())
        log_req_counts = log_req_counts[: self.current_turn]
        self.log_key(
            metrics.METRIC_REQUEST_COUNT_PARSED, log_req_counts)

        log_req_counts = list(self.violated_request_counts.values())
        log_req_counts = log_req_counts[: self.current_turn]

        self.log_key(
            metrics.METRIC_REQUEST_COUNT_VIOLATED, log_req_counts)
        self.log_key("Evaluation", self.game_result)


class WordleGameScorer(GameScorer):
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)
        self.cm = ComputeMetrics()

    def _compute_log_game_success(self, results: Dict) -> None:
        """Compute game success (mandatory)."""
        aborted = results[metrics.METRIC_ABORTED]
        loss = results[metrics.METRIC_LOSE]

        if not aborted and not loss:
            success = 1
        else:
            success = 0

        aborted = 1 if aborted else 0
        self.log_episode_score(metrics.METRIC_ABORTED, aborted)

        loss = 1 if loss else 0
        self.log_episode_score(metrics.METRIC_LOSE, loss)
        self.log_episode_score(metrics.METRIC_SUCCESS, success)

        return aborted, loss, success

    def _compute_log_request_count(self, results: Dict) -> None:
        """Compute request count (mandatory)."""
        turns_req_values = results[metrics.METRIC_REQUEST_COUNT]
        request_count = sum(turns_req_values)

        turns_parse_values = results[metrics.METRIC_REQUEST_COUNT_PARSED]
        parsed_request_count = sum(turns_parse_values)

        turns_violate_values = results[metrics.METRIC_REQUEST_COUNT_VIOLATED]
        violated_request_count = sum(turns_violate_values)

        req_success_ratio = round((parsed_request_count / request_count), 2)

        self.log_episode_score(metrics.METRIC_REQUEST_COUNT, request_count)
        self.log_episode_score(
            metrics.METRIC_REQUEST_COUNT_PARSED, parsed_request_count
        )
        self.log_episode_score(
            metrics.METRIC_REQUEST_COUNT_VIOLATED, violated_request_count
        )
        self.log_episode_score(metrics.METRIC_REQUEST_SUCCESS, req_success_ratio)

        log_scores = {
            "Request Count": turns_req_values,
            "Parsed Request Count": turns_parse_values,
            "Violated Request Count": turns_violate_values,
        }

        for key, value in log_scores.items():
            for idx, score in enumerate(value):
                self.log_turn_score(idx + 1, key, score)

    def compute_scores(self, episode_interactions: Dict) -> None:
        """Compute episode-level and turn-level scores (mandatory)."""

        results = episode_interactions["Evaluation"]
        aborted, loss, success = self._compute_log_game_success(episode_interactions)
        self._compute_log_request_count(episode_interactions)

        if aborted:
            episode_score = np.nan
            # Turn-scores can be logged even for aborted scenario
            # turn_score = [np.nan]
            # turn_strategy_score = [np.nan]
            speed = np.nan
            repeats_guess = np.nan
            num_guess_repeats = np.nan
        else:
            if loss:
                episode_score = 0
                speed = 0
            else:
                # Compute Episode Scores
                episode_score = self.cm.episodes(results["guess"])
                # Compute Rank
                speed = self.cm.speed(results["guess"], self.name)
            # Compute Guess repetition
            repeats_guess, num_guess_repeats = self.cm.repeats_guess(results["guess"])

        if results["use_critic"]:
            total_yes = np.nan
            total_no = np.nan
            use_same_guess_yes = np.nan
            use_diff_guess_yes = np.nan
            use_same_guess_no = np.nan
            use_diff_guess_no = np.nan
            overall_change = [np.nan]

            check_opinion = []
            for turn, value in results["critic_feedback"].items():
                if "before_critic" not in value or "after_critic" not in value or "critic_response" not in value:
                    continue
                check_opinion.append(
                    (
                        value["before_critic"]["guess"],
                        value["after_critic"]["guess"],
                        results["critic_feedback"][turn]["critic_response"][
                            "agreement"
                        ],
                    )
                )

            if check_opinion:
                change_results = self.cm.change_of_opinion(check_opinion)
                total_yes = change_results["total_yes"]
                total_no = change_results["total_no"]
                use_same_guess_yes = change_results["use_same_guess_yes"]
                use_diff_guess_yes = change_results["use_diff_guess_yes"]
                use_same_guess_no = change_results["use_same_guess_no"]
                use_diff_guess_no = change_results["use_diff_guess_no"]
                overall_change = change_results["overall_change"]

        # Compute Turn-wise Scores
        turn_score = [np.nan]
        turn_strategy_score = [np.nan]
        if results["guess"]:
            turn_score = self.cm.turns(results["guess"])
            # Compute strategy score
            turn_strategy_score = self.cm.turns_strategy(results["guess"])
            if len(turn_strategy_score) == 1:
                if aborted:
                    turn_strategy_score = [0]

        # self.log_episode_score("success", episode_score)
        self.log_episode_score(metrics.BENCH_SCORE, speed)
        self.log_episode_score("repeats guess", repeats_guess)
        self.log_episode_score("total guess repetitions", num_guess_repeats)
        # self.log_key("Target Word Difficulty", target_word_difficulty) todo scoring should not change the interaction

        for idx, score in enumerate(turn_score):
            self.log_turn_score(idx + 1, "closeness score", score)
        for idx, score in enumerate(turn_strategy_score):
            self.log_turn_score(idx + 1, "strategy score", score)

        if results["use_critic"]:
            for idx, score in enumerate(overall_change):
                self.log_turn_score(idx + 1, "change_of_opinion", overall_change[idx])

            if total_yes == np.nan:
                self.log_episode_score("Repetition-Guesser-On-Critic-Agreement", np.nan)
                self.log_episode_score(
                    "Non-Repetition-Guesser-On-Critic-Agreement", np.nan
                )
                self.log_episode_score(
                    "Repetition-Guesser-On-Critic-Disagreement", np.nan
                )
                self.log_episode_score(
                    "Non-Repetition-Guesser-On-Critic-Disagreement", np.nan
                )
            else:
                if total_yes != 0:
                    self.log_episode_score(
                        "Repetition-Guesser-On-Critic-Agreement",
                        round(use_same_guess_yes / total_yes, 2),
                    )
                    self.log_episode_score(
                        "Non-Repetition-Guesser-On-Critic-Agreement",
                        round(use_diff_guess_yes / total_yes, 2),
                    )
                else:
                    self.log_episode_score("Repetition-Guesser-On-Critic-Agreement", 0)
                    self.log_episode_score(
                        "Non-Repetition-Guesser-On-Critic-Agreement", 0
                    )

                if total_no != 0:
                    self.log_episode_score(
                        "Repetition-Guesser-On-Critic-Disagreement",
                        round(use_same_guess_no / total_no, 2),
                    )
                    self.log_episode_score(
                        "Non-Repetition-Guesser-On-Critic-Disagreement",
                        round(use_diff_guess_no / total_no, 2),
                    )
                else:
                    self.log_episode_score(
                        "Repetition-Guesser-On-Critic-Disagreement", 0
                    )
                    self.log_episode_score(
                        "Non-Repetition-Guesser-On-Critic-Disagreement", 0
                    )


class WordleGameBenchmark(GameBenchmark):
    def __init__(self):
        super().__init__(GAME_NAME)

    def get_description(self):
        return "Wordle Game"

    def create_game_master(
        self, experiment: Dict, player_models: List[Model]
    ) -> GameMaster:
        return WordleGameMaster(self.name, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return WordleGameScorer(self.name, experiment, game_instance)

    def is_single_player(self) -> bool:
        return True
