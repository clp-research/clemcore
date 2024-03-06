from typing import List, Tuple, Dict
import numpy as np

from backends import Model, HumanModel
from clemgame.clemgame import GameMaster, GameBenchmark, GameScorer
from clemgame import get_logger
import clemgame.metrics as metrics
from games.wordle.game import WordleGame
from games.wordle.utils.compute_metrics import ComputeMetrics
from games.wordle.utils.guessvalidator import GuessValidator

logger = get_logger(__name__)
GAME_NAME = "wordle"


class WordleGameMaster(GameMaster):
    def __init__(self, game_name: str, experiment: Dict, player_models: List[Model]):
        super().__init__(game_name, experiment, player_models)
        self.config = experiment
        self.player_model_names = [
            player_model.get_name() for player_model in player_models
        ]

    def setup(self, game_id, target_word, target_word_clue, target_word_difficulty):
        self.game_id = game_id

        self.players_dict = {"GM": "Game master for wordle"}
        self.players_dict["Player 1"] = f"Word Guesser ({self.player_model_names[0]})"

        if len(self.player_model_names) == 1:
            if self.config["use_critic"]:
                self.players_dict[
                    "Player 2"
                ] = f"Word Guesser Critic ({self.player_model_names[0]})"
            else:
                self.players_dict["Player 2"] = "Guess Word Evaluator (Programmatic)"
        else:
            self.players_dict[
                "Player 2"
            ] = f"Word Guesser Critic ({self.player_model_names[1]})"

        self.target_word = target_word.strip()
        self.target_word_clue = target_word_clue.strip()
        if self.config["use_clue"]:
            if isinstance(self.player_models[0], HumanModel):
                logger.info(f"Target word clue: {self.target_word_clue}")
        self.target_word_difficulty = target_word_difficulty

        self.guessvalidator = GuessValidator(self.target_word)

        game_config = {}
        game_config["max_attempts_per_game"] = self.config["common_config"][
            "max_attempts_per_game"
        ]
        game_config["max_retry_per_error"] = self.config["common_config"][
            "max_retry_per_error"
        ]
        game_config["max_retry_invalid_word"] = self.config["common_config"][
            "max_retry_invalid_word"
        ]
        game_config["max_word_length"] = self.config["common_config"]["max_word_length"]
        game_config["use_critic"] = self.config["use_critic"]
        game_config["max_critic_opinion_count"] = self.config["common_config"][
            "max_critic_opinion_count"
        ]
        game_config["english_words_list"] = self.config["english_words"]
        game_config["models"] = self.player_models
        game_config["response_format_keywords"] = self.config[
            "response_format_keywords"
        ]

        prompt_generator_config = {}
        prompt_generator_config["use_error_explanation"] = self.config["common_config"][
            "use_error_explanation"
        ]
        prompt_generator_config["use_system_message"] = self.config["common_config"][
            "use_system_message"
        ]
        prompt_generator_config["system_definition"] = self.config["system_definition"]
        prompt_generator_config["guesser_prompt"] = self.config["guesser_prompt"]
        prompt_generator_config["guesser_critic_prompt"] = self.config[
            "guesser_critic_prompt"
        ]
        prompt_generator_config["use_clue"] = self.config["use_clue"]
        prompt_generator_config["target_word_clue"] = self.target_word_clue
        prompt_generator_config["use_critic"] = self.config["use_critic"]
        prompt_generator_config["max_token_limit_openai_models"] = self.config[
            "common_config"
        ]["max_token_limit_openai_models"]

        self.game = WordleGame(prompt_generator_config, **game_config)

        self.turn_results = []

        self.log_players(self.players_dict)

    def _log_api_calls(
        self, utterance, send_prompt, message, response, result, from_, to_
    ):
        call = [send_prompt, message]
        action = {"type": "send message", "content": utterance[-1]["content"]}
        self.log_event(from_=to_, to=from_, action=action, call=call)

        action = {"type": "get message", "content": response}
        self.log_event(from_=from_, to=to_, action=action)

        action = {"type": "parse", "content": result}
        self.log_event(from_=to_, to=to_, action=action)

    def _log_metadata_single_content(self, data, data_for_computation=None):
        if not data_for_computation:
            action = {"type": "metadata", "content": data}
        else:
            action = {
                "type": "metadata",
                "content": data,
                "data_for_computation": data_for_computation,
            }
        self.log_event(from_="GM", to="GM", action=action)

    def _log_metadata(
        self,
        for_critic,
        guess,
        explanation,
        critic_agreement,
        critic_explanation,
        error,
        guess_feedback=None,
    ):
        if for_critic:
            metadata = {
                "target_word_clue": self.target_word_clue,
                "guesser": self.players_dict["Player 1"],
                "guess_critic": self.players_dict["Player 2"],
                "guess": guess,
                "explanation": explanation,
                "critic_agreement": critic_agreement,
                "critic_explanation": critic_explanation,
                "critic_error": error,
            }
            if error:
                content = f"Critic Error: {error} while parsing Player 2's (critic: {self.player_model_names[1]}) response, retrying"
            else:
                if critic_agreement == "yes":
                    content = f"Player 2 (model: {self.player_model_names[1]}) agrees with Player 1's (model: {self.player_model_names[0]}) guess, proceeding for validation"
                else:
                    content = f"Player 2 (model: {self.player_model_names[1]}) disagreed with Player 1's (model: {self.player_model_names[0]}) guess, so needs to make another guess"

            action = {"type": "metadata", "content": content, "critic_info": metadata}
        else:
            if not error:
                attempts = self.game.attempts + 1
            else:
                attempts = self.game.attempts
            if not error:
                if self.game.use_clue:
                    content = f"attempts: {attempts}\ntarget_word = {self.target_word}\ntarget_word_clue = {self.target_word_clue}\nguess: {guess}\nguess_feedback: {guess_feedback}"
                else:
                    content = f"attempts: {attempts}\ntarget_word = {self.target_word}\nguess: {guess}\nguess_feedback: {guess_feedback}"
            else:
                content = f"Guesser Error: {error} while parsing Player 1's (model: {self.player_model_names[0]}) response, retrying"
            metadata = {
                "attempts": attempts,
                "target_word": self.target_word,
                "target_word_difficulty": self.target_word_difficulty,
                "guesser": self.players_dict["Player 1"],
                "guess": guess,
                "explanation": explanation,
                "error": error,
            }
            if self.game.use_clue:
                metadata["target_word_clue"] = self.target_word_clue

            if self.game.use_critic:
                metadata["critic_agreement"] = critic_agreement
                metadata["critic_explanation"] = critic_explanation
                metadata["guess_critic"] = self.players_dict["Player 2"]
            action = {"type": "metadata", "content": content, "game_info": metadata}
        self.log_event(from_="GM", to="GM", action=action)

    def _log_error(self, error, from_, to_):
        action = {"type": "error", "content": error}
        self.log_event(from_=from_, to=to_, action=action)

    def _call_turn(
        self,
        for_critic,
        guess,
        explanation,
        guess_feedback,
        critic_agreement,
        critic_explanation,
        error,
    ):
        game = self.game

        utterance, send_prompt, message, response, result, error = game.turn(
            for_critic,
            guess,
            explanation,
            guess_feedback,
            critic_agreement,
            critic_explanation,
            error,
        )

        if not for_critic:
            # Log the guesser's response
            self._log_api_calls(
                utterance, send_prompt, message, response, result, "Player 1", "GM"
            )
            guess = result[self.config["response_format_keywords"]["guess"]]
            explanation = result[self.config["response_format_keywords"]["explanation"]]
            logger.debug("Receieved guess = {%s}", guess)
            return guess, explanation, error
        else:
            # Log the critic's response
            self._log_api_calls(
                utterance, send_prompt, message, response, result, "Player 2", "GM"
            )
            critic_agreement = result[
                self.config["response_format_keywords"]["agreement"]
            ]
            critic_explanation = result[
                self.config["response_format_keywords"]["explanation"]
            ]
            return critic_agreement, critic_explanation, error

    def _validate_guess(self, guess):
        guess_feedback = self.guessvalidator.validate(guess)
        self.game.check_guess_status(guess_feedback)
        self.turn_results.append([guess, guess_feedback])
        if self.game.use_critic:
            self.turn_req_count.append(
                self.game.guesser_req_count + self.game.critic_req_count
            )
            self.turn_parse_count.append(
                self.game.guesser_parsed_req_count + self.game.critic_parsed_req_count
            )
        else:
            self.turn_req_count.append(self.game.guesser_req_count)
            self.turn_parse_count.append(self.game.guesser_parsed_req_count)
        return guess_feedback

    def _handle_guesser_response_after_critics_opinion(
        self, guess, guess_before_criticism, critic_agreement
    ):
        guess_after_criticism = guess
        self.change_guess_words.append(
            [guess_before_criticism, guess_after_criticism, critic_agreement]
        )
        if guess_before_criticism != guess_after_criticism:
            content_to_log = f"Change in player1's guess\nguess_before_critic_opinion: {guess_before_criticism}\n\
                                                critic_agreement: {critic_agreement}\nguess_after_critic_opinion: {guess_after_criticism}\n\
                                                Proceeding with guess validation"
            logger.debug(
                "Player1 changed the guess word after sharing the critic's opinion"
            )
        else:
            content_to_log = f"No change in player1's guess\nguess_before_critic_opinion: {guess_before_criticism}\n\
                                                critic_agreement: {critic_agreement}\nguess_after_critic_opinion: {guess_after_criticism}\n\
                                                Proceeding with guess validation"
            logger.debug(
                "Player1 did not change the guess word after sharing the critic's opinion"
            )

        self._log_metadata_single_content(content_to_log)

        return guess_after_criticism

    def play(self) -> None:
        game = self.game
        guess = ""
        explanation = ""
        guess_feedback = None
        guesser_error = None
        critic_error = None
        critic_agreement = None
        critic_explanation = None
        self.turn_results = []
        self.turn_req_count = []
        self.turn_parse_count = []
        self.change_guess_words = []
        critic_opinion_count = 0
        guess_before_criticism = ""
        guess_after_criticism = ""
        if game.use_critic:
            ignore_critic = False
        else:
            ignore_critic = True

        while game.proceeds():
            # Call guesser
            logger.debug(
                "Game Master: attemtps: {%s} guess: {%s}, agreement {%s} error {%s}",
                game.attempts,
                guess,
                critic_agreement,
                guesser_error,
            )
            self.log_next_turn()
            guess, explanation, guesser_error = self._call_turn(
                False,
                guess,
                explanation,
                guess_feedback,
                critic_agreement,
                critic_explanation,
                guesser_error,
            )
            logger.info("Game Master: guess: {%s}, error {%s}", guess, guesser_error)
            if not guesser_error:
                if game.use_critic:
                    if not ignore_critic:
                        guess_before_criticism = guess
                        while game.proceeds:
                            (
                                critic_agreement,
                                critic_explanation,
                                critic_error,
                            ) = self._call_turn(
                                True,
                                guess,
                                explanation,
                                guess_feedback,
                                critic_agreement,
                                critic_explanation,
                                critic_error,
                            )

                            if not critic_error:
                                logger.info(
                                    "Game Master: critic_agreement: {%s}, error {%s}",
                                    critic_agreement,
                                    critic_error,
                                )
                                break

                        if not critic_error:
                            # Log critic opinion
                            if critic_agreement == "no":
                                critic_status = "Critic disagrees with the Guesser -- Sharing the critic's explanation with the guesser"
                            else:
                                critic_status = "Critic agrees with the Guesser -- Sharing the critic's explanation with the guesser"
                            logger.debug(critic_status)
                            self._log_metadata_single_content(critic_status)

                            critic_opinion_count += 1
                            if critic_opinion_count == game.max_critic_opinion_count:
                                logger.debug("Passing critic's opinion to guesser")
                                ignore_critic = True
                                guess_feedback = ""
                        else:
                            # Critic Error - log details and stop the game play
                            logger.info("Critic Error = %s", critic_error)
                            self._log_error(critic_error, "Player 2", "GM")
                            break
                    else:
                        # Received guesser's response after sharing critic's opinion
                        guess_after_criticism = (
                            self._handle_guesser_response_after_critics_opinion(
                                guess, guess_before_criticism, critic_agreement
                            )
                        )
                        # Proceed for guess validation
                        guess_feedback = self._validate_guess(guess)
                        self._log_metadata(
                            False,
                            guess,
                            explanation,
                            critic_agreement,
                            critic_explanation,
                            critic_error,
                            guess_feedback,
                        )
                        self.game.increment_attempt()

                        # Since no critic agreement available for this guess, in the next turn these fields should not be used/left empty!
                        critic_agreement = "do_not_use"
                        critic_explanation = "do_not_use"
                        ignore_critic = False
                        critic_opinion_count = 0

                else:
                    # Guess Validation
                    guess_feedback = self._validate_guess(guess)
                    self._log_metadata(
                        False,
                        guess,
                        explanation,
                        critic_agreement,
                        critic_explanation,
                        guesser_error,
                        guess_feedback,
                    )
                    self.game.increment_attempt()

            else:
                # Guesser Error - log details and retry
                logger.info("Guess = %s, Error = %s", guess, guesser_error)
                # self._log_error(error, "Player 1", "GM")
                self._log_metadata(
                    False,
                    guess,
                    explanation,
                    critic_agreement,
                    critic_explanation,
                    guesser_error,
                )

        if guess_feedback:
            logger.debug(f"Attempt [{game.attempts}] | Guess_Status [{guess_feedback}]")

        logger.info("Target Word: %s", self.target_word)

        if game.guesser_error == "NO_CLUE_FOUND":
            logger.info("No clue found for the target word: %s", self.target_word)
            self._log_metadata(
                False,
                guess,
                explanation,
                critic_agreement,
                critic_explanation,
                game.guesser_error,
            )

        # Update the final status
        self.game_final_status = game.get_game_status()

        if self.game_final_status in [
            "INVALID_FORMAT",
            "INVALID_WORD",
            "INVALID_WORD_LENGTH",
            "NOT_VALID_ENGLISH_WORD",
        ]:
            actual_status = self.game_final_status
            self.game_final_status = "ABORTED"
        elif self.game_final_status == "MAX_ATTEMPTS_REACHED":
            self.game_final_status = "LOSS"
        else:
            self.game_final_status = "WIN"

        logger.info("Game Result: %s", self.game_final_status)

        if self.game_final_status == "ABORTED":
            action = {
                "type": "invalid format",
                "content": "Aborted due to invalid format in response",
                "original_content": actual_status,
            }
            self.log_event(from_="GM", to="GM", action=action)

        # Log the turn-wise results
        data_for_computation = {}
        data_for_computation["player_1"] = self.player_model_names[0]
        if len(self.player_model_names) > 1:
            data_for_computation["player_2"] = self.player_model_names[1]
        else:
            data_for_computation["player_2"] = ""
        data_for_computation["total_attempts"] = self.game.attempts
        data_for_computation["turns_req_count"] = self.turn_req_count
        data_for_computation["turns_parse_count"] = self.turn_parse_count
        data_for_computation["turns_guess_feedback"] = self.turn_results
        data_for_computation["critic_guesses_change"] = self.change_guess_words
        data_for_computation["guesser_error"] = self.game.guesser_error
        data_for_computation["critic_error"] = self.game.critic_error
        data_for_computation["guesser retry count"] = self.game.guesser_retry
        data_for_computation["critic retry count"] = self.game.critic_retry
        data_for_computation["guesser_req_count"] = self.game.guesser_req_count
        data_for_computation["critic_req_count"] = self.game.critic_req_count
        data_for_computation[
            "guesser_parsed_req_count"
        ] = self.game.guesser_parsed_req_count
        data_for_computation[
            "critic_parsed_req_count"
        ] = self.game.critic_parsed_req_count
        data_for_computation["target_word"] = self.target_word
        data_for_computation["target_word_clue"] = self.target_word_clue
        data_for_computation["target_word_difficulty"] = self.target_word_difficulty
        data_for_computation["game_final_status"] = self.game_final_status
        data_for_computation["use_clue"] = self.game.use_clue
        data_for_computation["use_critic"] = self.game.use_critic
        self._log_metadata_single_content(
            f"game_result = {self.game_final_status}", data_for_computation
        )


class WordleGameScorer(GameScorer):
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)
        self.cm = ComputeMetrics()

    def compute_scores(self, episode_interactions: Dict) -> None:
        for key, val in episode_interactions.items():
            if key == "turns":
                # Look for last turn data and in that 'action' key
                if (
                    val
                    and val[-1]
                    and "action" in val[-1][-1]
                    and "data_for_computation" in val[-1][-1]["action"]
                ):
                    data_to_compute_scores = val[-1][-1]["action"][
                        "data_for_computation"
                    ]
                    if data_to_compute_scores:
                        aborted, loss = self._compute_game_status(
                            data_to_compute_scores["game_final_status"]
                        )
                        self._compute_req_count(
                            data_to_compute_scores["guesser_req_count"],
                            data_to_compute_scores["critic_req_count"],
                            data_to_compute_scores["guesser_parsed_req_count"],
                            data_to_compute_scores["critic_parsed_req_count"],
                            data_to_compute_scores["turns_req_count"],
                            data_to_compute_scores["turns_parse_count"],
                        )
                        self._compute_game_specific_metrics(
                            aborted,
                            loss,
                            data_to_compute_scores["turns_guess_feedback"],
                            data_to_compute_scores["use_critic"],
                            data_to_compute_scores["critic_guesses_change"],
                            data_to_compute_scores["target_word_difficulty"],
                        )
                        return

    def _compute_game_status(self, status):
        aborted = 0
        loss = 0
        success = 0

        if status == "ABORTED":
            aborted = 1
        elif status == "LOSS":
            loss = 1
        else:
            success = 1

        self.log_episode_score(metrics.METRIC_ABORTED, aborted)
        self.log_episode_score(metrics.METRIC_LOSE, loss)
        self.log_episode_score(metrics.METRIC_SUCCESS, success)
        return aborted, loss

    def _compute_req_count(
        self,
        guesser_req_count,
        critic_req_count,
        guesser_parsed_req_count,
        critic_parsed_req_count,
        turns_req_count,
        turns_parse_count,
    ):
        # Log API request count and parsed request count
        req_count = guesser_req_count + critic_req_count
        parsed_req_count = guesser_parsed_req_count + critic_parsed_req_count

        violated_req_count = req_count - parsed_req_count
        req_success_ratio = round((parsed_req_count / req_count), 2)

        self.log_episode_score(metrics.METRIC_REQUEST_COUNT, req_count)
        self.log_episode_score(metrics.METRIC_REQUEST_COUNT_PARSED, parsed_req_count)
        self.log_episode_score(
            metrics.METRIC_REQUEST_COUNT_VIOLATED, violated_req_count
        )
        self.log_episode_score(metrics.METRIC_REQUEST_SUCCESS, req_success_ratio)

        turns_req_values = []
        if turns_req_count:
            # Since the count is incremented for each turn, subtract the current count from the previous count to get actual count for this turn
            turns_req_values = [turns_req_count[0]]
            turns_req_count = [
                turns_req_count[i + 1] - turns_req_count[i]
                for i in range(len(turns_req_count) - 1)
            ]
            turns_req_values.extend(turns_req_count)
            for idx, score in enumerate(turns_req_values):
                self.log_turn_score(idx + 1, "Request Count", score)

        turns_parse_values = []
        if turns_parse_count:
            # Since the count is incremented for each turn, subtract the current count from the previous count to get actual count for this turn
            turns_parse_values = [turns_parse_count[0]]
            turns_parse_count = [
                turns_parse_count[i + 1] - turns_parse_count[i]
                for i in range(len(turns_parse_count) - 1)
            ]
            turns_parse_values.extend(turns_parse_count)
            for idx, score in enumerate(turns_parse_values):
                self.log_turn_score(idx + 1, "Parsed Request Count", score)

        turns_violate_count = [
            turns_req_values[i] - turns_parse_values[i]
            for i in range(len(turns_req_values))
        ]
        if turns_violate_count:
            for idx, score in enumerate(turns_violate_count):
                self.log_turn_score(idx + 1, "Violated Request Count", score)

    def _compute_game_specific_metrics(
        self,
        aborted,
        loss,
        turn_results,
        use_critic,
        change_guess_words,
        target_word_difficulty,
    ):
        if aborted:
            episode_score = np.nan
            # Turn-scores can be logged even for aborted scenario
            # turn_score = [np.nan]
            # turn_strategy_score = [np.nan]
            speed = np.nan
            repeats_guess = np.nan
            num_guess_repeats = np.nan
        elif loss:
            episode_score = 0
            speed = 0
            # Compute Guess repetition
            repeats_guess, num_guess_repeats = self.cm.repeats_guess(turn_results)
        else:
            # Compute Episode Scores
            episode_score = self.cm.episodes(turn_results)
            # Compute Rank
            speed = self.cm.speed(turn_results)
            # Compute Guess repetition
            repeats_guess, num_guess_repeats = self.cm.repeats_guess(turn_results)

        if use_critic:
            total_yes = np.nan
            total_no = np.nan
            use_same_guess_yes = np.nan
            use_diff_guess_yes = np.nan
            use_same_guess_no = np.nan
            use_diff_guess_no = np.nan
            overall_change = [np.nan]
            if change_guess_words:
                results = self.cm.change_of_opinion(change_guess_words)
                total_yes = results["total_yes"]
                total_no = results["total_no"]
                use_same_guess_yes = results["use_same_guess_yes"]
                use_diff_guess_yes = results["use_diff_guess_yes"]
                use_same_guess_no = results["use_same_guess_no"]
                use_diff_guess_no = results["use_diff_guess_no"]
                overall_change = results["overall_change"]

        # Compute Turn-wise Scores
        turn_score = [np.nan]
        turn_strategy_score = [np.nan]
        if turn_results:
            turn_score = self.cm.turns(turn_results)
            # Compute strategy score
            turn_strategy_score = self.cm.turns_strategy(turn_results)
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

        if use_critic:
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


def main(dry_run):
    # master = WordleGameMaster(dry_run)
    # master.setup()
    # master.play()
    bm = WordleGameBenchmark(dry_run)
    master = bm.create_game_master({}, dry_run)
    master.setup()
    master.play()


if __name__ == "__main__":
    main(dry_run=True)
