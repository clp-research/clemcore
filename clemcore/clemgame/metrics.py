"""
Definition of metrics/scores that should be defined and logged for all games.
This constants should be used so that the naming is standardised across games.

Important: If the game is aborted, all episode-level scores must be set to numpy.nan 
and turn-level scores can be computed for the valid turns before the abortion action.
"""
import abc
import logging
from pathlib import Path
from typing import Dict, Union, final, Any, List

from clemcore.clemgame.resources import store_file

# common names
METRIC_ABORTED = "Aborted"
"""
At the episode level, either 0 or 1 whether the game play has been aborted (1) or not (0) 
(due to violation of the game rules e.g. not parsable response or re-prompt for n turns)) 
(this metric does not include games lost).
Record level: episode
"""

METRIC_LOSE = "Lose"
"""
At the episode level, either 0 or 1 whether the game play has been lost (1) or not (0) 
(this metric does not include aborted games; the game is lost, when the game goal is not reached 
within the declared number of max_turns, in this sense it’s the opposite of success).

This is always 0 if the game was aborted.

Record level: episode
"""

METRIC_SUCCESS = "Success"
"""
At the episode level, either 0 or 1 whether the game play has been successful (1) or not (0) 
(this metric does not include aborted games; the game is successful, when the game goal is reached 
within the declared number of max_turns, in this sense it’s the opposite of lost).

This is always 0 if the game was aborted.

Record level: episode
"""

METRIC_REQUEST_COUNT = "Request Count"
"""
How many requests to API calls have been made during the whole game play.
Record level: episode (and optionally also turn)
"""

METRIC_REQUEST_COUNT_PARSED = "Parsed Request Count"
"""
How many requests to API calls have been made during the whole game play that
could be successfully parsed.
Record level: episode (and optionally also turn)
"""

METRIC_REQUEST_COUNT_VIOLATED = "Violated Request Count"
"""
How many requests to API calls have been made during the whole game play that
could NOT be succesfully parsed.
Record level: episode (and optionally also turn)
"""

METRIC_REQUEST_SUCCESS_RATIO = "Request Success Ratio"
"""
METRIC_REQUEST_COUNT_PARSED / METRIC_REQUEST_COUNT
Record level: episode (and optionally also turn)
"""

BENCH_SCORE = 'Main Score'
""" 
The main score of the game. It is a value between 0 and 100 that summarises
the overall performance of a game play.

Should be np.nan if the game was aborted.
Record level: episode 
"""

METRIC_PLAYED = 'Played'
""" 
1 - ABORTED
This is computed and used by the eval scripts, which infer the % played from the aborted 
score. This metric should not be implemented/stored for new games if the given eval
scripts are used, to avoid duplicates.
Record level: episode 
"""

module_logger = logging.getLogger(__name__)

KEY_META = "meta"
KEY_PLAYERS = "players"
KEY_ROUND_SCORES = "round scores"
KEY_EPISODE_SCORES = "episode scores"


class GameScorer:
    """Calculates scores based on interaction logs. The resulting scores.json is structured like, for example:

    {
      "turn scores": {
        "0": { # here come your metrics
          "Accuracy": 0, # exemplary game specific metric
          "Violated Request Count": 0, # framework metric
          "Parsed Request Count": 1, # framework metric
          "Request Count": 1 # framework metric
        },
        ...
      "episode scores": {
        "Violated Request Count": 0, # framework metric
        "Parsed Request Count": 3, # framework metric
        "Request Count": 3, # framework metric
        "Request Success Ratio": 1.0, # framework metric
        "Aborted": 0, # framework metric
        "Success": 0, # framework metric
        "Lose": 1, # framework metric
        "Main Score": 0, # exemplary game specific metric
        "Repetition-Guesser": 0, # exemplary game specific metric
        "Repetition-Describer": 0 # exemplary game specific metric
      }
    }
    """

    def __init__(self, name: str, experiment: Dict, game_instance: Dict):
        """
        Args:
            name: The name of the game.
            experiment: The experiment to score.
            game_instance: The game instance to score.
        """
        self.game_name = name
        self.experiment = experiment
        self.game_instance = game_instance
        """ Stores values of score computation """
        self.scores = {
            KEY_META: {},
            KEY_PLAYERS: {},
            KEY_ROUND_SCORES: {},
            KEY_EPISODE_SCORES: {},
        }

    @final
    def store_scores(self, interactions_dir: Union[str, Path]):
        """Store calculated scores to scores.json file.
        Args:
            interactions_dir: The game's record directory path.
        """
        assert BENCH_SCORE in self.scores[KEY_EPISODE_SCORES], \
            "BENCH_SCORE is mandatory for evaluation, but missing in the calculated scores"
        file_path = store_file(self.scores, "scores.json", interactions_dir)
        self._on_store_scores(file_path)

    def _on_store_scores(self, file_path: str):
        """Hook to perform additional stuff after file has been saved"""
        pass

    @final
    def log_round_score(self, round_idx: int, score_name: str, score_value: Any):
        """Helper method to record a round (former 'turn') score/metric.
        Args:
            round_idx: The index for the round the score is to be recorded for.
            score_name: The name of the turn-level score/metric to record.
            score_value: The value to be recorded for the turn-level score/metric for this turn.
        """
        if isinstance(score_value, bool):
            module_logger.warning(f"{self.game_name}: Score {score_name} value is boolean, this can break the eval!")
        if round_idx not in self.scores[KEY_ROUND_SCORES]:
            self.scores[KEY_ROUND_SCORES][round_idx] = {}
        if score_name in self.scores[KEY_ROUND_SCORES][round_idx]:
            module_logger.warning(f"{self.game_name}: Score {score_name} overwritten at round {round_idx}!")
        self.scores[KEY_ROUND_SCORES][round_idx][score_name] = score_value
        module_logger.info(f"{self.game_name}: Logged round {round_idx} score {score_name}={score_value}.")

    @final
    def log_episode_score(self, score_name, score_value):
        """Helper method to record an episode-level score/metric for the whole episode.
        Args:
            score_name: The name of the episode-level score/metric to record.
            score_value: The value to be recorded for the episode-level score/metric.
        """
        if score_name in self.scores[KEY_EPISODE_SCORES]:
            module_logger.warning(f"{self.game_name}: Episode score {score_name} overwritten!")
        self.scores[KEY_EPISODE_SCORES][score_name] = score_value
        module_logger.info(f"{self.game_name}: Logged episode score {score_name}={score_value}.")

    @final
    def compute_scores(self, interactions: Dict) -> None:
        """Compute and log scores for a game episode. This method is used to perform complete scoring of an episode.
        Args:
            interactions: Dict containing the episode's interactions recorded during a benchmark run.
        """
        if KEY_META in interactions:  # if given, copy over meta info
            self.scores[KEY_META] = interactions[KEY_META]
        if "player_models" in interactions:  # if given, copy over players info
            self.scores["player_models"] = interactions["player_models"]
        if KEY_PLAYERS in interactions:  # if given, copy over players info
            self.scores[KEY_PLAYERS] = interactions[KEY_PLAYERS]
        self.score_rounds(interactions)
        self.score_episode(interactions)

    @final
    def score_rounds(self, interactions: Dict) -> None:
        """Iterate over episode rounds, calculate and log round scores.
        Args:
            interactions: Dict containing the episode's actions recorded during a benchmark run.
        """
        for round_idx, round_events in enumerate(interactions["turns"]):
            # compute standard framework metrics for the round
            round_request_count = interactions[METRIC_REQUEST_COUNT][round_idx]
            self.log_round_score(round_idx, METRIC_REQUEST_COUNT, round_request_count)

            round_violated_request_count = interactions[METRIC_REQUEST_COUNT_VIOLATED][round_idx]
            self.log_round_score(round_idx, METRIC_REQUEST_COUNT_VIOLATED, round_violated_request_count)

            round_parsed_request_count = interactions[METRIC_REQUEST_COUNT_PARSED][round_idx]
            self.log_round_score(round_idx, METRIC_REQUEST_COUNT_PARSED, round_parsed_request_count)

            round_request_success_ratio = round_parsed_request_count / round_request_count
            self.log_round_score(round_idx, METRIC_REQUEST_SUCCESS_RATIO, round_request_success_ratio)

            # compute game specific round metrics
            self.compute_round_score(round_idx, round_events)

    @abc.abstractmethod
    def compute_round_score(self, round_idx, round_events: List[Dict]) -> None:
        """Calculate and log round scores/metrics. This method is intended to contain any game-specific round scoring.

        Note: Use the log_turn_score helper method to log values.
        Args:
            round_idx: The index for the round the score is to be recorded for.
            round_events: List of player actions logged during the round.
        """
        pass

    @final
    def score_episode(self, interactions: Dict) -> None:
        """Calculate and record standard scores/metrics for the overall episode.
        Args:
            interactions: Dict containing the episode's interactions. This contains the actions recorded during
                a benchmark run.
        """
        # compute standard framework metrics for the overall episode
        overall_request_count = sum(interactions[METRIC_REQUEST_COUNT])
        self.log_episode_score(METRIC_REQUEST_COUNT, overall_request_count)

        overall_request_parsed = sum(interactions[METRIC_REQUEST_COUNT_PARSED])
        self.log_episode_score(METRIC_REQUEST_COUNT_PARSED, overall_request_parsed)

        overall_request_violated = sum(interactions[METRIC_REQUEST_COUNT_VIOLATED])
        self.log_episode_score(METRIC_REQUEST_COUNT_VIOLATED, overall_request_violated)

        overall_success_ratio = overall_request_parsed / overall_request_count
        self.log_episode_score(METRIC_REQUEST_SUCCESS_RATIO, overall_success_ratio)

        self.log_episode_score(METRIC_ABORTED, interactions[METRIC_ABORTED])
        self.log_episode_score(METRIC_LOSE, interactions[METRIC_LOSE])
        self.log_episode_score(METRIC_SUCCESS, interactions[METRIC_SUCCESS])

        # compute game specific episode metrics
        self.compute_episode_scores(interactions)

    @abc.abstractmethod
    def compute_episode_scores(self, interactions: Dict):
        """Compute any game specific game episode scores/metrics e.g. an overall accuracy metric.

        Note: This method must log the game's main BENCH_SCORE

        Args:
            interactions: Dict containing the logged episode's interactions.
        """
        pass
