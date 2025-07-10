"""
Definition of metrics/scores that should be defined and logged for all games.
This constants should be used so that the naming is standardised across games.

Important: If the game is aborted, all episode-level scores must be set to numpy.nan
and turn-level scores can be computed for the valid turns before the abortion action.
"""
import logging
from pathlib import Path
from typing import Dict, Union, Any, List

from clemcore.clemgame.metrics import METRIC_REQUEST_COUNT, METRIC_REQUEST_COUNT_PARSED, METRIC_REQUEST_COUNT_VIOLATED, \
    METRIC_REQUEST_SUCCESS_RATIO, METRIC_ABORTED, METRIC_LOSE, METRIC_SUCCESS
from clemcore.clemgame.resources import store_file

module_logger = logging.getLogger(__name__)

KEY_META = "meta"
KEY_PLAYERS = "players"
KEY_TURN_SCORES = "turn scores"
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
            KEY_TURN_SCORES: {},
            KEY_EPISODE_SCORES: {},
        }

    def store_scores(self, interactions_dir: Union[str, Path]):
        """Store calculated scores to scores.json file.
        Args:
            interactions_dir: The game's record directory path.
        """
        store_file(self.scores, "scores.json", interactions_dir)

    def log_turn_score(self, turn_idx: int, score_name: str, score_value: Any):
        """Helper method to record a round (former 'turn') score/metric.
        Args:
            turn_idx: The index for the round the score is to be recorded for.
            score_name: The name of the turn-level score/metric to record.
            score_value: The value to be recorded for the turn-level score/metric for this turn.
        """
        if isinstance(score_value, bool):
            module_logger.warning(f"{self.game_name}: Score {score_name} value is boolean, this can break the eval!")
        if turn_idx not in self.scores[KEY_TURN_SCORES]:
            self.scores[KEY_TURN_SCORES][turn_idx] = {}
        if score_name in self.scores[KEY_TURN_SCORES][turn_idx]:
            module_logger.warning(f"{self.game_name}: Score {score_name} overwritten at turn {turn_idx}!")
        self.scores[KEY_TURN_SCORES][turn_idx][score_name] = score_value
        module_logger.info(f"{self.game_name}: Logged turn {turn_idx} score {score_name}={score_value}.")

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
        self.score_turns(interactions)
        self.score_game(interactions)

    def score_turns(self, interactions: Dict) -> None:
        """Iterate over episode turns, calculate and log turn-level scores.
        Args:
            interactions: Dict containing the episode's actions recorded during a benchmark run.
        """
        raise NotImplementedError()

    def score_game(self, episode_interactions: Dict) -> None:
        """Calculate and record standard clembench metric scores for an episode.
        Args:
            episode_interactions: Dict containing the episode's interactions. This contains the actions recorded during
                a benchmark run.
        """
        self.score_game_end(episode_interactions)
        self.score_requests(episode_interactions)
        self.log_main_score(episode_interactions)

    def score_game_end(self, episode_interactions: Dict) -> None:
        """Calculate and record the ABORTED, LOSE and SUCCESS standard clembench metric scores.
        Convenience method to cover mandatory clembench metric scores, so their calculation does not need to be
        implemented anew for each new clemgame.
        Args:
            episode_interactions: Dict containing the episode's interactions. This contains the actions recorded during
                a benchmark run.
        """
        aborted = int(episode_interactions[METRIC_ABORTED])
        lose = int(episode_interactions[METRIC_LOSE]) if not aborted else 0
        success = 1 - lose if not aborted else 0

        self.log_episode_score(METRIC_ABORTED, aborted)
        self.log_episode_score(METRIC_LOSE, lose)
        self.log_episode_score(METRIC_SUCCESS, success)

    def score_requests(self, episode_interactions: Dict):
        """Calculate and record standard request-based clembench metric scores.
        Records total request count, parsed, violated, and success ratio of parsed requests over all requests in an
        episode.
        Convenience method to cover mandatory clembench metric scores, so their calculation does not need to be
        implemented anew for each new clemgame.
        Args:
            episode_interactions: Dict containing the episode's interactions. This contains the actions recorded during
                a benchmark run.
        """
        request_count = episode_interactions[
            METRIC_REQUEST_COUNT]  # could also be calculated by adding parsed and violated requests
        parsed_requests = episode_interactions[METRIC_REQUEST_COUNT_PARSED]
        violated_requests = episode_interactions[METRIC_REQUEST_COUNT_VIOLATED]

        self.log_episode_score(METRIC_REQUEST_COUNT, request_count)
        self.log_episode_score(METRIC_REQUEST_COUNT_PARSED, parsed_requests)
        self.log_episode_score(METRIC_REQUEST_COUNT_VIOLATED, violated_requests)
        self.log_episode_score(METRIC_REQUEST_SUCCESS_RATIO, parsed_requests / request_count)

    def log_main_score(self, episode_interactions: Dict):
        """Record the game's main score.
        Replace this method with a method that calculates and logs the clemgame's main score aka BENCH_SCORE.
        Must be implemented! Recording BENCH_SCORE is mandatory.
        Args:
            episode_interactions: Dict containing the episode's interactions. This contains the actions recorded during
                a benchmark run.
        """
        raise NotImplementedError()
