"""
Definition of metrics/scores that should be defined and logged for all games.
This constants should be used so that the naming is standardised across games.

Important: If the game is aborted, all episode-level scores must be set to numpy.nan 
and turn-level scores can be computed for the valid turns before the abortion action.
"""
import logging
from pathlib import Path
from typing import Dict, Union

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

METRIC_REQUEST_SUCCESS = "Request Success Ratio"
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


class GameScorer:
    """Calculates scores based on interaction logs."""

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
            "meta": {},
            "players": {},
            "turn scores": {},
            "episode scores": {},
        }

    def store_scores(self, interactions_dir: Union[str, Path]):
        """Store calculated scores to scores.json file."""
        store_file(self.scores, "scores.json", interactions_dir)

    def log_turn_score(self, turn_idx, score_name, score_value):
        """Record a turn-level score for a single turn.
        Args:
            turn_idx: The turn index for the turn the score is to be recorded for.
            score_name: The name of the turn-level score to record.
            score_value: The value to be recorded for the turn-level score for this turn.
        """
        if isinstance(score_value, bool):
            module_logger.warning(f"{self.game_name}: Score {score_name} value is boolean, this can break the eval!")
        if turn_idx not in self.scores["turn scores"]:
            self.scores["turn scores"][turn_idx] = {}
        if score_name in self.scores["turn scores"][turn_idx]:
            module_logger.warning(f"{self.game_name}: Score {score_name} overwritten at turn {turn_idx}!")
        self.scores["turn scores"][turn_idx][score_name] = score_value
        module_logger.info(f"{self.game_name}: Logged turn {turn_idx} score {score_name}={score_value}.")

    def log_episode_score(self, score_name, score_value):
        """Record an episode-level score for a single turn.
        Args:
            score_name: The name of the episode-level score to record.
            score_value: The value to be recorded for the episode-level score.
        """
        if score_name in self.scores["episode scores"]:
            module_logger.warning(f"{self.game_name}: Episode score {score_name} overwritten!")
        self.scores["episode scores"][score_name] = score_value
        module_logger.info(f"{self.game_name}: Logged episode score {score_name}={score_value}.")

    def compute_scores(self, episode_interactions: Dict) -> None:
        """Compute and log scores for a game episode.
        This method is used to perform complete scoring of an episode.
        Args:
            episode_interactions: Dict containing the episode's interactions. This contains the actions recorded during
                a benchmark run.
        """
        if "meta" in episode_interactions:  # if given, copy over meta info
            self.scores["meta"] = episode_interactions["meta"]
        if "player_models" in episode_interactions:  # if given, copy over players info
            self.scores["player_models"] = episode_interactions["player_models"]
        if "players" in episode_interactions:  # if given, copy over players info
            self.scores["players"] = episode_interactions["players"]
        self.score_turns(episode_interactions)
        self.score_game(episode_interactions)

    def score_turns(self, episode_interactions: Dict) -> None:
        """Iterate over episode turns, calculate and log turn-level scores.
        This method is intended to contain any game-specific turn-level scoring. Must be implemented!
        Use the log_turn_score method to log turn-level scores.
        Args:
            episode_interactions: Dict containing the episode's interactions. This contains the actions recorded during
                a benchmark run.
        """
        # Loop over turns, calculate and log turn-specific scores
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
        self.log_episode_score(METRIC_REQUEST_SUCCESS, parsed_requests / request_count)

    def log_main_score(self, episode_interactions: Dict):
        """Record the game's main score.
        Replace this method with a method that calculates and logs the clemgame's main score aka BENCH_SCORE.
        Must be implemented! Recording BENCH_SCORE is mandatory.
        Args:
            episode_interactions: Dict containing the episode's interactions. This contains the actions recorded during
                a benchmark run.
        """
        raise NotImplementedError()
