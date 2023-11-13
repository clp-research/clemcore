"""
Definition of metrics/scores that should be defined and logged for all games.
This constants should be used so that the naming is standardised across games.
"""

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
Record level: episode 
"""

METRIC_PLAYED = 'Played'
""" 
1 - ABORTED
This is used by the eval scripts, which infer the % played from the aborted score
This metric should thus not be computed for new games if the given eval
scripts are used, to avoid duplicates.
Record level: episode 
"""
