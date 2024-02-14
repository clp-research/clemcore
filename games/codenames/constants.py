from enum import Enum

### Game related constants
SEED = 42
MAX_RETRIES = 2

### Game related string constants
GAME_NAME = "codenames"
TEAM = "team"
OPPONENT = "opponent"
INNOCENT = "innocent"
ASSASSIN = "assassin"
REVEALED = "revealed"
HIDDEN = "hidden"
TARGET = "target"
TOTAL = "total"
BOARD = "board"

class Turn_logs(str, Enum):
    CLUEGIVER_INVALID_FORMAT = "cluegiver invalid format"
    GUESSER_INVALID_FORMAT = "guesser invalid format"
    TARGETS = "targets"
    TEAM_REVEALED = f"{TEAM} {REVEALED}"
    OPPONENT_REVEALED = f"{OPPONENT} {REVEALED}"
    TARGET_REVEALED = f"target {REVEALED}"
    TURN_END_AFTER = "turn end after"

class Game_ends(str, Enum):
    ABORTED = "aborted"
    TEAM_WON = f"{TEAM} won"
    OPPONENT_WON = f"{OPPONENT} won"
    TEAM_WON_THROUGH_ASSASSIN = f"{TEAM_WON} through assassin"
    OPPONENT_WON_THROUGH_ASSASSIN = f"{OPPONENT_WON} through assassin"
BOARD_STATUS = "board status"
NUMBER_OF_TURNS = "number of turns"
GAME_END = "game end"

### Experiment related string constants
NAME = "name"
TYPE = "type"
OPPONENT_DIFFICULTY = "opponent difficulty"
ASSIGNMENTS = "assignments"