from enum import Enum
import string

### Game related constants
SEED = 42
MAX_RETRIES = 2
CHARS_TO_STRIP = " .,<>\"'"
NUMBERS_TO_STRIP = " ," + ''.join(string.digits)

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
CLUEGIVER = "cluegiver"
GUESSER = "guesser"

class Turn_logs(str, Enum):
    VALIDATION_ERROR = "validation error"
    CLUE = "clue"
    TARGETS = "targets"
    GUESSES = "guesses"
    TEAM_REVEALED = f"{TEAM} {REVEALED}"
    OPPONENT_REVEALED = f"{OPPONENT} {REVEALED}"
    TARGET_REVEALED = f"target {REVEALED}"
    TURN_END_AFTER = "turn end after"

BOARD_STATUS = "board status"
NUMBER_OF_TURNS = "Number of turns"
GAME_ENDED_THROUGH_ASSASSIN = "Game ended through assassin"

class ValidationError_types(str, Enum):
    RAMBLING_ERROR = "rambling error"
    PREFIX_ERROR = "prefix error"
    WRONG_NUMBER_OF_GUESSES = "wrong number of guesses"
    INVALID_GUESS = "invalid guess"
    RELATED_CLUE_ERROR = "clue is morphologically related to word on the board"
    TOO_FEW_TEXT = "answer only contained one line"
    CLUE_NOT_SINGLE_WORD = "clue is not a single word"
    CLUE_NOT_WORD = "clue is not a word"
    CLUE_ON_BOARD = "clue is word on board"
    INVALID_TARGET = "target is invalid"

### Experiment related string constants
NAME = "name"
TYPE = "type"
OPPONENT_DIFFICULTY = "opponent difficulty"
ASSIGNMENTS = "assignments"