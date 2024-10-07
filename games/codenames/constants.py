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
TARGETED = "targeted"
HIDDEN = "hidden"
TARGET = "target"
TOTAL = "total"
BOARD = "board"
CLUEGIVER = "Cluegiver"
GUESSER = "Guesser"

class Turn_logs(str, Enum):
    VALIDATION_ERROR = "validation error"
    CLUE = "clue"
    TARGETS = "targets"
    GUESSES = "guesses"
    TEAM_REVEALED = f"{TEAM} {REVEALED}"
    OPPONENT_REVEALED = f"{OPPONENT} {REVEALED}"
    TARGET_REVEALED = f"target {REVEALED}"
    WORD_TARGETED = f"word targeted"
    TURN_END_AFTER = "turn end after"
    BOARD_STATUS = "board status"

BOARD_END_STATUS = "board end status"
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

class Turn_Scores(str, Enum):
    CLUEGIVER_NUMBER_OF_TARGETS = "Cluegiver Number of Targets"
    CLUEGIVER_TEAM_PRECISION = "Cluegiver Team Precision"
    CLUEGIVER_TEAM_RECALL = "Cluegiver Team Recall"
    CLUEGIVER_TEAM_F1 = "Cluegiver Team F1"
    GUESSER_NUMBER_OF_GUESSES = "Guesser Number of Guesses"
    GUESSER_NUMBER_OF_REVEALED_WORDS = "Guesser Number of Revealed Words"
    GUESSER_NUMBER_OF_UNREVEALED_GUESSES = "Guesser Number of Unrevealed Guesses"
    GUESSER_TARGET_PRECISION = "Guesser Target Precision"
    GUESSER_TARGET_RECALL = "Guesser Target Recall"
    GUESSER_TARGET_F1 = "Guesser Target F1"
    GUESSER_TEAM_PRECISION = "Guesser Team Precision"
    GUESSER_TEAM_RECALL = "Guesser Team Recall"
    GUESSER_TEAM_F1 = "Guesser Team F1"

class Episode_Scores(str, Enum):
    EFFICIENCY = "Efficiency"
    TARGET_F1 = "Average Target F1"
    RECALL = "Episode Recall"
    NEGATIVE_RECALL = "Episode Negative Recall"

### Experiment related string constants
NAME = "name"
TYPE = "type"
OPPONENT_DIFFICULTY = "opponent difficulty"
ASSIGNMENTS = "assignments"
VARIABLE = "experiment variable"
EXPERIMENT_NAME = "experiment name"