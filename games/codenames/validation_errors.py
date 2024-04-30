from enum import Enum
from .constants import CLUEGIVER, GUESSER

class ValidationErrorTypes(str, Enum):
    RAMBLING_ERROR = "rambling error"
    PREFIX_ERROR = "prefix error"
    RELATED_CLUE_ERROR = "clue is morphologically related to word on the board"
    TOO_FEW_TEXT = "answer only contained one line"
    CLUE_CONTAINS_SPACES = "clue contains spaces"
    CLUE_CONTAINS_INVALID_CHARACTERS = "clue contains non-alphabetical characters"
    GUESS_CONTAINS_INVALID_CHARACTERS = "guess contains non-alphabetical characters"
    CLUE_CONTAINS_NUMBER_OF_TARGETS = "clue line contains the number of targets"
    CLUE_ON_BOARD = "clue is word on board"
    NO_CORRECT_TARGET = "no correct target"
    REPEATED_TARGET = "target was already guessed"
    DOUBLE_TARGET = "target appears more than once in utterance"
    HALLUCINATED_TARGET = "target is hallucination"
    WRONG_NUMBER_OF_GUESSES = "wrong number of guesses"
    GUESS_IS_CLUE = "guess is clue word"
    REPEATED_GUESS = "guess was already guessed"
    DOUBLE_GUESS = "guess appears more than once in utterance"
    GUESS_HALLUCINATION = "guess word is hallucination"
    NO_CORRECT_GUESS = "no correct guess"

# general errors
class ValidationError(Exception):
    def __init__(self, player, error_type, utterance, message="Response does not follow the rules and is hence invalid."):
        super().__init__(message)
        self.attributes = {"player": player,
                           "type": error_type,
                           "utterance": utterance,
                           "message": message}
        
    def get_dict(self):
        return self.attributes

class PrefixError(ValidationError):
    def __init__(self, player, utterance, message, prefix):
        super().__init__(player, ValidationErrorTypes.PREFIX_ERROR, utterance, message)
        self.attributes["prefix"] = prefix

# Guesser errors
class MissingGuessPrefix(PrefixError):
    def __init__(self, utterance, prefix):
        message = f"Your guesses did not start with the correct prefix ({prefix})."
        super().__init__(GUESSER, utterance, message, prefix)

class GuesserRamblingError(ValidationError):
    def __init__(self, utterance):
        message = f"Your answer contained more than one line, please only give one round of guesses on one line."
        super().__init__(GUESSER, ValidationErrorTypes.RAMBLING_ERROR, utterance, message)

class WrongNumberOfGuessesError(ValidationError):
    def __init__(self, utterance, guesses, number_of_allowed_guesses):
        message = f"You made too many guesses ({len(guesses)}). You are only allowed to make {number_of_allowed_guesses} guesses!"
        super().__init__(GUESSER, ValidationErrorTypes.WRONG_NUMBER_OF_GUESSES, utterance, message)
        self.attributes["guesses"] = guesses
        self.attributes["number_of_allowed_guesses"] = number_of_allowed_guesses

class NoCorrectGuessError(ValidationError):
    def __init__(self, utterance, guesses, remaining_words):
        message = f"All of your guesses are invalid. You can only choose words as guesses from the remaining words and at least one guess has to be valid."
        super().__init__(GUESSER, ValidationErrorTypes.NO_CORRECT_GUESS, utterance, message)
        self.attributes["guesses"] = guesses
        self.attributes["remaining_words"] = remaining_words
    
class GuessContainsInvalidCharacters(ValidationError):
    def __init__(self, utterance, guess):
        message = f"Guessed word '{guess}' contains invalid characters, only put your target word, no other characters around it (apart from commas)."
        super().__init__(GUESSER, ValidationErrorTypes.GUESS_CONTAINS_INVALID_CHARACTERS, utterance, message)
        self.attributes["guess"] = guess
    
class GuessIsClueError(ValidationError):
    def __init__(self, utterance, clue, guess):
        message = f"Guessed word '{guess}' is the same word as the provided clue word, you should only select words from the provided list."
        super().__init__(GUESSER, ValidationErrorTypes.GUESS_IS_CLUE, utterance, message)
        self.attributes["guess"] = guess
        self.attributes["clue"] = clue

class HallucinatedGuessError(ValidationError):
    def __init__(self, utterance, guess, previous_guesses, remaining_words):
        message = f"Guessed word '{guess}' was not listed, you can only guess words provided in the lists."
        super().__init__(GUESSER, ValidationErrorTypes.GUESS_HALLUCINATION, utterance, message)
        self.attributes["guess"] = guess
        self.attributes["previous_guesses"] = previous_guesses
        self.attributes["remaining_words"] = remaining_words

class DoubleGuessError(ValidationError):
    def __init__(self, utterance, guess, remaining_words):
        message = f"Guessed word '{guess}' appears more than once in your answer, put each guessed word only once."
        super().__init__(GUESSER, ValidationErrorTypes.DOUBLE_GUESS, utterance, message)
        self.attributes["guess"] = guess
        self.attributes["remaining_words"] = remaining_words
    
class RepeatedGuessError(ValidationError):
    def __init__(self, utterance, guess, previous_guesses):
        message = f"Guessed word '{guess}' was already guessed previously, pay attention to the newly provided list of remaining words."
        super().__init__(GUESSER, ValidationErrorTypes.REPEATED_GUESS, utterance, message)
        self.attributes["guess"] = guess
        self.attributes["previous_guesses"] = previous_guesses

# Cluegiver errors

class RelatedClueError(ValidationError):
    def __init__(self, utterance, clue, similar_board_word):
        message = f"Your clue '{clue}' is morphologically similar to the word {similar_board_word}, please choose another clue word."
        super().__init__(CLUEGIVER, ValidationErrorTypes.RELATED_CLUE_ERROR, utterance, message)
        self.attributes["clue"] = clue
        self.attributes["similar_board_word"] = similar_board_word

class TooFewTextError(ValidationError):
    def __init__(self, utterance):
        message = f"Your answer did not contain clue and targets on two separate lines."
        super().__init__(CLUEGIVER, ValidationErrorTypes.TOO_FEW_TEXT, utterance, message)

class CluegiverRamblingError(ValidationError):
    def __init__(self, utterance):
        message = f"Your answer contained more than two lines, please only give one clue and your targets on two separate lines."
        super().__init__(CLUEGIVER, ValidationErrorTypes.RAMBLING_ERROR, utterance, message)

class RepeatedTargetError(ValidationError):
    def __init__(self, utterance, target, previous_targets):
        message = f"Targeted word '{target}' was already guessed, pay attention to the newly provided list of remaining words."
        super().__init__(CLUEGIVER, ValidationErrorTypes.REPEATED_TARGET, utterance, message)
        self.attributes["target"] = target
        self.attributes["previous_targets"] = previous_targets

class HallucinatedTargetError(ValidationError):
    def __init__(self, utterance, target, previous_targets, remaining_words):
        message = f"Targeted word '{target}' was not listed, you can only target words provided in the lists."
        super().__init__(CLUEGIVER, ValidationErrorTypes.HALLUCINATED_TARGET, utterance, message)
        self.attributes["target"] = target
        self.attributes["previous_targets"] = previous_targets
        self.attributes["remaining_words"] = remaining_words

class DoubleTargetError(ValidationError):
    def __init__(self, utterance, target, remaining_words):
        message = f"Targeted word '{target}' was targeted more than once, put each target only once."
        super().__init__(CLUEGIVER, ValidationErrorTypes.DOUBLE_TARGET, utterance, message)
        self.attributes["target"] = target
        self.attributes["remaining_words"] = remaining_words

class NoCorrectTargetError(ValidationError):
    def __init__(self, utterance, targets, remaining_words):
        message = f"All of your targets are invalid. You can only choose words as targets from the remaining words and at least one target has to be valid."
        super().__init__(CLUEGIVER, ValidationErrorTypes.NO_CORRECT_TARGET, utterance, message)
        self.attributes["targets"] = targets
        self.attributes["remaining_words"] = remaining_words

class ClueOnBoardError(ValidationError):
    def __init__(self, utterance, clue, board):
        message = f"Clue '{clue}' is one of the words on the board, please come up with a new word."
        super().__init__(CLUEGIVER, ValidationErrorTypes.CLUE_ON_BOARD, utterance, message)
        self.attributes["clue"] = clue
        self.attributes["board"] = board
    
class ClueContainsNumberOfTargets(ValidationError):
    def __init__(self, utterance, clue):
        message = f"Only provide the clue word and the target words in the requested format, do not add the number of targets on your own."
        super().__init__(GUESSER, ValidationErrorTypes.CLUE_CONTAINS_NUMBER_OF_TARGETS, utterance, message)
        self.attributes["clue"] = clue

class ClueContainsSpaces(ValidationError):
    def __init__(self, utterance, clue):
        message = f"Clue '{clue}' contains spaces and thus is not a single word."
        super().__init__(CLUEGIVER, ValidationErrorTypes.CLUE_CONTAINS_SPACES, utterance, message)
        self.attributes["clue"] = clue

class ClueContainsNonAlphabeticalCharacters(ValidationError):
    def __init__(self, utterance, clue):
        message = f"Clue '{clue}' contains non-alphabetical characters, please give a correct single English word."
        super().__init__(CLUEGIVER, ValidationErrorTypes.CLUE_CONTAINS_INVALID_CHARACTERS, utterance, message)
        self.attributes["clue"] = clue
    
class MissingCluePrefix(PrefixError):
    def __init__(self, utterance, prefix):
        message = f"Your clue did not start with the correct prefix ({prefix})."
        super().__init__(CLUEGIVER, utterance, message, prefix)

class MissingTargetPrefix(PrefixError):
    def __init__(self, utterance, prefix):
        message = f"Your targets did not start with the correct prefix ({prefix})."
        super().__init__(CLUEGIVER, utterance, message, prefix)