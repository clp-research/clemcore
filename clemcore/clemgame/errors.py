from typing import Optional


class ResponseError(Exception):
    """
    General error class for problems with the player response.

    Developers can introduce more specific error types by subclassing this error.
    Alternatively, the 'key' attribute can be used to define more granular error types.
    """

    def __init__(self, reason: Optional[str] = None, response: Optional[str] = None, key: Optional[str] = None):
        """
        :param reason: (optional) a brief description of the cause
        :param response: (optional) the player's response
        :param key: (optional) a key word
        """
        super().__init__(reason)
        self.reason = reason
        self.response = response
        self.key = key

    def __str__(self):
        return f"{self.__class__.__name__}: {self.reason}"


class ProtocolError(ResponseError):
    """Raised when a message does not follow the communication protocol expected by the game master."""
    pass


class ParseError(ProtocolError):
    """
    This error is supposed to be raised when player messages cannot be parsed or understood by the game master e.g.
    because the response does not start with a specified prefix.

    For example:
        - taboo: clue giver messages should start with 'CLUE:'
        - wordle: guesser messages should start with 'GUESS:'
    """
    pass


class GameError(ResponseError):
    """Raised when a verbal action of a player causes problems for advancing the game."""
    pass


class RuleViolationError(GameError):
    """Raised when a verbal action of a player violates the specified game rules.

    For example:
        - taboo: mentioning the target word as the clue giver
        - wordle: guessing words that are not exactly 5 letters long
    """
    pass


class NotApplicableError(GameError):
    """Raised when a verbal action of a player cannot be applied to advance the game state."""
    pass
