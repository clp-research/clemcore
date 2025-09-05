import abc
from typing import Any, List, Dict
import json


class GameEventLogger(abc.ABC):
    """
    Abstract base class defining the interface for logging game events.

    Implementers of this interface handle recording various types of
    game-related events, such as rounds, player info, requests, and generic actions.
    """

    @abc.abstractmethod
    def log_next_round(self):
        """
        Notify that a new round or turn has started in the game.

        Use this to group or segment events chronologically by rounds.
        """
        pass

    @abc.abstractmethod
    def log_game_end(self, auto_count_logging: bool = True):
        """
        Notify that the current game has ended.

        This can be used to finalize or flush logs associated with the current game session.

        Args:
            auto_count_logging: For legacy games, the counts were logged by the games.
                New games can rely on the count logging by the recorder.
        """
        pass

    @abc.abstractmethod
    def count_request(self):
        """
        Track or increment a counter for successful player responses.

        The exact semantics depend on the implementation, used for monitoring usage or interactions.
        """
        pass

    @abc.abstractmethod
    def count_request_violation(self):
        """
        Track or increment a counter for violating player responses.

        Useful for monitoring policy violations, errors, or unexpected conditions during gameplay.
        """
        pass

    @abc.abstractmethod
    def log_key(self, key: str, value: Any):
        """
        Log a generic key-value pair associated with the game session.

        Args:
            key: A string identifier for the type or category of the logged data.
            value: The corresponding data or content to log. Must be JSON-serializable or convertible.
        """
        pass

    @abc.abstractmethod
    def log_player(self, player_name: str, game_role: str, model_name: str):
        """
        Log details about a player participating in the current game session.

        Args:
            player_name: The display or identifier name of the player (e.g., "Player 1", "Game Master").
            game_role: The role or function of the player within the game (e.g., "Guesser", "Answerer").
            model_name: The name or type of the model or agent representing the player, e.g. AI model identifier.
        """
        pass

    @abc.abstractmethod
    def log_event(self, from_, to, action, call=None):
        """
        Log a game event representing an action from one entity to another.

        Args:
            from_: Identifier string of the entity (player or game master) originating the event.
            to: Identifier string of the entity targeted by the event.
            action: The action or event description, typically a structured data dict or object.
            call: Optional tuple representing an API call related to the event. The first element is
                  the processed input prompt; the second element is the raw response from the API.
                  This enables logging both the event and its associated request/response data.
        """
        pass


class GameEventSource(GameEventLogger):
    """
    Composite event source that aggregates multiple GameEventLogger instances.

    This class acts as a central notifier that delegates event logging calls
    to all registered loggers, enabling multi-target logging without changing the event interface.

    For backward compatibility, instead of using a generic notify(event) method,
    it exposes explicit methods matching those in the GameEventLogger interface.
    """

    def __init__(self):
        """
        Initialize a new GameEventSource with no registered loggers.
        """
        self._loggers: List[GameEventLogger] = []

    def register(self, logger: GameEventLogger):
        """
        Register a new GameEventLogger to receive delegated event notifications.

        Args:
            logger: An instance implementing GameEventLogger to register.
        """
        self._loggers.append(logger)

    def register_many(self, loggers: List[GameEventLogger]):
        """
        Register multiple GameEventLogger instances at once.

        Args:
            loggers: A list of GameEventLogger instances to register.
        """
        self._loggers.extend(loggers)

    def log_next_round(self):
        """Delegate notification of a new round to all registered loggers."""
        for logger in self._loggers:
            logger.log_next_round()

    def log_game_end(self, auto_count_logging: bool = True):
        """
        Delegate notification of game end to all registered loggers.

        Args:
            auto_count_logging: For legacy games, the counts were logged by the games.
                New games can rely on the count logging by the recorder."""
        for logger in self._loggers:
            logger.log_game_end(auto_count_logging)

    def count_request(self):
        """Delegate request count increment to all registered loggers."""
        for logger in self._loggers:
            logger.count_request()

    def count_request_violation(self):
        """Delegate request violation count increment to all registered loggers."""
        for logger in self._loggers:
            logger.count_request_violation()

    def log_event(self, from_, to, action, call=None):
        """
        Delegate logging of a game event to all registered loggers.

        Args:
            from_: Originator identifier.
            to: Target identifier.
            action: Event action details.
            call: Optional associated API call information.
        """
        for logger in self._loggers:
            logger.log_event(from_, to, action, call)

    def log_key(self, key: str, value: Any):
        """
        Delegate logging of a key-value pair to all registered loggers.

        Args:
            key: Identifier string for the log entry.
            value: The data associated with the key.
        """
        if type(value) == set:
            value = list(value)
        try:
            json.dumps(value)  # Ensure value is JSON-serializable
        except TypeError:
            raise ValueError(f"Value for key '{key}' is of type {type(value)} and thus not JSON-serializable.\nValue: {value}")
        for logger in self._loggers:
            logger.log_key(key, value)

    def log_player(self, player_name: str, game_role: str, model_name: str):
        """
        Delegate logging of player information to all registered loggers.

        Args:
            player_name: Name of the player.
            game_role: Role of the player in the game.
            model_name: Name of the player's model or agent.
        """
        for logger in self._loggers:
            logger.log_player(player_name, game_role, model_name)
