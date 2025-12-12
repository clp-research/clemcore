import logging
from functools import wraps
from typing import Any, Callable, TypeVar

# Define a TypeVariable to represent the signature of the function being decorated
# Callable[..., Any] means any function signature that returns anything
F = TypeVar("F", bound=Callable[..., Any])


def temporary_loglevel(logger: logging.Logger, level: int) -> Callable[[F], F]:
    """
    A decorator to temporarily set the log level for the decorated function's execution.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            original_level = logger.level
            logger.setLevel(level)
            try:
                # The type checker knows 'func' is of type F and has a specific signature
                result = func(*args, **kwargs)
                return result
            finally:
                logger.setLevel(original_level)

        return wrapper  # type: ignore[return-value]

    return decorator
