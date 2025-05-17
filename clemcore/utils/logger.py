"""
Centralized logging configuration for clemcore.
"""

import json
import logging
from typing import Any, Optional

import colorlog

logging.SUCCESS = 25
logging.addLevelName(logging.SUCCESS, "SUCCESS")


def success(self, message, *args, **kwargs):
    """Log a message with the SUCCESS level."""
    if self.isEnabledFor(logging.SUCCESS):
        self._log(logging.SUCCESS, message, args, **kwargs)


logging.Logger.success = success


def format_message(message: str, level: str, highlight: Optional[str] = None) -> str:
    """
    Format a log message with optional highlighting.

    Args:
        message: The message to format
        level: The log level
        highlight: Optional text to highlight in the message

    Returns:
        Formatted message string
    """
    if highlight and highlight in message:
        message = message.replace(highlight, f"**{highlight}**")

    if ":" in message:
        parts = message.split(":", 1)
        message = f"{parts[0]}: {parts[1].strip()}"

    return message


def setup_logger(name: str) -> logging.Logger:
    """
    Set up a logger with consistent formatting and configuration.

    Args:
        name: The name of the logger (typically __name__ of the module)

    Returns:
        A configured logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = colorlog.ColoredFormatter(
            "%(asctime)s | %(name)s.%(funcName)s | %(log_color)s%(levelname)s%(reset)s | %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "SUCCESS": "bold_green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )

        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

        logger.propagate = False

    return logger


def format_json(data: Any) -> str:
    """Format a dictionary or object as a pretty JSON string with proper newlines."""
    json_str = json.dumps(
        data, indent=2, sort_keys=True, default=str, ensure_ascii=False
    )
    return json_str.replace("\\n", "\n")
