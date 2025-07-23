import json
import string
from typing import List, Any


def to_pretty_json(data: Any) -> str:
    """Format a dictionary or object as a pretty JSON string with proper newlines."""
    json_str = json.dumps(data, indent=2, sort_keys=True, default=str, ensure_ascii=False)
    return json_str.replace("\\n", "\n")


def remove_punctuation(text: str) -> str:
    """Remove punctuation from a string.
    Args:
        text: The string to remove punctuation from.
    Returns:
        The passed string without punctuation.
    """
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text
