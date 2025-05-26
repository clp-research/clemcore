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


# the methods below might be composed into a new class
# ---
def to_pair_descriptor(model_pair: List[str]) -> str:
    """Converts a model pair list into a model pair string.
    Args:
        model_pair: A list with two model ID/name strings.
    Returns:
        The model pair descriptor string for the model pair.
    """
    assert len(model_pair) == 2, "Model pair should have exactly two entries"
    return "--".join(model_pair)


def to_model_pair(pair_descriptor: str) -> List[str]:
    """Converts a model pair descriptor into a model pair list.
    Args:
        pair_descriptor: A string with two model IDs/names connected by '--'.
    Returns:
        The model pair list for the model pair.
    """
    model_pair = pair_descriptor.split("--")
    assert len(model_pair) == 2, "Model pair should have exactly two entries"
    return model_pair


def is_pair_descriptor(text: str) -> bool:
    """Checks if a string is a model pair descriptor.
    Actually just checks if the passed string contains '--' and nothing further.
    Args:
        text: The string to check.
    Returns:
        True, if the string contains the model pair descriptor-defining '--', False otherwise.
    """
    return "--" in text

# ---
