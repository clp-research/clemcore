import string
from typing import List


def remove_punctuation(text: str) -> str:
    """Remove punctuation from a string.
    Args:
        text: The string to remove punctuation from.
    Returns:
        The passed string without punctuation.
    """
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text
