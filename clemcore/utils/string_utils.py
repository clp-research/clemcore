import json
import string
from typing import Any, Optional


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


def str_to_bool(s: str):
    if s.lower() in ("true", "yes", "on", "1"): return True
    if s.lower() in ("false", "no", "off", "0"): return False
    raise ValueError


def try_convert(value: str, data_types: tuple) -> Any:
    for type_constructor in data_types:
        try:
            return type_constructor(value)
        except ValueError:
            continue
    return value


def read_query_string(query_string: Optional[str]) -> Optional[dict[str, Any]]:
    if query_string is None:
        return None
    if not query_string:
        return {}

    kv_pairs = query_string.split(",")
    kv_dict = {}
    for kv_pair in kv_pairs:
        if "=" not in kv_pair:
            raise ValueError(f"Invalid query string pair: {kv_pair}")
        k, v = kv_pair.split("=", 1)
        k, v = k.strip(), v.strip()
        kv_dict[k] = try_convert(v, (str_to_bool, int, float))
    return kv_dict
