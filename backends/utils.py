import copy
from functools import wraps
from typing import List, Dict

from backends import get_logger

logger = get_logger(__name__)


def ensure_alternating_roles(messages: List[Dict], cull_system_message: bool = True) -> List[Dict]:
    """
    The messages format assumes alternating roles of user and assistant. This method checks, if this constraint
    is satisfied. If this is not the case and there are consecutive user or assistant messages,
    then these are merged into a single one.

    :param messages: to be checked
    :return: a new messages object with the alternating roles ensured
    """
    _messages = copy.deepcopy(messages)

    if cull_system_message:
        if _messages[0]['role'] == "system" and not _messages[0]["content"]:
            del _messages[0]

    def is_same_role(msg1, msg2):
        return msg1["role"] == msg2["role"]

    delimiter = "\n\n"

    def join_content(msg1, msg2):
        return f"{msg1['content']}{delimiter}{msg2['content']}"

    # combine consecutive user messages:
    for msg_idx, message in enumerate(_messages):
        if msg_idx == 0:
            continue
        prev_message = _messages[msg_idx - 1]
        if is_same_role(prev_message, message):
            warn_msg = (f"Found consecutive role assignments. These will be merged into one:\n"
                        f"{prev_message}\n"
                        f"{message}")
            logger.warning(warn_msg)
            prev_message['content'] = join_content(prev_message, message)
            del _messages[msg_idx]

    return _messages


def ensure_messages_format(generate_response_fn):
    @wraps(generate_response_fn)
    def wrapped_fn(self, messages):
        _messages = ensure_alternating_roles(messages)
        return generate_response_fn(self, _messages)

    return wrapped_fn
