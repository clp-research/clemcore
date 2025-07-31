import logging
import copy
from datetime import datetime
from functools import wraps
from typing import List, Dict, Tuple

from clemcore.backends import ContextExceededError

logger = logging.getLogger(__name__)


def ensure_alternating_roles(messages: List[Dict], cull_system_message: bool = True) -> List[Dict]:
    """Ensure alternating chat roles by concatenating consecutive same-role messages.
    The messages format assumes alternating roles of user and assistant. This method checks if this constraint is
    satisfied. If this is not the case and there are consecutive user or assistant messages, then these are merged into
    a single message.
    Args:
        messages: List of message dicts to be checked.
        cull_system_message: Determines if empty system message(s) are removed. This assures compatibility with models
            that do not support system messages. Default: True
    Returns:
        A new messages list with alternating message roles.
    """
    _messages = copy.deepcopy(messages)

    if cull_system_message:
        if _messages[0]['role'] == "system" and not _messages[0]["content"]:
            del _messages[0]

    def is_same_role(msg1, msg2):
        """Check if two messages have the same role.
        Args:
            msg1: The first message to be checked.
            msg2: The second message to be checked.
        Returns:
            True if both messages have the same role, False otherwise.
        """
        return msg1["role"] == msg2["role"]

    delimiter = "\n\n"

    def join_content(msg1, msg2):
        """Join the content of two messages.
        Args:
            msg1: The first message to be checked.
            msg2: The second message to be checked.
        Returns:
            The passed messages, joined with a '\n\n' delimiter.
        """
        return f"{msg1['content']}{delimiter}{msg2['content']}"

    if len(_messages) <= 1:
        return _messages

    def is_valid(idx):
        """Check if a message index is valid.
        Args:
            idx: The message index.
        Returns:
            True if the message index is inside the bounds of the message list.
        """
        return idx < len(_messages)

    msg_idx = 1
    while is_valid(msg_idx):
        prev_message = _messages[msg_idx - 1]
        message = _messages[msg_idx]
        if is_same_role(prev_message, message):
            msg = (f"Found consecutive role assignments. These will be merged into one:\n"
                        f"{prev_message}\n"
                        f"{message}")
            logger.debug(msg)
            prev_message['content'] = join_content(prev_message, message)
            del _messages[msg_idx]
        else:
            msg_idx += 1

    return _messages


def ensure_messages_format(generate_response_fn):
    """
    Decorator to ensure messages have properly alternating roles before calling
    the backend's generate_response or generate_batch_response method.

    This wrapper validates and adjusts the `messages` argument to enforce
    alternating roles (e.g., user, assistant, user, assistant) for each message
    or each list of messages in a batch.

    It supports:
      - Single-response methods: expects `messages` as a list of dicts.
      - Batch-response methods: expects `messages` as a list of lists of dicts.

    The decorator automatically detects whether the input is batch or single
    and applies role checks accordingly.

    Note:
        If used with `augment_response_object`, apply this decorator *before*
        that one, i.e., put `@ensure_messages_format` below
        `@augment_response_object` to ensure formatting happens first.

    Args:
        generate_response_fn (callable): The original generate_response or
            generate_batch_response method of a backend class.

    Returns:
        callable: A wrapped version of the method that ensures input messages
        have alternating roles before invoking the original method.
    """

    @wraps(generate_response_fn)
    def wrapped_fn(self, messages, *args, **kwargs):
        if isinstance(messages, list) and all(isinstance(m, list) for m in messages):
            # Batch mode: apply to each list of messages
            _messages = [ensure_alternating_roles(message) for message in messages]
        else:  # Single mode: apply directly
            _messages = ensure_alternating_roles(messages)

        return generate_response_fn(self, _messages, *args, **kwargs)

    return wrapped_fn


def augment_response_object(generate_response_fn):
    """
    Decorator to augment the response object(s) with `clem_player` metadata.

    This wrapper works with both single-response methods (returning a tuple)
    and batch-response methods (returning a list of tuples). It adds metadata
    about the call start time, call duration, response text, and model name
    to the `response_object` dictionary inside the returned tuple(s).

    Note:
        If you are using this decorator together with `ensure_messages_format`,
        apply this decorator *after* that one, i.e., put
        `@augment_response_object` above `@ensure_messages_format`.

    Args:
        generate_response_fn (callable): The original generate_response or
            generate_batch_response method of a backend class.

    Returns:
        callable: A wrapped version of the method that adds `clem_player`
        metadata to the `response_object`. Returns either a single tuple
        or a list of tuples, matching the original method's return type.
    """

    @wraps(generate_response_fn)
    def wrapped_fn(self, messages, *args, **kwargs):
        call_start = datetime.now()
        result = generate_response_fn(self, messages, *args, **kwargs)
        call_duration = datetime.now() - call_start

        def add_clem_player_metadata(t):
            prompt, response_object, response_text = t
            response_object["clem_player"] = {
                "call_start": str(call_start),
                "call_duration": str(call_duration),
                "response": response_text,
                "model_name": self.name,
            }
            return prompt, response_object, response_text

        if isinstance(result, list):  # batch mode - update each tuple in the list
            return [add_clem_player_metadata(t) for t in result]
        return add_clem_player_metadata(result)

    return wrapped_fn


def check_context_limit_generic(context_size: int, prompt_tokens: List, model_name: str, max_new_tokens: int = 100) \
        -> Tuple[bool, int, int, int]:
    """Internal context limit check to run in generate_response.
    Used to assure that the context limit of a model is not exceeding during benchmark runs. Allows to fail gracefully
    in case the context limit is exceeded, assuring proper record keeping.
    The potentially raised ContextExceedError is intended to be caught by game code to modify input message histories
    without impacting the game experiment.
    Args:
        context_size: The context size limit of the model.
        prompt_tokens: List of prompt token IDs.
        model_name: Name of the model checked for.
        max_new_tokens: How many tokens to generate ('at most', but no stop sequence is defined).
    Returns:
        Tuple with

        - Bool: True if context limit is not exceeded, False if too many tokens
        - Number of tokens for the given messages and maximum new tokens
        - Number of tokens of 'context space left'
        - Total context token limit
    Raises:
        ContextExceededError: A ContextExceededError exception containing context usage information is raised if the
            context limit is exceeded.
    """
    prompt_size = len(prompt_tokens)
    tokens_used = prompt_size + max_new_tokens  # context includes tokens to be generated
    tokens_left = context_size - tokens_used
    fits = tokens_used <= context_size

    if not fits:
        logger.info(f"Context token limit for {model_name} exceeded: {tokens_used}/{tokens_left}")
        # fail gracefully:
        raise ContextExceededError(f"Context token limit for {model_name} exceeded",
                                   tokens_used=tokens_used, tokens_left=tokens_left, context_size=context_size)

    return fits, tokens_used, tokens_left, context_size
