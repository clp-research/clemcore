"""
    Utility functions for clembench backends.
"""

from typing import List, Dict, Tuple, Any, Union


def check_context_limit_generic(context_size, prompt_tokens, max_new_tokens: int = 100) -> Tuple[bool, int, int, int]:
    """
    Internal context limit check to run in generate_response.
    :param prompt_tokens: List of prompt token IDs.
    :param max_new_tokens: How many tokens to generate ('at most', but no stop sequence is defined).
    :return: Tuple with
            Bool: True if context limit is not exceeded, False if too many tokens
            Number of tokens for the given messages and maximum new tokens
            Number of tokens of 'context space left'
            Total context token limit
    """
    prompt_size = len(prompt_tokens)
    tokens_used = prompt_size + max_new_tokens  # context includes tokens to be generated
    tokens_left = context_size - tokens_used
    fits = tokens_used <= context_size
    return fits, tokens_used, tokens_left, context_size