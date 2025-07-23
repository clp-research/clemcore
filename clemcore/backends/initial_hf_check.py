"""Initial checks on models to be added to the HuggingFace local backend."""

import argparse
from transformers import AutoTokenizer
from jinja2.exceptions import TemplateError
import copy


def preprocess_messages(messages) -> list:
    """Preprocess messages as the HuggingFace local backend does.
    Removes empty system message and flattens consecutive same-role messages.
    Args:
        messages: A message history. For example:
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
    Returns:
        A preprocessed message history list.
    """
    # deepcopy messages to prevent reference issues:
    current_messages = copy.deepcopy(messages)

    # cull empty system message:
    if current_messages[0]['role'] == "system":
        if not current_messages[0]['content']:
            del current_messages[0]

    # flatten consecutive user messages:
    for msg_idx, message in enumerate(current_messages):
        if msg_idx > 0 and message['role'] == "user" and current_messages[msg_idx - 1]['role'] == "user":
            current_messages[msg_idx - 1]['content'] += f" {message['content']}"
            del current_messages[msg_idx]
        elif msg_idx > 0 and message['role'] == "assistant" and current_messages[msg_idx - 1]['role'] == "assistant":
            current_messages[msg_idx - 1]['content'] += f" {message['content']}"
            del current_messages[msg_idx]

    return current_messages


def model_pre_check(args):
    """Checks various model data to determine if a HuggingFace model is usable with clembench.
    Results are printed to terminal.
    Args:
        args: Arguments as parsed with argparse. This contains at least the model HF model ID, optionally if tokenizer
            and chat template information is to be printed.
    """
    # load tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(args.name, verbose=False)

    # tokenizer info:
    if args.tokenizer_info:
        print("Tokenizer info:")
        print(tokenizer)
        print()

    # template return:
    if args.show_template:
        print("Chat template:")
        print(tokenizer.chat_template)
        print()

    model_passes = True

    # proper minimal messages:
    proper_messages = [
        {"role": "user", "content": "What is your favourite condiment?"},
        {"role": "assistant", "content": "Lard!"},
        {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]

    try:
        result_tokens = tokenizer.apply_chat_template(proper_messages, add_generation_prompt=True, return_tensors="pt")
        result = tokenizer.batch_decode(result_tokens)[0]
        print("Applied chat template:")
        print(result)
        print()
    except TemplateError:
        print("Chat template application on minimal proper messages list failed! "
              "Something is wrong with the tokenizer or its config!")
        model_passes = False

    # improper double user messages:
    double_role_messages = [
        {"role": "user", "content": "Hello there!"},
        {"role": "user", "content": "What is your favourite condiment?"},
        {"role": "assistant", "content": "Lard!"},
        {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]

    current_messages = preprocess_messages(double_role_messages)
    try:
        result_tokens = tokenizer.apply_chat_template(current_messages, add_generation_prompt=True, return_tensors="pt")
    except TemplateError:
        print("Chat template application on flattened messages list failed! "
              "Something is wrong with the tokenizer or its config!")
        model_passes = False

    # improper first assistant message:
    first_assistant_messages = [
        {"role": "assistant", "content": "Hello there!"},
        {"role": "user", "content": "What is your favourite condiment?"},
        {"role": "assistant", "content": "Lard!"},
        {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]

    current_messages = preprocess_messages(first_assistant_messages)
    try:
        result_tokens = tokenizer.apply_chat_template(current_messages, add_generation_prompt=True, return_tensors="pt")
    except TemplateError:
        print("Chat template application on messages list with assistant first failed! "
              "This means this model's tokenizer does not accept first assistant messages. "
              "No current clemgame uses this, but be warned.")

    # system message:
    system_messages = [
        {"role": "system", "content": "You love all kinds of fat."},
        {"role": "user", "content": "What is your favourite condiment?"},
        {"role": "assistant", "content": "Lard!"},
        {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]

    current_messages = preprocess_messages(system_messages)
    try:
        result_tokens = tokenizer.apply_chat_template(current_messages, add_generation_prompt=True, return_tensors="pt")
    except TemplateError:
        print("Chat template application on messages list with system message failed! "
              "This means this model's tokenizer does not accept system messages. "
              "No current clemgame uses system message content, but be warned.")

    print()
    if model_passes:
        print("The model passes the preliminary chat template checks!")
    else:
        print("The model does NOT pass the preliminary chat template checks!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model_name", type=str,
                        help="The full HuggingFace model ID/link. "
                             "Example: openchat/openchat_3.5")

    parser.add_argument("-i", "--tokenizer_info", type=str,
                        help="Optional argument to show tokenizer information.")

    parser.add_argument("-t", "--show_template", type=str,
                        help="Optional argument to show tokenizer chat template as jinja string.")

    args = parser.parse_args()
    model_pre_check(args)
