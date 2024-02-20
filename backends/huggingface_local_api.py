"""
    Backend using HuggingFace transformers & ungated models.
    Uses HF tokenizers instruct/chat templates for proper input format per model.
"""
from typing import List, Dict, Tuple, Any, Union
import torch
import backends

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import copy

from jinja2 import TemplateError

logger = backends.get_logger(__name__)

FALLBACK_CONTEXT_SIZE = 256


def load_config_and_tokenizer(model_spec: backends.ModelSpec) -> Union[AutoTokenizer, AutoConfig, int]:
    logger.info(f'Loading huggingface model config and tokenizer: {model_spec.model_name}')

    use_api_key = False
    api_key = None
    if 'requires_api_key' in model_spec:
        if model_spec['requires_api_key']:
            # load HF API key:
            creds = backends.load_credentials("huggingface")
            api_key = creds["huggingface"]["api_key"]
            use_api_key = True
        else:
            requires_api_key_info = (f"{model_spec['model_name']} registry setting has requires_api_key, "
                                     f"but it is not 'true'. Please check the model entry.")
            print(requires_api_key_info)
            logger.info(requires_api_key_info)

    hf_model_str = model_spec['huggingface_id']

    # use 'slow' tokenizer for models that require it:
    if 'slow_tokenizer' in model_spec:
        if model_spec['slow_tokenizer']:
            tokenizer = AutoTokenizer.from_pretrained(hf_model_str, device_map="auto", torch_dtype="auto",
                                                      verbose=False, use_fast=False)
        else:
            tokenizer = None
            slow_tokenizer_info = (f"{model_spec['model_name']} registry setting has slow_tokenizer, "
                                   f"but it is not 'true'. Please check the model entry.")
            print(slow_tokenizer_info)
            logger.info(slow_tokenizer_info)
    elif use_api_key:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_str, token=api_key, device_map="auto",
                                                  torch_dtype="auto", verbose=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_str, device_map="auto", torch_dtype="auto",
                                                  verbose=False)

    # apply proper chat template:
    if not model_spec['premade_chat_template']:
        if 'custom_chat_template' in model_spec:
            tokenizer.chat_template = model_spec['custom_chat_template']
        else:
            logger.info(
                f"No custom chat template for {model_spec.model_name} found in model settings from model registry "
                f"while model has no pre-made template! Generic template will be used, likely leading to "
                f"bad results.")

    if use_api_key:
        model_config = AutoConfig.from_pretrained(hf_model_str, token=api_key)
    else:
        model_config = AutoConfig.from_pretrained(hf_model_str)

    # get context token limit for model:
    if hasattr(model_config, 'max_position_embeddings'):  # this is the standard attribute used by most
        context_size = model_config.max_position_embeddings
    elif hasattr(model_config, 'n_positions'):  # some models may have their context size under this attribute
        context_size = model_config.n_positions
    else:  # few models, especially older ones, might not have their context size in the config
        context_size = FALLBACK_CONTEXT_SIZE

    return tokenizer, model_config, context_size


def load_model(model_spec: backends.ModelSpec) -> Any:
    logger.info(f'Start loading huggingface model weights: {model_spec.model_name}')

    hf_model_str = model_spec['huggingface_id']
    if 'requires_api_key' in model_spec and model_spec['requires_api_key']:
        # load HF API key:
        creds = backends.load_credentials("huggingface")
        api_key = creds["huggingface"]["api_key"]
        # load model using its default configuration:
        model = AutoModelForCausalLM.from_pretrained(hf_model_str, token=api_key, device_map="auto", torch_dtype="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(hf_model_str, device_map="auto", torch_dtype="auto")

    logger.info(f"Finished loading huggingface model: {model_spec.model_name}")
    logger.info(f"Model device map: {model.hf_device_map}")
    return model


class HuggingfaceLocal(backends.Backend):
    def __init__(self):
        super().__init__()

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        return HuggingfaceLocalModel(model_spec)


class HuggingfaceLocalModel(backends.Model):

    def __init__(self, model_spec: backends.ModelSpec):
        super().__init__(model_spec)
        # fail-fast
        self.tokenizer, self.config, self.context_size = load_config_and_tokenizer(model_spec)
        self.model = load_model(model_spec)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate_response(self, messages: List[Dict],
                          return_full_text: bool = False,
                          log_messages: bool = False) -> Tuple[Any, Any, str]:
        """
        :param messages: for example
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        :param model: model name
        :param return_full_text: If True, whole input context is returned.
        :param log_messages: If True, raw and cleaned messages passed will be logged.
        :return: the continuation
        """
        # log current given messages list:
        if log_messages:
            logger.info(f"Raw messages passed: {messages}")

        current_messages = _clean_messages(messages)

        # log current flattened messages list:
        if log_messages:
            logger.info(f"Flattened messages: {current_messages}")

        # apply chat template & tokenize:
        prompt_tokens = self.tokenizer.apply_chat_template(current_messages, add_generation_prompt=True,
                                                           return_tensors="pt")
        prompt_tokens = prompt_tokens.to(self.device)

        prompt_text = self.tokenizer.batch_decode(prompt_tokens)[0]
        prompt = {"inputs": prompt_text, "max_new_tokens": self.get_max_tokens(),
                  "temperature": self.get_temperature(), "return_full_text": return_full_text}

        # check context limit:
        context_check = _check_context_limit(self.context_size, prompt_tokens[0],
                                             max_new_tokens=self.get_max_tokens())
        if not context_check[0]:  # if context is exceeded, context_check[0] is False
            logger.info(f"Context token limit for {self.model_spec.model_name} exceeded: "
                        f"{context_check[1]}/{context_check[3]}")
            # fail gracefully:
            raise backends.ContextExceededError(f"Context token limit for {self.model_spec.model_name} exceeded",
                                                tokens_used=context_check[1], tokens_left=context_check[2],
                                                context_size=context_check[3])

        # greedy decoding:
        do_sample: bool = False
        if self.get_temperature() > 0.0:
            do_sample = True

        if do_sample:
            model_output_ids = self.model.generate(
                prompt_tokens,
                temperature=self.get_temperature(),
                max_new_tokens=self.get_max_tokens(),
                do_sample=do_sample
            )
        else:
            model_output_ids = self.model.generate(
                prompt_tokens,
                max_new_tokens=self.get_max_tokens(),
                do_sample=do_sample
            )

        model_output = self.tokenizer.batch_decode(model_output_ids)[0]

        response = {'response': model_output}

        # cull input context; equivalent to transformers.pipeline method:
        if not return_full_text:
            response_text = model_output.replace(prompt_text, '').strip()

            if 'output_split_prefix' in self.model_spec:
                response_text = model_output.rsplit(self.model_spec['output_split_prefix'], maxsplit=1)[1]

            eos_len = len(self.model_spec['eos_to_cull'])

            if response_text.endswith(self.model_spec['eos_to_cull']):
                response_text = response_text[:-eos_len]

        else:
            response_text = model_output.strip()

        return prompt, response, response_text


def _check_context_limit(context_size, prompt_tokens, max_new_tokens: int = 100) -> Tuple[bool, int, int, int]:
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


def _clean_messages(messages: List[Dict]) -> List[Dict]:
    """
    Remove message issues indiscriminately for compatibility with certain model's chat templates. Empty first system
    message is removed (for Mistral models and others that do not use system messages). Messages are concatenated
    to create consistent user-assistant pairs (for Llama-based chat formats).
    :param messages: for example
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
                {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                {"role": "user", "content": "Where was it played?"}
            ]
    :return: Cleaned and flattened messages list.
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
        elif msg_idx > 0 and message['role'] == "assistant" and current_messages[msg_idx - 1][
            'role'] == "assistant":
            current_messages[msg_idx - 1]['content'] += f" {message['content']}"
            del current_messages[msg_idx]

    return current_messages


def check_messages(messages: List[Dict], model_spec: backends.ModelSpec) -> bool:
    """
    Message checking for clemgame development. This checks if the model's chat template accepts the given messages
    as passed, before the standard flattening done for generation. This allows clemgame developers to construct
    message lists that are sound as-is and are not affected by the indiscriminate flattening of the generation
    method. Deliberately verbose.
    :param model_spec: model spec
    :param messages: for example
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
                {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                {"role": "user", "content": "Where was it played?"}
            ]
    :return: True if messages are sound as-is, False if messages are not compatible with the model's template.
    """
    tokenizer, _, _ = load_config_and_tokenizer(model_spec)

    # bool for message acceptance:
    messages_accepted: bool = True

    # check for system message:
    has_system_message: bool = False
    if messages[0]['role'] == "system":
        print("System message detected.")
        has_system_message = True
        if not messages[0]['content']:
            print(f"Initial system message is empty. It will be removed when generating responses.")
        else:
            print(f"Initial system message has content! It will not be removed when generating responses. This "
                  f"will lead to issues with models that do not allow system messages.")
        """
        print("Checking model system message compatibility...")
        # unfortunately Mistral models, which do not accept system message, currently do not raise a distinct 
        # exception for this...
        try:
            self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        except TemplateError:
            print("The model's chat template does not allow for system message!")
            messages_accepted = False
        """

    # check for message order:
    starts_with_assistant: bool = False
    double_user: bool = False
    double_assistant: bool = False
    ends_with_assistant: bool = False

    for msg_idx, message in enumerate(messages):
        if not has_system_message:
            if msg_idx == 0 and message['role'] == "assistant":
                starts_with_assistant = True
        else:
            if msg_idx == 1 and message['role'] == "assistant":
                starts_with_assistant = True
        if msg_idx > 0 and message['role'] == "user" and messages[msg_idx - 1]['role'] == "user":
            double_user = True
        elif msg_idx > 0 and message['role'] == "assistant" and messages[msg_idx - 1]['role'] == "assistant":
            double_assistant = True
    if messages[-1]['role'] == "assistant":
        ends_with_assistant = True

    if starts_with_assistant or double_user or double_assistant or ends_with_assistant:
        print("Message order issue(s) found:")
        if starts_with_assistant:
            print("First message has role:'assistant'.")
        if double_user:
            print("Messages contain consecutive user messages.")
        if double_assistant:
            print("Messages contain consecutive assistant messages.")
        if ends_with_assistant:
            print("Last message has role:'assistant'.")

    # proper check of chat template application:
    try:
        tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    except TemplateError:
        print(f"The {model_spec.model_name} chat template does not accept these messages! "
              f"Cleaning applied before generation might still allow these messages, but is indiscriminate and "
              f"might lead to unintended generation inputs.")
        messages_accepted = False
    else:
        print(
            f"The {model_spec.model_name} chat template accepts these messages. Cleaning before generation is still "
            f"applied to these messages, which is indiscriminate and might lead to unintended generation inputs.")

    return messages_accepted


def check_context_limit(messages: List[Dict], model_spec: backends.ModelSpec,
                        max_new_tokens: int = 100, clean_messages: bool = False,
                        verbose: bool = True) -> Tuple[bool, int, int, int]:
    """
    Externally-callable context limit check for clemgame development.
    :param messages: for example
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
                {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                {"role": "user", "content": "Where was it played?"}
            ]
    :param model: model name
    :param max_new_tokens: How many tokens to generate ('at most', but no stop sequence is defined).
    :param clean_messages: If True, the standard cleaning method for message lists will be applied.
    :param verbose: If True, prettyprint token counts.
    :return: Tuple with
            Bool: True if context limit is not exceeded, False if too many tokens
            Number of tokens for the given messages and maximum new tokens
            Number of tokens of 'context space left'
            Total context token limit
    """
    tokenizer, _, context_size = load_config_and_tokenizer(model_spec)

    # optional messages processing:
    if clean_messages:
        current_messages = _clean_messages(messages)
    else:
        current_messages = messages
    # the actual tokens, including chat format:
    prompt_tokens = tokenizer.apply_chat_template(current_messages, add_generation_prompt=True)
    context_check_tuple = _check_context_limit(context_size, prompt_tokens, max_new_tokens=max_new_tokens)
    tokens_used = context_check_tuple[1]
    tokens_left = context_check_tuple[2]
    if verbose:
        print(f"{tokens_used} input tokens, {tokens_left} tokens of {context_size} left.")
    fits = context_check_tuple[0]
    return fits, tokens_used, tokens_left, context_size
