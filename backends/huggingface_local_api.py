""" Backend using HuggingFace transformers & ungated models. Uses HF tokenizers instruct/chat templates for proper input format per model. """
from typing import List, Dict, Tuple, Any
import torch
import backends

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os
import copy
import json

from jinja2 import TemplateError

# load model registry:
model_registry_path = os.path.join(backends.project_root, "backends", "hf_local_models.json")
with open(model_registry_path, 'r', encoding="utf-8") as model_registry_file:
    MODEL_REGISTRY = json.load(model_registry_file)

logger = backends.get_logger(__name__)

NAME = "huggingface"

SUPPORTED_MODELS = [model_setting['model_name'] for model_setting in MODEL_REGISTRY if model_setting['backend'] == NAME]

FALLBACK_CONTEXT_SIZE = 256


class HuggingfaceLocal(backends.Backend):
    def __init__(self):
        self.temperature: float = -1.
        self.use_api_key: bool = False
        self.config_and_tokenizer_loaded: bool = False
        self.model_loaded: bool = False
        self.model_name: str = ""

    def load_config_and_tokenizer(self, model_name):
        logger.info(f'Loading huggingface model config and tokenizer: {model_name}')
        # model cache handling
        root_data_path = os.path.join(os.path.abspath(os.sep), "data")
        # check if root/data exists:
        if not os.path.isdir(root_data_path):
            logger.info(f"{root_data_path} does not exist, creating directory.")
            # create root/data:
            os.mkdir(root_data_path)
        self.CACHE_DIR = os.path.join(root_data_path, "huggingface_cache")

        # get settings from model registry for the first name match that uses this backend:
        for model_setting in MODEL_REGISTRY:
            if model_setting['model_name'] == model_name:
                if model_setting['backend'] == "huggingface":
                    self.model_settings = model_setting
                    break

        assert self.model_settings['model_name'] == model_name, (f"Model settings for {model_name} not properly loaded "
                                                                 f"from model registry!")

        if 'requires_api_key' in self.model_settings:
            if self.model_settings['requires_api_key']:
                # load HF API key:
                creds = backends.load_credentials("huggingface")
                self.api_key = creds["huggingface"]["api_key"]
                self.use_api_key = True
            else:
                requires_api_key_info = (f"{self.model_settings['model_name']} registry setting has requires_api_key, "
                                         f"but it is not 'true'. Please check the model entry.")
                print(requires_api_key_info)
                logger.info(requires_api_key_info)

        hf_model_str = self.model_settings['huggingface_id']

        # use 'slow' tokenizer for models that require it:
        if 'slow_tokenizer' in self.model_settings:
            if self.model_settings['slow_tokenizer']:
                self.tokenizer = AutoTokenizer.from_pretrained(hf_model_str, device_map="auto", torch_dtype="auto",
                                                               cache_dir=self.CACHE_DIR, verbose=False, use_fast=False)
            else:
                slow_tokenizer_info = (f"{self.model_settings['model_name']} registry setting has slow_tokenizer, "
                                       f"but it is not 'true'. Please check the model entry.")
                print(slow_tokenizer_info)
                logger.info(slow_tokenizer_info)
        elif self.use_api_key:
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_str, token=self.api_key, device_map="auto",
                                                           torch_dtype="auto", cache_dir=self.CACHE_DIR, verbose=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_str, device_map="auto", torch_dtype="auto",
                                                           cache_dir=self.CACHE_DIR, verbose=False)

        # apply proper chat template:
        if not self.model_settings['premade_chat_template']:
            if 'custom_chat_template' in self.model_settings:
                self.tokenizer.chat_template = self.model_settings['custom_chat_template']
            else:
                logger.info(f"No custom chat template for {model_name} found in model settings from model registry "
                            f"while model has no pre-made template! Generic template will be used, likely leading to "
                            f"bad results.")

        if self.use_api_key:
            model_config = AutoConfig.from_pretrained(hf_model_str, token=self.api_key)
        else:
            model_config = AutoConfig.from_pretrained(hf_model_str)

        # get context token limit for model:
        if hasattr(model_config, 'max_position_embeddings'):  # this is the standard attribute used by most
            self.context_size = model_config.max_position_embeddings
        elif hasattr(model_config, 'n_positions'):  # some models may have their context size under this attribute
            self.context_size = model_config.n_positions
        else:  # few models, especially older ones, might not have their context size in the config
            self.context_size = FALLBACK_CONTEXT_SIZE

        self.model_name = model_name
        self.config_and_tokenizer_loaded = True

    def load_model(self, model_name):
        # different model name might be passed:
        if not model_name == self.model_name:
            self.config_and_tokenizer_loaded = False

        if not self.config_and_tokenizer_loaded:
            self.load_config_and_tokenizer(model_name)

        logger.info(f'Start loading huggingface model weights: {model_name}')

        hf_model_str = self.model_settings['huggingface_id']

        # load model using its default configuration:
        if self.use_api_key:
            self.model = AutoModelForCausalLM.from_pretrained(hf_model_str, token=self.api_key, device_map="auto",
                                                              torch_dtype="auto", cache_dir=self.CACHE_DIR)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(hf_model_str, device_map="auto", torch_dtype="auto",
                                                              cache_dir=self.CACHE_DIR)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = True

    def _clean_messages(self, messages: List[Dict]) -> List[Dict]:
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

    def check_messages(self, messages: List[Dict], model: str) -> bool:
        """
        Message checking for clemgame development. This checks if the model's chat template accepts the given messages
        as passed, before the standard flattening done for generation. This allows clemgame developers to construct
        message lists that are sound as-is and are not affected by the indiscriminate flattening of the generation
        method. Deliberately verbose.
        :param messages: for example
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        :param model: model name
        :return: True if messages are sound as-is, False if messages are not compatible with the model's template.
        """
        if not model == self.model_name:
            self.config_and_tokenizer_loaded = False

        if not self.config_and_tokenizer_loaded:
            self.load_config_and_tokenizer(model)

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
            self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        except TemplateError:
            print(f"The {self.model_name} chat template does not accept these messages! "
                  f"Cleaning applied before generation might still allow these messages, but is indiscriminate and "
                  f"might lead to unintended generation inputs.")
            messages_accepted = False
        else:
            print(f"The {self.model_name} chat template accepts these messages. Cleaning before generation is still "
                  f"applied to these messages, which is indiscriminate and might lead to unintended generation inputs.")

        return messages_accepted

    def _check_context_limit(self, prompt_tokens, max_new_tokens: int = 100) -> Tuple[bool, int, int, int]:
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
        tokens_left = self.context_size - tokens_used
        fits = tokens_used <= self.context_size
        return fits, tokens_used, tokens_left, self.context_size

    def check_context_limit(self, messages: List[Dict], model: str,
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
        # different model name might be passed:
        if not model == self.model_name:
            self.config_and_tokenizer_loaded = False

        if not self.config_and_tokenizer_loaded:
            self.load_config_and_tokenizer(model)
        # optional messages processing:
        if clean_messages:
            current_messages = self._clean_messages(messages)
        else:
            current_messages = messages
        # the actual tokens, including chat format:
        prompt_tokens = self.tokenizer.apply_chat_template(current_messages, add_generation_prompt=True)
        context_check_tuple = self._check_context_limit(prompt_tokens, max_new_tokens=max_new_tokens)
        tokens_used = context_check_tuple[1]
        tokens_left = context_check_tuple[2]
        if verbose:
            print(f"{tokens_used} input tokens, {tokens_left} tokens of {self.context_size} left.")
        fits = context_check_tuple[0]
        return fits, tokens_used, tokens_left, self.context_size

    def generate_response(self, messages: List[Dict], model: str,
                          max_new_tokens: int = 100, return_full_text: bool = False,
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
        :param max_new_tokens: How many tokens to generate ('at most', but no stop sequence is defined).
        :param return_full_text: If True, whole input context is returned.
        :param log_messages: If True, raw and cleaned messages passed will be logged.
        :return: the continuation
        """
        assert 0.0 <= self.temperature <= 1.0, "Temperature must be in [0.,1.]"
        # different model name might be passed:
        if not model == self.model_name:
            self.model_loaded = False

        # load the model to the memory
        if not self.model_loaded:
            self.load_model(model)
            logger.info(f"Finished loading huggingface model: {model}")
            logger.info(f"Model device map: {self.model.hf_device_map}")

        # log current given messages list:
        if log_messages:
            logger.info(f"Raw messages passed: {messages}")

        current_messages = self._clean_messages(messages)

        # log current flattened messages list:
        if log_messages:
            logger.info(f"Flattened messages: {current_messages}")

        # apply chat template & tokenize:
        prompt_tokens = self.tokenizer.apply_chat_template(current_messages, add_generation_prompt=True,
                                                           return_tensors="pt")
        prompt_tokens = prompt_tokens.to(self.device)

        prompt_text = self.tokenizer.batch_decode(prompt_tokens)[0]
        prompt = {"inputs": prompt_text, "max_new_tokens": max_new_tokens,
                  "temperature": self.temperature, "return_full_text": return_full_text}

        # check context limit:
        context_check = self._check_context_limit(prompt_tokens[0], max_new_tokens=max_new_tokens)
        if not context_check[0]:  # if context is exceeded, context_check[0] is False
            logger.info(f"Context token limit for {self.model_name} exceeded: {context_check[1]}/{context_check[3]}")
            # fail gracefully:
            raise backends.ContextExceededError(f"Context token limit for {self.model_name} exceeded",
                                                tokens_used=context_check[1], tokens_left=context_check[2],
                                                context_size=context_check[3])

        # greedy decoding:
        do_sample: bool = False
        if self.temperature > 0.0:
            do_sample = True

        if do_sample:
            model_output_ids = self.model.generate(
                prompt_tokens,
                temperature=self.temperature,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample
            )
        else:
            model_output_ids = self.model.generate(
                prompt_tokens,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample
            )

        model_output = self.tokenizer.batch_decode(model_output_ids)[0]

        response = {'response': model_output}

        # cull input context; equivalent to transformers.pipeline method:
        if not return_full_text:
            response_text = model_output.replace(prompt_text, '').strip()

            if 'output_split_prefix' in self.model_settings:
                response_text = model_output.rsplit(self.model_settings['output_split_prefix'], maxsplit=1)[1]

            eos_len = len(self.model_settings['eos_to_cull'])

            if response_text.endswith(self.model_settings['eos_to_cull']):
                response_text = response_text[:-eos_len]

        else:
            response_text = model_output.strip()

        return prompt, response, response_text

    def supports(self, model_name: str):
        return model_name in SUPPORTED_MODELS


if __name__ == "__main__":
    # initialize a backend instance:
    test_backend = HuggingfaceLocal()

    # MESSAGES CHECKING
    print("--- Messages checking examples ---")
    # proper minimal messages:
    minimal_messages = [
        {"role": "user", "content": "What is your favourite condiment?"},
        {"role": "assistant", "content": "Lard!"},
        {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]
    # check proper minimal messages with Mistral-7B-Instruct-v0.1:
    print("Minimal messages:")
    test_backend.check_messages(minimal_messages, "Mistral-7B-Instruct-v0.1")
    print()

    # improper double user messages:
    double_user_messages = [
        {"role": "user", "content": "Hello there!"},
        {"role": "user", "content": "What is your favourite condiment?"},
        {"role": "assistant", "content": "Lard!"},
        {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]
    # check improper double user messages with Mistral-7B-Instruct-v0.1:
    print("Double user messages:")
    test_backend.check_messages(double_user_messages, "Mistral-7B-Instruct-v0.1")
    print()

    # improper first assistant message:
    first_assistant_messages = [
        {"role": "assistant", "content": "Hello there!"},
        {"role": "user", "content": "What is your favourite condiment?"},
        {"role": "assistant", "content": "Lard!"},
        {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]
    # check improper first assistant message with Mistral-7B-Instruct-v0.1:
    print("First message role assistant:")
    test_backend.check_messages(first_assistant_messages, "Mistral-7B-Instruct-v0.1")
    print()

    # system message:
    system_messages = [
        {"role": "system", "content": "You love all kinds of fat."},
        {"role": "user", "content": "What is your favourite condiment?"},
        {"role": "assistant", "content": "Lard!"},
        {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]
    # check system message with Mistral-7B-Instruct-v0.1:
    print("System message:")
    test_backend.check_messages(system_messages, "Mistral-7B-Instruct-v0.1")
    print()

    # empty system message:
    empty_system_messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "What is your favourite condiment?"},
        {"role": "assistant", "content": "Lard!"},
        {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]
    # check empty system message with Mistral-7B-Instruct-v0.1:
    print("Empty system message:")
    test_backend.check_messages(empty_system_messages, "Mistral-7B-Instruct-v0.1")
    print("-----")

    # CONTEXT LIMIT CHECKING
    print("--- Context limit checking ---")
    # check minimal messages with Mistral-7B-Instruct-v0.1:
    print("Minimal messages context check with Mistral-7B-Instruct-v0.1:")
    minimal_context_check_tuple = test_backend.check_context_limit(minimal_messages, "Mistral-7B-Instruct-v0.1")
    print(f"Minimal messages context check output: {minimal_context_check_tuple}")
    print()
    # excessive number of messages:
    excessive_messages = list()
    for _ in range(2000):
        excessive_messages.append({"role": "user", "content": "What is your favourite condiment?"})
        excessive_messages.append({"role": "assistant", "content": "Lard!"})
    excessive_messages.append({"role": "user", "content": "Do you have mayonnaise recipes?"})
    # check excessive messages with Mistral-7B-Instruct-v0.1:
    print("Excessive messages context check with Mistral-7B-Instruct-v0.1:")
    excessive_context_check_tuple = test_backend.check_context_limit(excessive_messages, "Mistral-7B-Instruct-v0.1")
    print(f"Excessive messages context check output: {excessive_context_check_tuple}")
    """Note: Mistral-7B-Instruct-v0.1 has an official context limit of 32768, and while the context limit checks might 
    pass, using the full context of models with large limits like this is likely to use a great amount of memory (VRAM) 
    which can lead to CUDA Out-Of-Memory errors that are not only hard to handle, but can also incapacitate shared 
    hardware until it is manually reset. Please test for this while developing clemgames to prevent hardware outages 
    when the full set of clemgames is run by others."""
