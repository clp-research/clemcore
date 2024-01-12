""" Backend using HuggingFace transformers & ungated models. Uses HF tokenizers instruct/chat templates for proper input format per model. """
from typing import List, Dict, Tuple, Any
import torch
import backends

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import copy
import json

with open("hf_local_models.json", 'r', encoding="utf-8") as model_registry_file:
    MODEL_REGISTRY = json.load(model_registry_file)

logger = backends.get_logger(__name__)

NAME = "huggingface"

SUPPORTED_MODELS = [model_setting['model_name'] for model_setting in MODEL_REGISTRY if model_setting['backend'] == NAME]

FALLBACK_CONTEXT_SIZE = 256


class HuggingfaceLocal(backends.Backend):
    def __init__(self):
        self.temperature: float = -1.
        self.model_loaded = False

    def load_model(self, model_name):
        logger.info(f'Start loading huggingface model: {model_name}')

        # model cache handling
        root_data_path = os.path.join(os.path.abspath(os.sep), "data")
        # check if root/data exists:
        if not os.path.isdir(root_data_path):
            logger.info(f"{root_data_path} does not exist, creating directory.")
            # create root/data:
            os.mkdir(root_data_path)
        CACHE_DIR = os.path.join(root_data_path, "huggingface_cache")

        # get settings from model registry for the first name match that uses this backend:
        for model_setting in MODEL_REGISTRY:
            if model_setting['model_name'] == model_name:
                if model_setting['backend'] == "huggingface":
                    self.model_settings = model_setting
                    break

        assert self.model_settings['model_name'] == model_name, (f"Model settings for {model_name} not properly loaded "
                                                                 f"from model registry!")

        hf_model_str = self.model_settings['huggingface_id']

        # use 'slow' tokenizer for models that require it:
        if 'slow_tokenizer' in self.model_settings:
            if self.model_settings['slow_tokenizer']:
                self.tokenizer = AutoTokenizer.from_pretrained(hf_model_str, device_map="auto", torch_dtype="auto",
                                                           cache_dir=CACHE_DIR, verbose=False, use_fast=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_str, device_map="auto", torch_dtype="auto",
                                                           cache_dir=CACHE_DIR, verbose=False)

        # apply proper chat template:
        if not self.model_settings['premade_chat_template']:
            if 'custom_chat_template' in self.model_settings:
                self.tokenizer.chat_template = self.model_settings['custom_chat_template']
            else:
                logger.info(f"No custom chat template for {model_name} found in model settings from model registry "
                            f"while model has no pre-made template! Generic template will be used, likely leading to "
                            f"bad results.")

        # load model using its default configuration:
        self.model = AutoModelForCausalLM.from_pretrained(hf_model_str, device_map="auto", torch_dtype="auto",
                                                          cache_dir=CACHE_DIR)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model_loaded = True

        # get context token limit for loaded model:
        if hasattr(self.model.config, 'max_position_embeddings'):  # this is the standard attribute used by most
            self.context_size = self.model.config.max_position_embeddings
        elif hasattr(self.model.config, 'n_positions'):  # some models may have their context size under this attribute
            self.context_size = self.model.config.n_positions
        else:  # few models, especially older ones, might not have their context size in the config
            self.context_size = FALLBACK_CONTEXT_SIZE

    def _check_messages(self, messages: List[Dict]):
        """
        internal message checking
        to be extended with model/template-specific checks and warnings
        """
        if messages[0]['role'] == "system":
            if not messages[0]['content']:
                print(f"Initial system message is empty!")

    def _check_context_limit(self, prompt_tokens, max_new_tokens: int = 100) -> Tuple[bool, int, int, int]:
        """
        internal context limit check to run in generate_response
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
        # print(f"{tokens_used} input tokens, {tokens_left}/{self.context_size} tokens left.")
        fits = tokens_used <= self.context_size
        return fits, tokens_used, tokens_left, self.context_size

    def check_context_limit(self, messages: List[Dict], model: str, max_new_tokens: int = 100) -> Tuple[bool, int, int, int]:
        """
        externally-callable context limit check
        needs more refactoring to not require full model being loaded as for inference
        """
        prompt_tokens = self.tokenizer.apply_chat_template(messages)  # the actual tokens, including chat format
        prompt_size = len(prompt_tokens)
        tokens_used = prompt_size + max_new_tokens  # context includes tokens to be generated
        tokens_left = self.context_size - tokens_used
        print(f"{tokens_used} input tokens, {tokens_left}/{self.context_size} tokens left.")
        fits = tokens_used <= self.context_size
        return fits, tokens_used, tokens_left, self.context_size

    def generate_response(self, messages: List[Dict], model: str,
                          max_new_tokens: int = 100, return_full_text: bool = False) -> Tuple[Any, Any, str]:
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
        :return: the continuation
        """
        assert 0.0 <= self.temperature <= 1.0, "Temperature must be in [0.,1.]"

        # load the model to the memory
        if not self.model_loaded:
            self.load_model(model)
            logger.info(f"Finished loading huggingface model: {model}")
            logger.info(f"Model device map: {self.model.hf_device_map}")

        # log current given messages list:
        # logger.info(f"Raw messages passed: {messages}")

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

        # log current flattened messages list:
        # logger.info(f"Flattened messages: {current_messages}")

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
            # fail gracefully here

        # greedy decoding:
        do_sample: bool = False
        if self.temperature > 0.0:
            do_sample = True

        # test to check if temperature is properly set on this Backend object:
        # logger.info(f"Currently used temperature for this instance of HuggingfaceLocal: {self.temperature}")

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

            if response_text[-eos_len:len(response_text)] == self.model_settings['eos_to_cull']:
                response_text = response_text[:-eos_len]

        else:
            response_text = model_output.strip()

        return prompt, response, response_text

    def supports(self, model_name: str):
        return model_name in SUPPORTED_MODELS


if __name__ == "__main__":
    # print(MODEL_REGISTRY)
    # print(SUPPORTED_MODELS)
    # test_instance = HuggingfaceLocal()
    pass
