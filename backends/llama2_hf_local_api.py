"""Llama2 backend API - uses HuggingFace transformers to load weights and run inference"""

from typing import List, Dict, Tuple, Any, Optional
import torch
import backends
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import copy

logger = backends.get_logger(__name__)

MODEL_LLAMA2_7B_HF = "llama-2-7b-hf"
MODEL_LLAMA2_13B_HF = "llama-2-13b-hf"
MODEL_LLAMA2_70B_HF = "llama-2-70b-hf"
MODEL_LLAMA2_7B_C_HF = "llama-2-7b-chat-hf"
MODEL_LLAMA2_13B_C_HF = "llama-2-13b-chat-hf"
MODEL_LLAMA2_70B_C_HF = "llama-2-70b-chat-hf"

SUPPORTED_MODELS = [MODEL_LLAMA2_7B_HF, MODEL_LLAMA2_13B_HF, MODEL_LLAMA2_70B_HF,
                    MODEL_LLAMA2_7B_C_HF, MODEL_LLAMA2_13B_C_HF, MODEL_LLAMA2_70B_C_HF]

NAME = "llama2-hf"


class Llama2LocalHF(backends.Backend):
    def __init__(self):
        # load HF API key:
        creds = backends.load_credentials("huggingface")
        self.api_key = creds["huggingface"]["api_key"]

        self.chat_models: List = [MODEL_LLAMA2_7B_C_HF, MODEL_LLAMA2_13B_C_HF, MODEL_LLAMA2_70B_C_HF]
        self.temperature: float = -1.
        self.model_loaded: bool = False

    def load_model(self, model_name: str):
        assert model_name in SUPPORTED_MODELS, f"{model_name} is not supported, please make sure the model name is correct."
        logger.info(f'Start loading llama2-hf model: {model_name}')

        # model cache handling
        root_data_path = os.path.join(os.path.abspath(os.sep), "data")
        # check if root/data exists:
        if not os.path.isdir(root_data_path):
            logger.info(f"{root_data_path} does not exist, creating directory.")
            # create root/data:
            os.mkdir(root_data_path)
        CACHE_DIR = os.path.join(root_data_path, "huggingface_cache")

        # full HF model id string:
        hf_id_str = f"meta-llama/{model_name.capitalize()}"
        # load tokenizer and model:
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id_str, token=self.api_key, device_map="auto",
                                                       cache_dir=CACHE_DIR, verbose=False)
        self.model = AutoModelForCausalLM.from_pretrained(hf_id_str, token=self.api_key,
                                                          torch_dtype="auto", device_map="auto",
                                                          cache_dir=CACHE_DIR)
        # use CUDA if available:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model_loaded = True

    def generate_response(self, messages: List[Dict], model: str,
                          max_new_tokens: Optional[int] = 100, top_p: float = 0.9) -> Tuple[str, Any, str]:
        """
        :param messages: for example
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        :param model: model name, chat models for chat-completion, otherwise text completion
        :param max_new_tokens: Maximum generation length.
        :param top_p: Top-P sampling parameter. Only applies when do_sample=True.
        :return: the continuation
        """
        assert 0.0 <= self.temperature <= 1.0, "Temperature must be in [0.,1.]"

        # load the model to the memory
        if not self.model_loaded:
            self.load_model(model)
            logger.info(f"Finished loading llama2-hf model: {model}")
            logger.info(f"Model device map: {self.model.hf_device_map}")

        # greedy decoding:
        do_sample: bool = False
        if self.temperature > 0.0:
            do_sample = True

        # turn off redundant transformers warnings:
        transformers.logging.set_verbosity_error()

        # deepcopy messages to prevent reference issues:
        current_messages = copy.deepcopy(messages)

        if model in self.chat_models:  # chat completion
            # flatten consecutive user messages:
            for msg_idx, message in enumerate(current_messages):
                if msg_idx > 0 and message['role'] == "user" and current_messages[msg_idx - 1]['role'] == "user":
                    current_messages[msg_idx - 1]['content'] += f" {message['content']}"
                    del current_messages[msg_idx]
                elif msg_idx > 0 and message['role'] == "assistant" and current_messages[msg_idx - 1]['role'] == "assistant":
                    current_messages[msg_idx - 1]['content'] += f" {message['content']}"
                    del current_messages[msg_idx]

            # apply chat template & tokenize
            prompt_tokens = self.tokenizer.apply_chat_template(current_messages, return_tensors="pt")
            prompt_tokens = prompt_tokens.to(self.device)
            # apply chat template for records:
            prompt_text = self.tokenizer.batch_decode(prompt_tokens)[0]
            prompt = {"inputs": prompt_text, "max_new_tokens": max_new_tokens,
                      "temperature": self.temperature}

            if do_sample:
                model_output_ids = self.model.generate(
                    prompt_tokens,
                    do_sample=do_sample,
                    max_new_tokens=max_new_tokens,
                    temperature=self.temperature,
                    top_p=top_p
                )
            else:
                model_output_ids = self.model.generate(
                    prompt_tokens,
                    do_sample=do_sample,
                    max_new_tokens=max_new_tokens
                )

            model_output = self.tokenizer.batch_decode(model_output_ids)[0]

            response = {"response": model_output}

            # cull prompt from output:
            response_text = model_output.replace(prompt_text, "").strip()
            # remove EOS token at the end of output:
            if response_text[-4:len(response_text)] == "</s>":
                response_text = response_text[:-4]



        else:  # default (text completion)
            prompt = "\n".join([message["content"] for message in current_messages])

            prompt_tokens = self.tokenizer.encode(
                prompt,
                add_bos_token=True,
                add_eos_token=False,
            )

            if do_sample:
                model_output_ids = self.model.generate(
                    prompt_tokens,
                    do_sample=do_sample,
                    max_new_tokens=max_new_tokens,
                    temperature=self.temperature,
                    top_p=top_p
                )
            else:
                model_output_ids = self.model.generate(
                    prompt_tokens,
                    do_sample=do_sample,
                    max_new_tokens=max_new_tokens
                )

            model_output = self.tokenizer.batch_decode(model_output_ids)[0]

            response_text = model_output.replace(prompt, '').strip()

            response = {'response': response_text}

        return prompt, response, response_text

    def supports(self, model_name: str):
        return model_name in SUPPORTED_MODELS
