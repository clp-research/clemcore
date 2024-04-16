"""
    Backend using llama.cpp for GGUF/GGML models.
"""

from typing import List, Dict, Tuple, Any, Union

import backends
from backends.utils import check_context_limit_generic

# import torch

import llama_cpp
from llama_cpp import Llama

logger = backends.get_logger(__name__)


def load_model(model_spec: backends.ModelSpec) -> Any:
    """
    Load GGUF/GGML model weights from HuggingFace, into VRAM if available. Weights are distributed over all available
    GPUs for maximum speed - make sure to limit the available GPUs using environment variables if only a subset is to be
    used.
    :param model_spec: The ModelSpec for the model.
    :return: The llama_cpp model class instance of the loaded model.
    """
    logger.info(f'Start loading llama.cpp model weights from HuggingFace: {model_spec.model_name}')

    hf_repo_id = model_spec['huggingface_id']
    hf_model_file = model_spec['filename']

    # checking for GPU availability in python would require additional dependencies (pyTorch)
    # so GPU offloading is hardcoded for now

    if 'requires_api_key' in model_spec and model_spec['requires_api_key']:
        # load HF API key:
        creds = backends.load_credentials("huggingface")
        api_key = creds["huggingface"]["api_key"]
        # model = Llama.from_pretrained(hf_repo_id, hf_model_file, token=api_key, verbose=False)
        # load model on GPU:
        model = Llama.from_pretrained(hf_repo_id, hf_model_file, token=api_key, verbose=False, n_gpu_layers=-1)  # offloads all layers to GPU)
    else:
        # model = Llama.from_pretrained(hf_repo_id, hf_model_file, verbose=False)
        model = Llama.from_pretrained(hf_repo_id, hf_model_file, verbose=False, n_gpu_layers=-1)  # offloads all layers to GPU

    logger.info(f"Finished loading llama.cpp model: {model_spec.model_name}")

    return model


class LlamaCPPLocal(backends.Backend):
    """
    Model/backend handler class for locally-run GGUF/GGML models.
    """
    def __init__(self):
        super().__init__()

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        """
        Get a LlamaCPPLocalModel instance with the passed model and settings. Will load all required data for using
        the model upon initialization.
        :param model_spec: The ModelSpec for the model.
        :return: The Model class instance of the model.
        """
        # torch.set_num_threads(1)
        return LlamaCPPLocalModel(model_spec)


class LlamaCPPLocalModel(backends.Model):
    """
    Class for loaded llama.cpp models ready for generation.
    """
    def __init__(self, model_spec: backends.ModelSpec):
        super().__init__(model_spec)
        self.model = load_model(model_spec)

        # fallback context size:
        self.context_size = 512
        # get maximum context size from model metadata:
        for key, value in self.model.metadata.items():
            # print(key, value)
            if "context_length" in key:
                self.context_size = int(value)
        # set model instance context size to maximum context size:
        self.model._n_ctx = self.context_size

        # placeholders for BOS/EOS:
        self.bos_string = None
        self.eos_string = None

        # check chat template:
        if model_spec.premade_chat_template:
            # jinja chat template available in metadata
            self.chat_template = self.model.metadata['tokenizer.chat_template']
        else:
            self.chat_template = model_spec.custom_chat_template

        if hasattr(self.model, 'chat_handler'):
            if not self.model.chat_handler:
                # no custom chat handler
                pass
            else:
                # specific chat handlers may be needed for multimodal models
                # see https://llama-cpp-python.readthedocs.io/en/latest/#multi-modal-models
                pass

        if hasattr(self.model, 'chat_format'):
            if not self.model.chat_format:
                # no guessed chat format
                pass
            else:
                if self.model.chat_format == "chatml":
                    # get BOS/EOS strings for chatml from llama.cpp:
                    self.bos_string = llama_cpp.llama_chat_format.CHATML_BOS_TOKEN
                    self.eos_string = llama_cpp.llama_chat_format.CHATML_EOS_TOKEN
                elif self.model.chat_format == "mistral-instruct":
                    # get BOS/EOS strings for mistral-instruct from llama.cpp:
                    self.bos_string = llama_cpp.llama_chat_format.MISTRAL_INSTRUCT_BOS_TOKEN
                    self.eos_string = llama_cpp.llama_chat_format.MISTRAL_INSTRUCT_EOS_TOKEN

        # get BOS/EOS token string from model file:
        # NOTE: These may not be the expected tokens, checking these when model is added is likely necessary!
        for key, value in self.model.metadata.items():
            # print(key, value)
            if "bos_token_id" in key:
                self.bos_string = self.model._model.token_get_text(int(value))
                # print("BOS string from metadata:", self.bos_string)
            if "eos_token_id" in key:
                self.eos_string = self.model._model.token_get_text(int(value))
                # print("EOS string from metadata:", self.eos_string)

        # get BOS/EOS strings for template from registry if not available from model file:
        if not self.bos_string:
            self.bos_string = model_spec.bos_string
        if not self.eos_string:
            self.eos_string = model_spec.eos_string

        # init llama-cpp-python jinja chat formatter:
        self.chat_formatter = llama_cpp.llama_chat_format.Jinja2ChatFormatter(
            template=self.chat_template,
            bos_token=self.bos_string,
            eos_token=self.eos_string
        )

    def generate_response(self, messages: List[Dict], return_full_text: bool = False) -> Tuple[Any, Any, str]:
        """
        :param messages: for example
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        :param return_full_text: If True, whole input context is returned.
        :return: the continuation
        """
        # use llama.cpp jinja to apply chat template for prompt:
        prompt_text = self.chat_formatter(messages=messages).prompt

        prompt = {"inputs": prompt_text, "max_new_tokens": self.get_max_tokens(),
                  "temperature": self.get_temperature(), "return_full_text": return_full_text}

        prompt_tokens = self.model.tokenize(prompt_text.encode(), add_bos=False)

        # check context limit:
        context_check = check_context_limit_generic(self.context_size, prompt_tokens,
                                                    max_new_tokens=self.get_max_tokens())
        if not context_check[0]:  # if context is exceeded, context_check[0] is False
            logger.info(f"Context token limit for {self.model_spec.model_name} exceeded: "
                        f"{context_check[1]}/{context_check[3]}")
            # fail gracefully:
            raise backends.ContextExceededError(f"Context token limit for {self.model_spec.model_name} exceeded",
                                                tokens_used=context_check[1], tokens_left=context_check[2],
                                                context_size=context_check[3])

        # NOTE: HF transformers models come with their own generation configs, but llama.cpp doesn't seem to have a
        # feature like that. There are default sampling parameters, and clembench only handles two of them so far, which
        # are set accordingly. Other parameters use the llama-cpp-python default values for now.

        # NOTE: llama.cpp has a set sampling order, which differs from that of HF transformers. The latter allows
        # individual sampling orders defined in the generation config that comes with HF models.

        model_output = self.model.create_chat_completion(
            messages,
            temperature=self.get_temperature(),
            max_tokens=self.get_max_tokens()
        )

        response = {'response': model_output}
        
        # cull input context:
        if not return_full_text:
            response_text = model_output['choices'][0]['message']['content'].strip()

            if 'output_split_prefix' in self.model_spec:
                response_text = response_text.rsplit(self.model_spec['output_split_prefix'], maxsplit=1)[1]

            eos_len = len(self.model_spec['eos_to_cull'])

            if response_text.endswith(self.model_spec['eos_to_cull']):
                response_text = response_text[:-eos_len]

        else:
            response_text = prompt_text + model_output['choices'][0]['message']['content'].strip()

        return prompt, response, response_text
