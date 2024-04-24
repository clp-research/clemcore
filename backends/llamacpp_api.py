"""
    Backend using llama.cpp for GGUF/GGML models.
"""

from typing import List, Dict, Tuple, Any

import backends
from backends.utils import check_context_limit_generic

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

    # default to GPU offload:
    gpu_layers_offloaded = -1  # -1 = offload all model layers to GPU
    # check for optional execute_on flag:
    if hasattr(model_spec, 'execute_on'):
        if model_spec.execute_on == "gpu":
            gpu_layers_offloaded = -1
        elif model_spec.execute_on == "cpu":
            gpu_layers_offloaded = 0
    # check for optional gpu_layers_offloaded value:
    elif hasattr(model_spec, 'gpu_layers_offloaded'):
        gpu_layers_offloaded = model_spec.gpu_layers_offloaded

    if 'requires_api_key' in model_spec and model_spec['requires_api_key']:
        # load HF API key:
        creds = backends.load_credentials("huggingface")
        api_key = creds["huggingface"]["api_key"]
        model = Llama.from_pretrained(hf_repo_id, hf_model_file, token=api_key, verbose=False,
                                      n_gpu_layers=gpu_layers_offloaded, n_ctx=0)
    else:
        model = Llama.from_pretrained(hf_repo_id, hf_model_file, verbose=False, n_gpu_layers=gpu_layers_offloaded,
                                      n_ctx=0)

    logger.info(f"Finished loading llama.cpp model: {model_spec.model_name}")

    return model


def get_chat_formatter(model: Llama, model_spec: backends.ModelSpec) -> llama_cpp.llama_chat_format.Jinja2ChatFormatter:
    # placeholders for BOS/EOS:
    bos_string = None
    eos_string = None

    # check chat template:
    if model_spec.premade_chat_template:
        # jinja chat template available in metadata
        chat_template = model.metadata['tokenizer.chat_template']
    else:
        chat_template = model_spec.custom_chat_template

    if hasattr(model, 'chat_format'):
        if not model.chat_format:
            # no guessed chat format
            pass
        else:
            if model.chat_format == "chatml":
                # get BOS/EOS strings for chatml from llama.cpp:
                bos_string = llama_cpp.llama_chat_format.CHATML_BOS_TOKEN
                eos_string = llama_cpp.llama_chat_format.CHATML_EOS_TOKEN
            elif model.chat_format == "mistral-instruct":
                # get BOS/EOS strings for mistral-instruct from llama.cpp:
                bos_string = llama_cpp.llama_chat_format.MISTRAL_INSTRUCT_BOS_TOKEN
                eos_string = llama_cpp.llama_chat_format.MISTRAL_INSTRUCT_EOS_TOKEN

    # get BOS/EOS token string from model file:
    # NOTE: These may not be the expected tokens, checking these when model is added is likely necessary!
    if "tokenizer.ggml.bos_token_id" in model.metadata:
        bos_string = model._model.token_get_text(int(model.metadata.get("tokenizer.ggml.bos_token_id")))
    if "tokenizer.ggml.eos_token_id" in model.metadata:
        eos_string = model._model.token_get_text(int(model.metadata.get("tokenizer.ggml.eos_token_id")))

    # get BOS/EOS strings for template from registry if not available from model file:
    if not bos_string:
        bos_string = model_spec.bos_string
    if not eos_string:
        eos_string = model_spec.eos_string

    # init llama-cpp-python jinja chat formatter:
    chat_formatter = llama_cpp.llama_chat_format.Jinja2ChatFormatter(
        template=chat_template,
        bos_token=bos_string,
        eos_token=eos_string
    )

    return chat_formatter


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
        return LlamaCPPLocalModel(model_spec)


class LlamaCPPLocalModel(backends.Model):
    """
    Class for loaded llama.cpp models ready for generation.
    """
    def __init__(self, model_spec: backends.ModelSpec):
        super().__init__(model_spec)
        self.model = load_model(model_spec)

        self.chat_formatter = get_chat_formatter(self.model, model_spec)

        if hasattr(self.model, 'chat_handler'):
            if not self.model.chat_handler:
                # no custom chat handler
                pass
            else:
                # specific chat handlers may be needed for multimodal models
                # see https://llama-cpp-python.readthedocs.io/en/latest/#multi-modal-models
                pass

        # get context size from model instance:
        self.context_size = self.model._n_ctx

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

        prompt_tokens = self.model.tokenize(prompt_text.encode(), add_bos=False)  # BOS expected in template

        # check context limit:
        check_context_limit_generic(self.context_size, prompt_tokens, self.model_spec.model_name,
                                    max_new_tokens=self.get_max_tokens())

        # NOTE: HF transformers models come with their own generation configs, but llama.cpp doesn't seem to have a
        # feature like that. There are default sampling parameters, and clembench only handles two of them so far, which
        # are set accordingly. Other parameters use the llama-cpp-python default values for now.

        # NOTE: llama.cpp has a set sampling order, which differs from that of HF transformers. The latter allows
        # individual sampling orders defined in the generation config that comes with HF models.

        model_output = self.model(
            prompt_text,
            temperature=self.get_temperature(),
            max_tokens=self.get_max_tokens()
        )

        response = {'response': model_output}

        # cull input context:
        if not return_full_text:
            response_text = model_output['choices'][0]['text'].strip()

            if 'output_split_prefix' in self.model_spec:
                response_text = response_text.rsplit(self.model_spec['output_split_prefix'], maxsplit=1)[1]

            eos_len = len(self.model_spec['eos_to_cull'])

            if response_text.endswith(self.model_spec['eos_to_cull']):
                response_text = response_text[:-eos_len]

        else:
            response_text = prompt_text + model_output['choices'][0]['text'].strip()

        return prompt, response, response_text
