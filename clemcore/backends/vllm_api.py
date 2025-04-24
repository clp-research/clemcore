"""
    Backend wrapping vLLM.
    Uses HF tokenizers instruct/chat templates for proper input format per model.
"""
import logging
import re
import torch
import vllm

from typing import List, Dict, Tuple, Any, Union
from transformers import AutoTokenizer, AutoConfig

from jinja2 import TemplateError
import clemcore.backends as backends
from clemcore.backends.utils import ensure_alternating_roles

logger = logging.getLogger(__name__)

FALLBACK_CONTEXT_SIZE = 256


def load_config_and_tokenizer(model_spec: backends.ModelSpec) -> Union[AutoTokenizer, AutoConfig, int]:
    """
    Load a HuggingFace model's standard config and tokenizer, and get context token limit from config. If the model
    config does not contain the context limit, it is set to 256 as fallback. Does not load the model weights, as those
    are handled by vLLM.
    :param model_spec: The ModelSpec for the model.
    :return: Tokenizer, model config and context token limit (int).
    """
    logger.info(f'Loading model config and tokenizer from HuggingFace: {model_spec.model_name}')

    use_api_key = False
    api_key = None
    if 'requires_api_key' in model_spec['model_config']:
        if model_spec['model_config']['requires_api_key']:
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
    if 'slow_tokenizer' in model_spec['model_config']:
        if model_spec['model_config']['slow_tokenizer']:
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
    if not model_spec['model_config']['premade_chat_template']:
        if 'custom_chat_template' in model_spec['model_config']:
            tokenizer.chat_template = model_spec['model_config']['custom_chat_template']
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

    # stopping transformers pad_token_id warnings
    # check if tokenizer has no set pad_token_id:
    if not tokenizer.pad_token_id:  # if not set, pad_token_id is None
        # preemptively set pad_token_id to eos_token_id as automatically done to prevent warning at each generation:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer, model_config, context_size


def load_model(model_spec: backends.ModelSpec) -> Any:
    """
    Load model weights from HuggingFace, onto the number of GPUs specified in the ModelSpec/model registry entry.
    :param model_spec: The ModelSpec for the model.
    :return: The vLLM LLM class instance of the loaded model.
    """
    assert "model_config" in model_spec, "vllm model requires model_config entry in model spec"
    model_config = model_spec.model_config

    default_args = dict(tensor_parallel_size=model_config['number_gpus'] if 'number_gpus' in model_config else 1)
    max_model_len = int(model_spec.context_size) if 'context_size' in model_spec and model_spec.context_size else None
    if max_model_len is not None:
        default_args["max_model_len"] = max_model_len

    vllm_args = model_config['vllm_args'] if 'vllm_args' in model_config else {}
    model_args = {**default_args, **vllm_args}
    logger.info(f"Number of GPUs used for model: {model_args['tensor_parallel_size']}")
    if "max_model_len" in model_args:
        logger.info(f"Context size forcefully limited to {model_args['max_model_len']} tokens.")

    logger.info(f'Start loading model weights from HuggingFace: {model_spec.model_name}')
    model = vllm.LLM(model_spec.huggingface_id, **model_args)
    logger.info(f"Finished loading model weights from HuggingFace: {model_spec.model_name}")
    return model


class VLLMLocal(backends.Backend):
    """
    Model/backend handler class for locally-run models using vLLM.
    """

    def __init__(self):
        super().__init__()

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        """
        Get a VLLMLocalModel instance with the passed model and settings. Will load all required data for using
        the model upon initialization.
        :param model_spec: The ModelSpec for the model.
        :return: The Model class instance of the model.
        """
        torch.set_num_threads(1)
        return VLLMLocalModel(model_spec)


class VLLMLocalModel(backends.Model):
    """
    Class for loaded vLLM models ready for generation.
    Implementation based on https://huggingface.co/neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8#use-with-vllm and the
    existing HuggingFace backend - *not* the vLLM example code for 'Offline Inference Chat' found at
    https://docs.vllm.ai/en/latest/getting_started/examples/offline_inference_chat.html !
    """
    def __init__(self, model_spec: backends.ModelSpec):
        super().__init__(model_spec)
        # fail-fast
        self.tokenizer, self.config, self.context_size = load_config_and_tokenizer(model_spec)
        self.model = load_model(model_spec)

        """
        # generation_config not used with vLLM
        # check if model's generation_config has pad_token_id set:
        if not self.model.generation_config.pad_token_id:
            # set pad_token_id to tokenizer's eos_token_id to prevent excessive warnings:
            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        """

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
        :param return_full_text: If True, whole input context is returned.
        :param log_messages: If True, raw and cleaned messages passed will be logged.
        :return: the continuation
        """
        # log current given messages list:
        if log_messages:
            logger.info(f"Raw messages passed: {messages}")

        current_messages = ensure_alternating_roles(messages)

        # log current flattened messages list:
        if log_messages:
            logger.info(f"Flattened messages: {current_messages}")

        # apply chat template & tokenize:
        prompt_tokens = self.tokenizer.apply_chat_template(current_messages, add_generation_prompt=True,
                                                           return_tensors="pt")
        # do not send to device for vLLM:
        # prompt_tokens = prompt_tokens.to(self.device)

        # decode again to get properly formatted prompt text:
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

        # vLLM sampling parameters:
        sampling_params = vllm.SamplingParams(
            temperature=self.get_temperature(),
            # using example value from https://huggingface.co/neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8#use-with-vllm:
            # top_p=0.9,
            max_tokens=self.get_max_tokens())

        model_output = self.model.generate(prompt_text, sampling_params)[0].outputs[0].text

        response = {'response': model_output}

        # cull input context; equivalent to transformers.pipeline method:
        if not return_full_text:
            response_text = model_output.replace(prompt_text, '').strip()

            if 'output_split_prefix' in self.model_spec['model_config']:
                response_text = model_output.rsplit(self.model_spec['model_config']['output_split_prefix'], maxsplit=1)[1]

            # remove eos token string:
            eos_to_cull = self.model_spec['model_config']['eos_to_cull']
            response_text = re.sub(eos_to_cull, "", response_text)
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


def check_messages(messages: List[Dict], model_spec: backends.ModelSpec) -> bool:
    """
    Message checking for clemgame development. This checks if the model's chat template accepts the given messages
    as passed, before the standard flattening done for generation. This allows clemgame developers to construct
    message lists that are sound as-is and are not affected by the indiscriminate flattening of the generation
    method. Deliberately verbose.
    :param model_spec: The ModelSpec for the model.
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
    :param model_spec: The ModelSpec for the model.
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
        current_messages = ensure_alternating_roles(messages)
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