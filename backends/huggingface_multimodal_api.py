"""
Backend for open-weight multimodal models.
"""
from typing import List, Dict, Tuple, Any
import torch
from transformers import (AutoProcessor, AutoConfig, AutoTokenizer)
import importlib

import backends

FALLBACK_CONTEXT_SIZE = 256
logger = backends.get_logger(__name__)

# CONTEXT UTILS
def get_context_limit(model_spec: backends.ModelSpec) -> int:
    """
    Get the context limit of the model.

    :param model_spec: Contains definitions about the model to be used.
    :return: Context limit of the model.
    """
    hf_model_str = model_spec['huggingface_id']
    trust_remote_code = getattr(model_spec, 'trust_remote_code', False)

    model_config = AutoConfig.from_pretrained(hf_model_str, trust_remote_code=trust_remote_code)

    # Some models have 'max_position_embeddings', others have 'max_sequence_length'
    context = getattr(
        getattr(model_config, 'text_config', model_config),
        'max_position_embeddings',
        getattr(model_config, 'max_sequence_length', FALLBACK_CONTEXT_SIZE)
    )

    logger.info(f"Context limit for model - {hf_model_str} is {context}")
    return context


# CONTEXT UTILS
def check_context_limit(context_size: int, prompt_tokens: list, max_new_tokens: int = 100) -> Tuple[bool, int, int, int]:
    """
    External context limit check
    :param context_size: max_sequence_length/max_position_embeddings of the model
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


# MODEL UTILS
def load_processor_or_tokenizer(model_spec: backends.ModelSpec):
    """
    Load processor/tokenizer from AutoProcessor/AutoTokenizer for a specific model
    (Ex. LlavaProcessor/InternLM2Tokenizer).
    Some models use AutoTokenizer and handle image processing separately (InternLM, for example)

    :param model_spec: Contains definitions the model to be used, loaded from Model Registry.
    :return input_handler: Processor/Tokenizer for the specific model.
    """
    hf_model_str = model_spec['huggingface_id']  # Get the model name

    """
    Change use_fast
    tokenizer -> use_tokenizer key in model registry
    Handle processor/tokenizer separately
    """

    use_fast = getattr(model_spec, 'use_fast', True)  # Default is True, set to False for Llava 34B via Model Registry
    use_tokenizer = getattr(model_spec, 'tokenizer', False)
    trust_remote_code = getattr(model_spec, 'trust_remote_code', False)
    not_distributed = getattr(model_spec, 'not_distributed', False)
    input_handler_class = AutoTokenizer if use_tokenizer else AutoProcessor

    input_handler = input_handler_class.from_pretrained(
        hf_model_str,
        use_fast=use_fast,
        device_map=None if not_distributed else "auto",
        verbose=False,
        trust_remote_code=trust_remote_code
    )

    logger.info(f'Loading {input_handler_class} for model : {model_spec.model_name}')
    return input_handler


# MODEL UTILS
def load_model(model_spec: backends.ModelSpec):
    """
    Load a specific model.

    :param model_spec: A dictionary that defines the model to be used, loaded from Model Registry.
    :return model: The specific model.
    """
    logger.info(f'Start loading huggingface model weights: {model_spec.model_name}')
    hf_model_str = model_spec['huggingface_id']  # Get the model name

    model_type_str = model_spec['automodel_type']  # Load the appropriate Auto class string to load the model
    module_path, class_name = model_type_str.rsplit('.', 1)
    module = importlib.import_module(module_path)
    model_type = getattr(module, class_name)

    trust_remote_code = getattr(model_spec, 'trust_remote_code', False)
    use_bf16 = getattr(model_spec, 'use_bf16', False)
    not_distributed = getattr(model_spec, 'not_distributed', False)

    model = model_type.from_pretrained(
        hf_model_str,
        torch_dtype=torch.bfloat16 if use_bf16 else "auto",
        trust_remote_code=trust_remote_code,
        device_map=None if not_distributed else "auto"
    )

    # Set pad_token_id to eos_token_id if it's not already set
    generation_config = model.generation_config
    if getattr(generation_config, 'pad_token_id', None) is None:
        generation_config.pad_token_id = generation_config.eos_token_id

    logger.info(f"Finished loading huggingface model: {model_spec.model_name}")

    # Log the device map if it's available
    # Some models do not support distributed inference, hence do not have device map arg in the model loader
    device_map = getattr(model, 'hf_device_map', None)
    if device_map:
        logger.info(f"Device Map: {device_map}")

    return model


# BACKEND UTILS
def check_multiple_images(messages: List[Dict]) -> bool:
    """
    Return True if any single message in messages contains multiple images.

    :param messages: A list[Dict] type object passed to the backend containing 'role', 'content', and 'image'.
    :return: True if any message contains multiple images, otherwise False.
    """
    return any('image' in msg and isinstance(msg['image'], list) and len(msg['image']) > 1 for msg in messages)


class HuggingfaceMultimodal(backends.Backend):
    def __init__(self):
        super().__init__()

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        return HuggingfaceMultimodalModel(model_spec)


class HuggingfaceMultimodalModel(backends.Model):

    def __init__(self, model_spec: backends.ModelSpec):
        super().__init__(model_spec)

        # Load instance variable used for evey model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.model_type = model_spec['automodel_type']
        self.model_name = model_spec['model_name']
        self.input_handler = load_processor_or_tokenizer(model_spec)
        self.multimodal_model = load_model(model_spec)
        self.split_prefix = model_spec['output_split_prefix']
        self.context_size = get_context_limit(model_spec)

        # Use the appropriate custom MLLM class to process inputs and generate outputs
        model_class_str = model_spec['model_class']
        module_path, class_name = model_class_str.rsplit('.', 1)
        module = importlib.import_module(module_path)
        self.model_class = getattr(module, class_name)()

        # Load model specific instance variables
        self.chat_template = getattr(model_spec, 'custom_chat_template', None)
        self.eos_to_cull = getattr(model_spec, 'eos_to_cull', None)
        self.supports_multiple_images = getattr(model_spec, 'supports_multiple_images', False)

    def generate_response(self, messages: List[Dict]) -> Tuple[Any, Any, str]:
        """
        :param messages: for example
                [
                    {"role": "user", "content": "Are there any clouds in the image? Answer with only "Yes" or "No"."},
                    {"role": "assistant", "content": "Yes"},
                    {"role": "user", "content": "This seems correct."},
                    {'role': 'user', 'content': 'Are there any chickens in the image? Answer with only "Yes" or "No".',
                     'image': 'games/cloudgame/resources/images/3.jpg'}
                ]
        :return: the continuation
        """

        # Check to see if game passes multiple images in a single turn
        # Proceed only if model supports multiple images, else return blanks for prompt, response and response_text
        has_multiple_images = check_multiple_images(messages=messages)
        if has_multiple_images and not self.supports_multiple_images:
            raise ValueError(f"Multiple images not supported in a single turn for model {self.model_name}")

        model_kwargs = {"template": self.chat_template, "max_tokens": self.get_max_tokens(), "device": self.device,
                        "split_prefix": self.split_prefix, "cull": self.eos_to_cull}

        inputs = self.model_class.prepare_inputs(messages=messages, **model_kwargs)

        prompt_text, images, output_kwargs = inputs['prompt'], inputs['images'], inputs['output_kwargs']

        prompt_tokens = self.model_class.get_tokens(prompt=prompt_text, handler=self.input_handler, **output_kwargs)

        # Check context limit
        context_check = check_context_limit(self.context_size, prompt_tokens, max_new_tokens=self.get_max_tokens())
        if not context_check[0]:  # if context is exceeded, context_check[0] is False
            logger.info(f"Context token limit for {self.model_spec.model_name} exceeded: "
                        f"{context_check[1]}/{context_check[3]}")
            # fail gracefully:
            raise backends.ContextExceededError(f"Context token limit for {self.model_spec.model_name} exceeded",
                                                tokens_used=context_check[1], tokens_left=context_check[2],
                                                context_size=context_check[3])

        prompt = {"inputs": prompt_text, "max_new_tokens": self.get_max_tokens(), "temperature": self.get_temperature()}

        response, response_text = self.model_class.generate_outputs(prompt=prompt_text, images=images,
                                                                    model=self.multimodal_model,
                                                                    handler=self.input_handler, **output_kwargs)

        return prompt, response, response_text