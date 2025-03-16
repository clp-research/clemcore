"""
Backend using HuggingFace transformers for open-weight multimodal models.
"""
from typing import List, Dict, Tuple, Any
import torch
import clemcore.backends as backends
from transformers import AutoTokenizer, AutoConfig
from jinja2 import Template
import warnings
import importlib
import logging
FALLBACK_CONTEXT_SIZE = 256

logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("huggingface.multimodal.api")


def check_context_limit(context_size: int, prompt_tokens: list, max_new_tokens: int = 100) -> Tuple[
    bool, int, int, int]:
    """
    Checks if the context limit is exceeded.

    Args:
        context_size (int): The maximum sequence length or position embeddings of the model.
        prompt_tokens (list): A list of prompt token IDs.
        max_new_tokens (int, optional): The maximum number of tokens to generate. Defaults to 100.

    Returns:
        Tuple[bool, int, int, int]: A tuple containing:
            - bool: True if the context limit is not exceeded, False if too many tokens.
            - int: The total number of tokens used (prompt + new tokens).
            - int: The number of tokens of 'context space left'.
            - int: The total context token limit.
    """
    prompt_size = len(prompt_tokens)
    tokens_used = prompt_size + max_new_tokens
    tokens_left = context_size - tokens_used
    fits = tokens_used <= context_size
    return fits, tokens_used, tokens_left, context_size

def import_method(method_path: str):
    """
    Import the method from the specified module path.
    
    Args:
        method_path (str): The method path separated by dots. Example - clemcore.backends.multimodal_utils.generate_internvl2_prompt_text

    Returns:
        type: The imported method.

    Raises:
        ImportError: If the method cannot be imported.
    """
    try:
        module_path, method_name = method_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, method_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import method '{method_name}' from module '{module_path}'. Error: {e}") from e



def load_processor(model_spec: backends.ModelSpec):
    """
    Load processor from AutoProcessor/AutoTokenizer for a specific model (Example - LlavaProcessor).

    Args:
        model_spec (backends.ModelSpec): A dictionary that defines the model to be used, loaded from Model Registry.

    Returns:
        Processor/Tokenizer for the specific model.

    Raises:
        ImportError: If the processor type cannot be imported.
    """
    hf_model_str = model_spec['huggingface_id']  # Get the model name
    processor_class_str = model_spec['model_config']['processor_class']  # Processor type - AutoProcessor/AutoTokenizer
    processor_config = model_spec['model_config']['processor_config']  # Processor kwargs

    processor_class = import_method(processor_class_str)

    if "trust_remote_code" in model_spec.model_config:
        processor = processor_class.from_pretrained(hf_model_str, trust_remote_code=True, **processor_config) # Load the processor with trust_remote_code=True
    else:
        processor = processor_class.from_pretrained(hf_model_str, **processor_config) # Load the processor with defined args

    stdout_logger.info(f'Loading Processor for model : {model_spec.model_name}')

    return processor


def load_model(model_spec: backends.ModelSpec):
    """
    Load a specific model.

    Args:
        model_spec (backends.ModelSpec): A dictionary that defines the model to be used, loaded from Model Registry.

    Returns:
        backends.Model: The specific model.

    Raises:
        ImportError: If the model class or device map (if custom) cannot be imported.
    """
    stdout_logger.info(f'Start loading huggingface model weights: {model_spec.model_name}')
    hf_model_str = model_spec['huggingface_id']  # Get the model name
    model_class_str = model_spec['model_config']['model_class']  # Model Loader Class
    model_config = model_spec['model_config']['mm_model_config']  # Model kwargs

    model_class = import_method(model_class_str)

    # Check if a custom device_map split is provided and adjust device_map accordingly
    if 'device_map' in model_config and not model_config['device_map'] == 'auto':
        stdout_logger.info(f"Loading Custom device map for model: {hf_model_str}")
        split_model = import_method(model_config['device_map'])
        device_map = split_model(model_spec['model_name'])
        model_config['device_map'] = device_map

    if 'trust_remote_code' in model_spec.model_config:
        model = model_class.from_pretrained(hf_model_str, trust_remote_code=True, **model_config)  # Load the model using from_pretrained
    else:
        model = model_class.from_pretrained(hf_model_str, **model_config)  # Load the model using from_pretrained

    # Check if model's generation_config has pad_token_id set:
    if not model.generation_config.pad_token_id:
        if model.generation_config.pad_token_id != 0: # Additional check for models that have preset pad_token_id = 0, example Gemma
            # Set pad_token_id to tokenizer's eos_token_id to prevent excessive warnings:
            model.generation_config.pad_token_id = model.generation_config.eos_token_id  # Same as processor.tokenizer.pad_token_id

    stdout_logger.info(f"Finished loading huggingface model: {model_spec.model_name}")
    stdout_logger.info(f"Device Map: {model.hf_device_map}")

    return model


def check_multiple_image(messages: List[Dict]):
    """
    Check if a single message contains multiple images.

    Args:
        messages (List[Dict]): A list of dictionaries passed to the backend,
                                each containing 'role', 'content', and possibly 'image'.

    Returns:
        bool: True if any message contains multiple images, False otherwise.
    """
    has_multiple_images = False
    for msg in messages:
        if 'image' in msg and type(msg['image']) == list:
            if len(msg['image']) > 1:
                has_multiple_images = True

    return has_multiple_images


class HuggingfaceMultimodal(backends.Backend):
    def __init__(self):
        super().__init__()

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        """Get the model for the specified model specification.

        Args:
            model_spec (backends.ModelSpec): The model specification.

        Returns:
            backends.Model: The model instance.
        """
        return HuggingfaceMultimodalModel(model_spec)


class HuggingfaceMultimodalModel(backends.Model):

    def __init__(self, model_spec: backends.ModelSpec):
        super().__init__(model_spec)

        # Load instance variable used for evey model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = load_processor(model_spec)
        self.multimodal_model = load_model(model_spec)
        self.context_size = int(model_spec['context_size'].replace("k", ""))*1000
        self.model_name = model_spec['model_name']

        self.split_prefix = model_spec.model_config['output_split_prefix'] if 'output_split_prefix' in model_spec.model_config else ""
        self.template = model_spec.model_config['custom_chat_template'] if 'custom_chat_template' in model_spec.model_config else None
        self.premade_template = True if 'premade_chat_template' in model_spec.model_config else False
        self.cull = model_spec.model_config['eos_to_cull'] if 'eos_to_cull' in model_spec.model_config else None
        self.supports_multiple_images = model_spec.model_config['multimodality']['multiple_images'] if 'multiple_images' in model_spec.model_config['multimodality'] else False
        self.do_sample = model_spec.model_config['do_sample'] if 'do_sample' in model_spec.model_config else None
        self.prompt_method = model_spec.model_config['prompt'] if 'prompt' in model_spec.model_config else None
        self.response_method = model_spec.model_config['response'] if 'response' in model_spec.model_config else None


    def generate_response(self, messages: List[Dict]) -> Tuple[Any, Any, str]:
        """Generate a response based on the provided messages.

        Args:
            messages (List[Dict]): A list of message dictionaries, each containing 'role', 'content' and possibly 'images'.

        Returns:
            Tuple[Any, Any, str]: A tuple containing:
                - dict: The prompt for the model.
                - dict: The response from the model.
                - str: The processed response text.

        Raises:
            AttributeError: If neither 'tokenizer.tokenize' nor 'processor.tokenize' exists.
            backends.ContextExceededError: If the context token limit is exceeded.
            ValueError: If neither custom chat template or custom prompt method is provided
        """
        # Check to see if game passes multiple images in a single turn
        # Proceed only if model supports multiple images, else return blanks for prompt, response and response_text
        has_multiple_images = check_multiple_image(messages=messages)
        if has_multiple_images and not self.supports_multiple_images:
            logger.warning(f"Multiple images not supported in a single turn for model {self.model_name}")
            return "", {"response": ""}, ""

        prompt_kwargs = {
            'model': self.multimodal_model,
            'processor': self.processor,
            'device': self.device,
        }
        prompt_text = ""
        # Get input prompt by applying jinja template, if template is provided
        if self.template:
            template_str = self.template
            template = Template(template_str)
            prompt_text = template.render(messages=messages)
        elif self.prompt_method:
            prompt_method = import_method(self.prompt_method)
            prompt_text = prompt_method(messages,  **prompt_kwargs)
        else:
            raise ValueError("Neither template nor prompt method is provided.")


        # Check context limit based on if AutoProcessor is loaded or AutoTokenizer
        if hasattr(self.processor, 'tokenize'):
            prompt_tokens = self.processor.tokenize(prompt_text)
        elif hasattr(self.processor.tokenizer, 'tokenize'):
            prompt_tokens = self.processor.tokenizer.tokenize(prompt_text)
        else:
            raise AttributeError("Neither 'tokenizer.tokenize' nor 'processor.tokenize' exists.")

        context_check = check_context_limit(self.context_size, prompt_tokens, max_new_tokens=self.get_max_tokens())
        if not context_check[0]:  # if context is exceeded, context_check[0] is False
            logger.info(f"Context token limit for {self.model_spec.model_name} exceeded: "
                        f"{context_check[1]}/{context_check[3]}")
            # fail gracefully:
            raise backends.ContextExceededError(f"Context token limit for {self.model_spec.model_name} exceeded",
                                                tokens_used=context_check[1], tokens_left=context_check[2],
                                                context_size=context_check[3])

        prompt = {"inputs": prompt_text, "max_new_tokens": self.get_max_tokens(), "temperature": self.get_temperature()}

        stdout_logger.info(f" INPUT PROMPT : \n\n{prompt_text} \n\n")
        response_method = import_method(self.response_method)
        response_kwargs = {
            'model': self.multimodal_model,
            'processor': self.processor,
            'device': self.device,
            'do_sample': self.do_sample,
            'messages': messages,
            'max_tokens': self.get_max_tokens(),
            'model_name': self.model_name
        }
        generated_response = response_method(**response_kwargs)

        # Store generated text
        response = {"response": generated_response}

        # Check if split_prefix is not empty before splitting
        response_text = generated_response
        if self.split_prefix:
            response_text = generated_response.split(self.split_prefix)[-1]  # Get the last assistant response
        if self.cull:
            rt_split = response_text.split(self.cull)  # Cull from End of String token
            response_text = rt_split[0]
        response_text = response_text.strip()

        stdout_logger.info(f" RESPONSE TEXT : \n\n{response_text} \n\n")
        return prompt, response, response_text