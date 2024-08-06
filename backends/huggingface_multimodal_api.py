"""
Backend using HuggingFace transformers for open-weight multimodal models.
"""
from typing import List, Dict, Tuple, Any
import torch
import backends
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForVision2Seq, IdeficsForVisionText2Text, AutoConfig
from jinja2 import Template

# Define a map to load model from transformers Auto Classes
# IdeficsForVisionText2Text is not yet supported by any Auto Class
MODEL_TYPE_MAP = {
    "Idefics": IdeficsForVisionText2Text,
    "Vision2Seq": AutoModelForVision2Seq
}

FALLBACK_CONTEXT_SIZE = 256

logger = backends.get_logger(__name__)

def get_context_limit(model_spec: backends.ModelSpec) -> int:
    """
    Get the context limit of the model

    :param model_spec: Contains definitions about the model to be used
    :return context: Context limit of the model
    """
    hf_model_str = model_spec['huggingface_id']
    model_config = AutoConfig.from_pretrained(hf_model_str)

    # Some models have 'max_position_embeddings' others have - 'max_sequence_length'
    if hasattr(model_config, "text_config"):
        context = model_config.text_config.max_position_embeddings
    elif hasattr(model_config, "max_sequence_length"):
        context = model_config.max_sequence_length
    else:
        context = FALLBACK_CONTEXT_SIZE
    logger.info(f"Context limit for model - {hf_model_str} is {context}")

    return context


def check_context_limit(context_size: int, prompt_tokens: list, max_new_tokens: int = 100) -> Tuple[
    bool, int, int, int]:
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


def load_processor(model_spec: backends.ModelSpec) -> AutoProcessor:
    """
    Load processor from AutoProcessor a specific model (Example - LlavaProcessor)

    :param model_spec: A dictionary that defines the model to be used, loaded from Model Registry
    :return processor: Processor for the specific model
    """
    hf_model_str = model_spec['huggingface_id']  # Get the model name

    if hasattr(model_spec, 'not_fast'):
        # Only used by LLaVA 1.6 34B (Throws mismatch <image> token error when use_fast is not set to False)
        processor = AutoProcessor.from_pretrained(hf_model_str, use_fast=False, device_map="auto", verbose=False)
    else:
        processor = AutoProcessor.from_pretrained(hf_model_str, device_map="auto", verbose=False)
    logger.info(f'Loading Processor for model : {model_spec.model_name}')

    return processor


def load_model(model_spec: backends.ModelSpec):
    """
    Load a specific model

    :param model_spec: A dictionary that defines the model to be used, loaded from Model Registry
    :return model: The specific model
    """
    logger.info(f'Start loading huggingface model weights: {model_spec.model_name}')
    hf_model_str = model_spec['huggingface_id']  # Get the model name

    model_type = MODEL_TYPE_MAP[model_spec['model_type']]  # Use the appropriate Auto class to  load the model

    model = model_type.from_pretrained(hf_model_str, device_map="auto", torch_dtype="auto")  # Load the model

    # check if model's generation_config has pad_token_id set:
    if not model.generation_config.pad_token_id:
        # set pad_token_id to tokenizer's eos_token_id to prevent excessive warnings:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id  # Same as processor.tokenizer.pad_token_id

    logger.info(f"Finished loading huggingface model: {model_spec.model_name}")
    logger.info(f"Device Map: {model.hf_device_map}")

    return model


def load_image(image: str):
    """
    Load an image based on a given local path or URL

    :param image: Image path/url
    :return loaded_image: PIL Image
    """

    if image.startswith('http') or image.startswith('https'):
        image = Image.open(requests.get(image, stream=True).raw).convert('RGB')
    else:
        image = Image.open(image).convert('RGB')

    return image


def get_images(messages: list[Dict]) -> list:
    """
    Return loaded images from messages

    :param messages: A list of messages passed to the model
    :return images: A list of PIL Image objects.
    """
    # Collect image links/file locations mentioned in messages
    images = []
    for message in messages:
        if 'image' in message:
            if type(message['image']) == list:
                for img in message['image']:
                    images.append(img)
            else:
                images.append(message['image'])

    # Return None if no image is passed
    # Use AutoTokenizer to generate output and not AutoProcessor, as only text is passed.
    if not images:
        return None

    # Load Images
    loaded_images = []
    for img in images:
        image = load_image(img)
        loaded_images.append(image)

    return loaded_images


# Separate Input and Output generation for Idefics
# Input is required for context check
def generate_idefics_input(messages: list[Dict]):
    """
    Return inputs specific to the format of Idefics

    param messages: A list[Dict] type object passed to the backend containing 'role', 'content' and 'image'
    """
    # Create a list containing the prompt text and images specific to Idefics input
    # Refer - https://huggingface.co/HuggingFaceM4/idefics-80b-instruct

    # Use idefics_input as is for input to the model
    # Use idefics_text, that contains everything from idefics_input, apart from image_urls/loaded_image, used for context check
    idefics_input = []
    idefics_text = ""
    for m in messages:
        if m['role'] == 'user':
            idefics_input.append('\nUser: ' + m['content'])
            idefics_text += 'User: ' + m['content']
            if 'image' in m.keys():
                if type(m['image']) == list:  # Check if multiple images are passed, append accordingly
                    for im in m['image']:
                        loaded_im = load_image(im)
                        idefics_input.append(loaded_im)
                else:
                    idefics_input.append(m['image'])
            idefics_input.append('<end_of_utterance>')
            idefics_text += '<end_of_utterance>'
        elif m['role'] == 'assistant':
            idefics_input.append('\nAssistant: ' + m['content'])
            idefics_input.append('<end_of_utterance>')
            idefics_text += '\nAssistant: ' + m['content']
            idefics_text += '<end_of_utterance>'
    idefics_input.append('\nAssistant:')
    idefics_input = [idefics_input]

    return idefics_input, idefics_text


def generate_idefics_output(messages: list[Dict],
                            model: IdeficsForVisionText2Text,
                            processor: AutoProcessor,
                            max_tokens: int,
                            device) -> list[str]:
    """
    Return generated text from Idefics model

    param messages: A list[Dict] type object passed to the backend containing 'role', 'content' and 'image'
    param model: Idefics model
    param processor: Idefics processor
    param device: Processing device - cuda/CPU
    """
    idefics_input, _ = generate_idefics_input(messages=messages)
    inputs = processor(idefics_input, add_end_of_utterance_token=False, return_tensors="pt").to(device)

    # Generation args for Idefics
    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids,
                                   max_new_tokens=max_tokens)
    generated_text = processor.batch_decode(generated_ids)

    return generated_text


def check_multiple_image(messages: List[Dict]):
    """
    Return True if a single message contains multiple images
    param messages: A list[Dict] type object passed to the backend containing 'role', 'content' and 'image'
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
        return HuggingfaceMultimodalModel(model_spec)


class HuggingfaceMultimodalModel(backends.Model):

    def __init__(self, model_spec: backends.ModelSpec):
        super().__init__(model_spec)

        # Load instance variable used for evey model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_spec['model_type']
        self.model_name = model_spec['model_name']
        self.processor = load_processor(model_spec)
        self.multimodal_model = load_model(model_spec)
        self.split_prefix = model_spec['output_split_prefix']
        self.context_size = get_context_limit(model_spec)

        # Type cast model_spec to a Dictionary, for cleaner loading of variables
        model_spec_dict = vars(model_spec)
        # Load model specific instance variables
        self.template = model_spec_dict.get('custom_chat_template', None)
        self.cull = model_spec_dict.get('eos_to_cull', None)
        self.supports_multiple_images = model_spec_dict.get('supports_multiple_images', False)
        self.padding = model_spec_dict.get('padding', False)
        self.idefics = 'idefics' in model_spec['model_name']

    def generate_response(self, messages: List[Dict]) -> Tuple[Any, Any, str]:
        """
        :param messages: for example
                [
                    {"role": "user", "content": "Are there any clouds in the image? Answer with only "Yes" or "No"."},
                    {"role": "assistant", "content": "Yes"},
                    {"role": "user", "content": "This seems correct."},
                    {'role': 'user', 'content': 'Are there any chickens in the image? Answer with only "Yes" or "No".', 'image': 'games/cloudgame/resources/images/3.jpg'}
                ]
        :return: the continuation
        """
        # Check to see if game passes multiple images in a single turn
        # Proceed only if model supports multiple images, else return blanks for prompt, response and response_text
        has_multiple_images = check_multiple_image(messages=messages)
        if has_multiple_images and not self.supports_multiple_images:
            print(f"Multiple images not supported in a single turn for model {self.model_name}")
            return "", {"response": ""}, ""

        prompt_text = ""
        # Get input prompt by applying jinja template, if template is provided
        if self.template:
            template_str = self.template
            template = Template(template_str)
            prompt_text = template.render(messages=messages)

        # Get input prompt if model is of type IdeficsForVisionText2Text
        if self.idefics:
            _, prompt_text = generate_idefics_input(messages=messages)

        # Check context limit
        prompt_tokens = self.processor.tokenizer.tokenize(prompt_text)
        context_check = check_context_limit(self.context_size, prompt_tokens, max_new_tokens=self.get_max_tokens())
        if not context_check[0]:  # if context is exceeded, context_check[0] is False
            logger.info(f"Context token limit for {self.model_spec.model_name} exceeded: "
                        f"{context_check[1]}/{context_check[3]}")
            # fail gracefully:
            raise backends.ContextExceededError(f"Context token limit for {self.model_spec.model_name} exceeded",
                                                tokens_used=context_check[1], tokens_left=context_check[2],
                                                context_size=context_check[3])

        # Get a list of images [as input to the Processor]
        images = get_images(messages)

        # Generate the output
        if self.idefics:
            generated_text = generate_idefics_output(messages=messages,
                                                     model=self.multimodal_model,
                                                     processor=self.processor,
                                                     max_tokens=self.get_max_tokens(),
                                                     device=self.device)

        else:
            if not images:  # If no images are present in the history + current utterance, use tokenizer to get inputs
                inputs = self.processor.tokenizer(prompt_text, return_tensors="pt").to(self.device)
            else:
                inputs = self.processor(prompt_text, images=images, return_tensors="pt").to(self.device)
            model_output = self.multimodal_model.generate(**inputs, max_new_tokens=self.get_max_tokens())
            generated_text = self.processor.batch_decode(model_output, skip_special_tokens=True)

        prompt = {"inputs": prompt_text, "max_new_tokens": self.get_max_tokens(), "temperature": self.get_temperature()}

        # Store generated text
        response = {"response": generated_text}

        response_text = generated_text[0].split(self.split_prefix)[-1]  # Get the last assistant response
        if self.cull:
            rt_split = response_text.split(self.cull)  # Cull from End of String token
            response_text = rt_split[0]
        response_text = response_text.strip()

        return prompt, response, response_text