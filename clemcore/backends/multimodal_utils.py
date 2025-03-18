"""
Util functions for multimodal models.
"""

from typing import List, Tuple
import math
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers.image_utils import load_image
import requests
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

"""
##### INTERNVL2 TYPE MODELS #####
"""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def generate_history_internvl2(messages: List[str]) -> Tuple[List[Tuple], str]:
    """
    Separates the history and query from the list of messages in the current game instance.
    Compatible with InternVL2 and Nvidia NVLM models.

    Args:
        messages: A list containing user messages, system messages or assistant responses.

    Returns:
        A list of tuples containing the history and a user message string, passed to the model in the current game instance.

    Raises:
        ValueError: if msg['role'] is different than 'user', 'system', or 'assistant'.
    """

    history = []
    for msg in messages:
        if msg['role'] == 'system':
            continue # Skip the system message, Not passed to the model. Ref - https://huggingface.co/OpenGVLab/InternVL2-40B
        elif msg['role'] == 'user':
            if 'image' in msg:
                user_message = f"<image>\n{msg['content']}" # Add <image> token if image is passed in this instance.
            else:
                user_message = msg['content']
        elif msg['role'] == 'assistant':
            history.append((user_message, msg['content']))
        else:
            raise ValueError(f"Invalid role: {msg['role']}. Expected 'user', 'system', or 'assistant'.")

    return history, user_message


def split_model(model_name):
    """
    Splits the model across available GPUs based on the model name.

    Args:
        model_name (str): The name of the model to be split.
                          Expected values include 'InternVL2-1B', 'InternVL2-2B',
                          'InternVL2-4B', 'InternVL2-8B', 'InternVL2-26B',
                          'InternVL2-40B', 'InternVL2-Llama3-76B'.

    Returns:
        dict: A mapping of model layers to GPU indices.
    """
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def build_transform(input_size):
    """Builds a transformation pipeline for image preprocessing.

    Args:
        input_size (int): The size to which the image will be resized.

    Returns:
        torchvision.transforms.Compose: A composed transform for the image.
    """
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Finds the closest aspect ratio from a set of target ratios.

    Args:
        aspect_ratio (float): The aspect ratio of the original image.
        target_ratios (list): A list of target aspect ratios.
        width (int): The width of the original image.
        height (int): The height of the original image.
        image_size (int): The size of the image for comparison.

    Returns:
        tuple: The best aspect ratio found.
    """
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Processes the image to fit the closest aspect ratio and splits it into blocks.

    Args:
        image (PIL.Image): The image to be processed.
        min_num (int): Minimum number of blocks.
        max_num (int): Maximum number of blocks.
        image_size (int): The size of the image.
        use_thumbnail (bool): Whether to create a thumbnail.

    Returns:
        list: A list of processed image blocks.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_internvl2_image(image_file, input_size=448, max_num=12):
    """Loads an image file and applies transformations.

    Args:
        image_file (str): The path to the image file.
        input_size (int): The size to which the image will be resized.
        max_num (int): Maximum number of blocks to create.

    Returns:
        torch.Tensor: A tensor containing the pixel values of the processed images.
    """
    if image_file.startswith("http"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_internvl2_image(messages: List[str], device: str):
    """
    Extracts the last user message containing image data and loads the corresponding images.

    Args:
        messages (List[str]): A list of message dictionaries containing user, system, and assistant messages.
        device (str): The device to which the image tensors will be moved (e.g., 'cuda' or 'cpu').

    Returns:
        torch.Tensor: A tensor containing the pixel values of the processed images.

    Raises:
        ValueError: If no user message is found.
    """
    # Get last user message
    last_user_message = next((msg for msg in reversed(messages) if msg['role'] == 'user'), None)

    if last_user_message is None:
        raise ValueError("No user message found in the provided messages.")

    if 'image' in last_user_message:
        # Load all images and concatenate them into a single tensor
        pixel_values = torch.cat(
            [load_internvl2_image(img, max_num=12).to(torch.bfloat16).to(device) for img in last_user_message['image']]
        , dim=0)
    else:
        pixel_values = None
        logger.info("*" * 50 + "  Pixel Values not found  " + "*" * 50)

    return pixel_values

def generate_internvl2_prompt_text(messages: List[str], **prompt_kwargs) -> str:
    """Generates input text for the InternVL2 model from a list of messages.

    Args:
        messages (List[str]): A list of message dictionaries containing user, system, and assistant messages.

    Returns:
        str: The concatenated prompt text generated from the message history and the last user question.
    """
    prompt_text = ""
    history, question = generate_history_internvl2(messages=messages)
    if history:
        for t in history:
            prompt_text += t[0] + t[1]
    prompt_text += question
    return prompt_text

def generate_internvl2_response(**response_kwargs) -> str:
    """Generates a response from the InternVL2 model based on the provided messages and configuration.

    Args:
        **response_kwargs: A dictionary containing the following keys:
            - messages (List[str]): A list of message dictionaries.
            - device (str): The device to which the image tensors will be moved (e.g., 'cuda' or 'cpu').
            - max_tokens (int): The maximum number of tokens to generate.
            - model: The model instance used for generating responses.
            - processor: The processor instance used for processing images.

    Returns:
        str: The generated response from the model.

    Raises:
        RuntimeError: If the model fails to generate a response.
    """
    messages = response_kwargs['messages']
    device = response_kwargs['device']
    max_tokens = response_kwargs['max_tokens']
    model = response_kwargs['model']
    processor = response_kwargs['processor']
    do_sample = response_kwargs['do_sample']

    images = get_internvl2_image(messages=messages, device=device)
    history, question = generate_history_internvl2(messages=messages)

    if not history:
        history = None
    generation_config = dict(max_new_tokens=max_tokens, do_sample=do_sample)
    try:
        generated_response, _ = model.chat(processor, images, question, generation_config,
                                                     history=history, return_history=True)

    except Exception as e:
        raise RuntimeError("Failed to generate response from the model.") from e



    return generated_response


"""
##### LLAVA TYPE MODELS #####
Compatible models - LLaVA 1.5, LLaVA 1.6, Idefics3
"""

def generate_llava_messages(messages: List[str]) -> Tuple[List, List]:
    """Generates LLAVA messages and image paths from a list of messages.

    Args:
        messages (List[str]): A list of message dictionaries containing user, system, and assistant messages.

    Returns:
        Tuple[List, List]: A tuple containing:
            - A list of formatted LLAVA messages.
            - A list of image paths extracted from the messages.
    """
    llava_messages = []
    image_paths = []
    for message in messages:
        message_dict = {}
        message_dict['content'] = []

        if message['role'] == 'user':
            message_dict['role'] = 'user'
            if 'image' in message:
                if isinstance(message['image'], str):
                    # Single image
                    message_dict['content'].append({"type": "image"})
                    image_paths.append(message['image'])
                elif isinstance(message['image'], list):
                    # List of images
                    for img in message['image']:
                        message_dict['content'].append({"type": "image"})
                        image_paths.append(img)
                else:
                    raise ValueError("Invalid image type in message - should be str or List[str]")

            # Add user text message at the end
            message_dict['content'].append({"type": "text", "text": message['content']})
            llava_messages.append(message_dict)

        elif message['role'] == 'assistant':
            message_dict['role'] = 'assistant'
            message_dict['content'].append({"type": "text", "text": message['content']})
            llava_messages.append(message_dict)

        elif message['role'] == 'system':
            continue # Skip System message
        else:
            raise ValueError(f"Invalid role: {message_dict['role']}. Expected 'user', 'system', or 'assistant'.")

    last_user_message = llava_messages[-1]
    if last_user_message['role'] == 'user':
        content = last_user_message['content']
        contains_image = False
        for val in content:
            if val["type"] == "image":
                contains_image = True

        if not contains_image: # Pass a blank image
            blank_image = Image.new('RGB', (128, 128), color='white')
            image_paths.append(blank_image)
            llava_messages[-1]['content'].append({"type": "image"})

    return llava_messages, image_paths

def generate_llava_prompt_text(messages: List[str], **prompt_kwargs) -> str:
    """Generates a prompt text for LLAVA from a list of messages.

    Args:
        messages (List[str]): A list of message dictionaries containing user, system, and assistant messages.
        **prompt_kwargs: Additional keyword arguments for processing.

    Returns:
        str: The generated prompt text for LLAVA.
    """
    llava_messages, _ = generate_llava_messages(messages=messages)
    processor = prompt_kwargs['processor']
    prompt = processor.apply_chat_template(llava_messages, add_generation_prompt=True)

    return prompt

def generate_llava_response(**response_kwargs) -> str:
    """Generates a response from the LLAVA model based on the provided messages and configuration.

    Args:
        **response_kwargs: A dictionary containing the following keys:
            - messages (List[str]): A list of message dictionaries.
            - device (str): The device to which the image tensors will be moved (e.g., 'cuda' or 'cpu').
            - max_tokens (int): The maximum number of tokens to generate.
            - model: The model instance used for generating responses.
            - processor: The processor instance used for processing images.

    Returns:
        str: The generated response from the LLAVA model.

    Raises:
        RuntimeError: If the model fails to generate a response.
    """
    messages = response_kwargs['messages']
    device = response_kwargs['device']
    max_tokens = response_kwargs['max_tokens']
    model = response_kwargs['model']
    processor = response_kwargs['processor']
    do_sample = response_kwargs['do_sample']

    llava_messages, image_paths = generate_llava_messages(messages=messages)
    prompt = processor.apply_chat_template(llava_messages, add_generation_prompt=True)

    # Process images
    processed_images = []
    for image in image_paths:
        if type(image) == str:
            processed_images.append(load_image(image))
        else:
            processed_images.append(image)

    inputs = processor(images=processed_images, text=prompt, return_tensors='pt').to(device)

    try:
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=do_sample)
        response = processor.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        raise RuntimeError("Failed to generate response from the LLAVA model.") from e

    return response

"""
##### GEMMA TYPE MODELS #####
"""

def generate_gemma_messages(messages: List[str]) -> Tuple[List, List]:

    gemma_message = []
    for msg in messages:
        gemma_msg = {"role": msg['role']}
        content_list = [{"type": "text", "text": msg['content']}]
        if 'image' in msg:
            if isinstance(msg['image'], str):
                # Single image
                content_list.append({"type": "image", "image": msg['image']})
            elif isinstance(msg['image'], list):
                # List of images
                for img in msg['image']:
                    content_list.append({"type": "image", "image": img})
            else:
                raise ValueError("Invalid image type in message - should be str or List[str]")
        gemma_msg['content'] = content_list


        gemma_message.append(gemma_msg)

    return gemma_message

def generate_gemma_prompt_text(messages: List[str], **prompt_kwargs) -> str:

    gemma_message = generate_gemma_messages(messages)

    processor = prompt_kwargs['processor']

    prompt_text = processor.apply_chat_template(
                gemma_message, add_generation_prompt=True, tokenize=False,
                return_dict=True, return_tensors="pt"
            )

    return prompt_text


def generate_gemma_response(**response_kwargs) -> str:
    """Generates a response from the LLAVA model based on the provided messages and configuration.

    Args:
        **response_kwargs: A dictionary containing the following keys:
            - messages (List[str]): A list of message dictionaries.
            - device (str): The device to which the image tensors will be moved (e.g., 'cuda' or 'cpu').
            - max_tokens (int): The maximum number of tokens to generate.
            - model: The model instance used for generating responses.
            - processor: The processor instance used for processing images.

    Returns:
        str: The generated response from the LLAVA model.

    Raises:
        RuntimeError: If the model fails to generate a response.
    """

    
    messages = response_kwargs['messages']
    device = response_kwargs['device']
    max_tokens = response_kwargs['max_tokens']
    model = response_kwargs['model']
    processor = response_kwargs['processor']
    do_sample = response_kwargs['do_sample']

    gemma_messages = generate_gemma_messages(messages)

    
    inputs = processor.apply_chat_template(
                gemma_messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=do_sample)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)

    return decoded

"""
##### IDEFICS TYPE MODELS #####
"""

def generate_idefics_prompt_text(messages: List[str], **prompt_kwargs) -> str:
    """Generates a prompt text from a list of messages for the IDEFICS model.

    Args:
        messages (List[str]): A list of message dictionaries containing user, system, and assistant messages.
        **prompt_kwargs: Additional keyword arguments for processing.

    Returns:
        str: The concatenated prompt text generated from the message history.
    """
    prompt_text = ""
    for msg in messages:
        if msg['role'] == 'system':
            continue  # Skip system message. Ref - https://huggingface.co/HuggingFaceM4/idefics-9b-instruct
        elif msg['role'] == 'user':
            prompt_text += f" User: {msg['content']} "
            if 'image' in msg:
                if len(msg['image']) > 1:
                    for img in msg['image']:
                        prompt_text += img
                else:
                    prompt_text += msg['image'][0]
            prompt_text += "<end_of_utterance>"
        elif msg['role'] == 'assistant':
            prompt_text += f" Assistant: {msg['content']} <end_of_utterance>"
        else:
            raise ValueError(f"Invalid role: {msg['role']}. Expected 'user', 'system', or 'assistant'.")

    return prompt_text

def generate_idefics_response(**response_kwargs) -> str:
    """Generates a response from the IDEFICS model based on the provided messages and configuration.

    Args:
        **response_kwargs: A dictionary containing the following keys:
            - messages (List[str]): A list of message dictionaries.
            - device (str): The device to which the image tensors will be moved (e.g., 'cuda' or 'cpu').
            - max_tokens (int): The maximum number of tokens to generate.
            - model: The model instance used for generating responses.
            - processor: The processor instance used for processing images.

    Returns:
        str: The generated response from the IDEFICS model.

    Raises:
        RuntimeError: If the model fails to generate a response.
    """
    messages = response_kwargs['messages']
    device = response_kwargs['device']
    max_tokens = response_kwargs['max_tokens']
    model = response_kwargs['model']
    processor = response_kwargs['processor']

    input_messages = []
    for msg in messages:
        if msg['role'] == 'system':
            continue  # Skip system message. Ref - https://huggingface.co/HuggingFaceM4/idefics-9b-instruct
        elif msg['role'] == 'user':
            input_messages.append(f"\nUser: {msg['content']}")
            if 'image' in msg:
                if len(msg['image']) > 1:
                    for img in msg['image']:
                        loaded_image = load_image(img)
                        input_messages.append(loaded_image)
                else:
                    loaded_image = load_image(msg['image'][0])
                    input_messages.append(loaded_image)
            input_messages.append("<end_of_utterance>")
        elif msg['role'] == 'assistant':
            input_messages.append(f"\nAssistant: {msg['content']} <end_of_utterance>")
        else:
            raise ValueError(f"Invalid role: {msg['role']}. Expected 'user', 'system', or 'assistant'.")

    # --batched mode
    inputs = processor(input_messages, add_end_of_utterance_token=False, return_tensors="pt").to(device)

    # Generation args
    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    try:
        generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=max_tokens)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    except Exception as e:
        raise RuntimeError("Failed to generate response from the IDEFICS model.") from e

    return generated_text[0]


