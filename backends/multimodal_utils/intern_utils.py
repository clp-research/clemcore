# Individual inference methods for InternLM X-Composer 2.5 7B

import os
import shutil
from typing import Dict, List, Any, Union, Tuple

import requests
from PIL import Image
from io import BytesIO

import torch
import torchvision.transforms.functional as F
from transformers import AutoModel, AutoTokenizer

from backends.multimodal_utils.base_utils import BaseMLLM

IMG_CACHE_DIR = 'image_cache'


class InternMLLM(BaseMLLM):

    @staticmethod
    def custom_padding(image: Image.Image) -> Image.Image:
        """
        Apply custom white padding to an image and convert it to RGB if it is in RGBA mode.
        Images in mm_referencegame are in RGBA format, InternVL does not support this.

        :param image: A PIL Image object to be padded.
        :return: A padded PIL Image object, converted to RGB if necessary.
        """

        # Padding configuration
        # Set to 0 as its required to update the fill.
        padding_size = (0, 0, 0, 0)
        num_channels = len(image.getbands())

        # Determine fill color based on the number of channels
        if num_channels == 4:  # RGBA
            fill_color = (255, 255, 255, 255)
        elif num_channels == 3:  # RGB
            fill_color = (255, 255, 255)
        else:
            raise ValueError(f"Unsupported number of channels: {num_channels}")

        # Apply padding to the image
        padded_image = F.pad(image, padding_size, fill=fill_color)

        # Convert RGBA image to RGB
        if padded_image.mode == 'RGBA':
            padded_image = padded_image.convert('RGB')

        return padded_image

    @staticmethod
    def prepare_inputs(messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Prepare the inputs for the model, including the prompt, images, and conversation history.

        :param messages: A list of dictionaries, where each dictionary contains:
                         - 'role': The role of the message sender ('user' or 'assistant').
                         - 'content': The text content of the message.
                         - 'image': Optional; a single image URL (str) or a list of image URLs (List[str]).
        :param kwargs: Additional keyword arguments that may be used in the process.
        :return: A dictionary containing:
                 - 'prompt': The final prompt to be used by the model.
                 - 'images': A list of image URLs to be processed.
                 - 'processor_kwargs': A dictionary with 'history' (list of user-assistant message pairs). Passed to
                                       generate_outputs and get_tokens
        """
        conversation_history = []
        images = []
        image_counter = 0
        previous_user_message = ""

        for message in messages:
            if message['role'] == 'user':
                previous_user_message = message['content']
                if 'image' in message:
                    if isinstance(message['image'], str):
                        # Single image
                        image_counter += 1
                        previous_user_message = f"Image{image_counter} <ImageHere>; " + previous_user_message
                        images.append(message['image'])
                    elif isinstance(message['image'], list):
                        # List of images
                        for img in message['image']:
                            image_counter += 1
                            previous_user_message = f"Image{image_counter} <ImageHere>; " + previous_user_message
                            images.append(img)
                    else:
                        raise ValueError("Invalid image type in message - should be str or List[str]")

            elif message['role'] == 'assistant':
                # Append user and assistant messages in sequence
                conversation_history.append((previous_user_message, message['content']))

        return {
            "prompt": previous_user_message,
            "images": images,
            "output_kwargs": {"history": conversation_history, "device": kwargs.get('device')}
        }

    @staticmethod
    def get_tokens(prompt: str, handler: AutoTokenizer, **output_kwargs) -> List[str]:
        """
        Generate tokens for the given prompt and conversation history.

        :param prompt: The current prompt to be tokenized.
        :param handler: The tokenizer used for tokenizing the prompt and history.
        :param kwargs: Additional keyword arguments, expecting 'history' which is a list of tuples (user message, assistant response).

        :return: A list of tokens generated from the combined prompt and conversation history.
        """

        # Extract conversation history from kwargs
        history = output_kwargs.get("history")

        # Combine the prompt with the conversation history
        combined_text = prompt + "".join([user_msg + assistant_response for user_msg, assistant_response in history])

        # Tokenize the combined text
        tokens = handler.tokenize(combined_text)

        return tokens

    def preprocess_image(self, image_list: List[Union[str, Image.Image]]) -> List[str]:
        """
        Preprocess a list of images (by downloading them first, if URLs are provided), applying padding, and saving them locally.

        :param image_list: A list of image sources, where each item can be a local image path or a URL of the image.
        :return: A list of paths to the preprocessed and saved images.
        """

        # Define the path for temporary image storage
        img_dir = os.path.join(os.getcwd(), IMG_CACHE_DIR)
        if os.path.exists(img_dir):
            shutil.rmtree(img_dir)
        os.makedirs(img_dir)

        processed_image_paths = []
        for index, image_source in enumerate(image_list):
            save_path = os.path.join(img_dir, f'{index}.jpg')

            try:
                if isinstance(image_source, str) and image_source.startswith('http'):
                    # Download image from URL
                    response = requests.get(image_source)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content))
                elif isinstance(image_source, str):
                    # Load image from local path
                    image = Image.open(image_source)
                elif isinstance(image_source, Image.Image):
                    # Use the provided PIL Image object directly
                    image = image_source
                else:
                    raise ValueError(f"Invalid image source: {image_source}")

                # Apply custom padding to the image
                padded_image = self.custom_padding(image)

                # Save the preprocessed image
                padded_image.save(save_path)
                processed_image_paths.append(save_path)

            except Exception as e:
                raise ValueError(f"Failed to process image '{image_source}': {e}")

        return processed_image_paths

    def generate_outputs(self, prompt: str, images: List[str], model: AutoModel,
                         handler: AutoTokenizer, **output_kwargs) -> Tuple[Dict[str, Any], str]:
        """
        Generate model outputs given a prompt, images, and additional parameters.

        :param prompt: The text prompt to be used for generating the response.
        :param images: A list of image URLs or paths to be included in the model's input.
        :param model: The model used for generating the output. This should be compatible with InternLM type models.
        :param handler: The tokenizer used to preprocess the prompt and handle the input.
        :param kwargs: Additional keyword arguments for the model, expected to include 'history'.
        :return:
             - response (Dict[str, Any]): The raw output from the model, formatted as a dictionary.
             - response_text (str): The decoded text response generated by the model.
        """

        # Ensure image preprocessing
        # Works for mm_mapworld, but is required for matchit (download images) and mm_referencegame (convert RGBA to RBG)
        processed_image_paths = self.preprocess_image(images)

        # Retrieve conversation history from kwargs
        history = output_kwargs.get("history")
        device = output_kwargs.get("device")

        # Disable gradient calculation for inference
        torch.set_grad_enabled(False)

        # Prepare model for inference
        model = model.to(device).eval()
        model.tokenizer = handler

        try:
            with torch.autocast(device_type=device, dtype=torch.float16):
                gen_text, _ = model.chat(
                    handler,
                    prompt,
                    processed_image_paths,
                    do_sample=False,
                    num_beams=3,
                    top_p=1,  # Unset top_p, to avoid a UserWarning - conflict between do_sample and top_p
                    history=history,
                    use_meta=True
                )

            # Process and clean response text
            response_text = gen_text.strip()

            # Format response
            response = {"response": gen_text}

        except RuntimeError as e:
            raise RuntimeError(f"Model execution failed: {e}")

        finally:
            # Clean up cached images
            shutil.rmtree(IMG_CACHE_DIR, ignore_errors=True)

        return response, response_text