import logging
from typing import List, Dict, Tuple, Any
from retry import retry
import anthropic
import json
import base64
import httpx
import imghdr

import clemcore.backends as backends
from clemcore.backends.utils import ensure_messages_format

logger = logging.getLogger(__name__)

NAME = "anthropic"


class Anthropic(backends.Backend):
    """Backend class for accessing the Anthropic remote API."""
    def __init__(self):
        creds = backends.load_credentials(NAME)
        self.client = anthropic.Anthropic(api_key=creds[NAME]["api_key"])

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        """Get an Anthropic model instance based on a model specification.
        Args:
            model_spec: A ModelSpec instance specifying the model.
        Returns:
            An Anthropic model instance based on the passed model specification.
        """
        return AnthropicModel(self.client, model_spec)


class AnthropicModel(backends.Model):
    """Model class accessing the Anthropic remote API."""
    def __init__(self, client: anthropic.Client, model_spec: backends.ModelSpec):
        """
        Args:
            client: An Anthropic library Client class.
            model_spec: A ModelSpec instance specifying the model.
        """
        super().__init__(model_spec)
        self.client = client

    def encode_image(self, image_path) -> Tuple[str, str]:
        """Encode an image to allow sending it to the Anthropic remote API.
        Args:
            image_path: Path to the image to be encoded.
        Returns:
            A tuple of the image encoded as base64 string and a string containing the image type.
        """
        if image_path.startswith('http'):
            image_bytes = httpx.get(image_path).content
        else:
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
        image_data = base64.b64encode(image_bytes).decode("utf-8")
        image_type = imghdr.what(None, image_bytes)
        return image_data, "image/"+str(image_type)

    def encode_messages(self, messages) -> Tuple[List, str]:
        """Encode a message history containing images to allow sending it to the Anthropic remote API.
        Args:
            messages: A message history. For example:
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        Returns:
            A tuple of the message history list with encoded images and the system message as string.
        """
        encoded_messages = []
        system_message = ''

        for message in messages:
            if message["role"] == "system":
                system_message = message["content"]
            else:

                content = list()
                content.append({
                    "type": "text",
                    "text": message["content"]
                })

                if "image" in message.keys() and 'multimodality' not in self.model_spec.model_config:
                    logger.info(
                        f"The backend {self.model_spec.__getattribute__('model_id')} does not support multimodal inputs!")
                    raise Exception(
                        f"The backend {self.model_spec.__getattribute__('model_id')} does not support multimodal inputs!")

                if 'multimodality' in self.model_spec.model_config:
                    if "image" in message.keys():

                        if not self.model_spec['model_config']['multimodality']['multiple_images'] and len(message['image']) > 1:
                            logger.info(
                                f"The backend {self.model_spec.__getattribute__('model_id')} does not support multiple images!")
                            raise Exception(
                                f"The backend {self.model_spec.__getattribute__('model_id')} does not support multiple images!")
                        else:
                            # encode each image
                            for image in message['image']:
                                encoded_image_data, image_type = self.encode_image(image)
                                content.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": image_type,
                                        "data": encoded_image_data,
                                    }
                                })

                claude_message = {
                    "role": message["role"],
                    "content": content
                }
                encoded_messages.append(claude_message)

        return encoded_messages, system_message

    @retry(tries=3, delay=0, logger=logger)
    @ensure_messages_format
    def generate_response(self, messages: List[Dict]) -> Tuple[str, Any, str]:
        """Request a generated response from the Anthropic remote API.
        Args:
            messages: A message history. For example:
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        Returns:
            The generated response message returned by the Anthropic remote API.
        """
        prompt, system_message = self.encode_messages(messages)

        if 'thinking_mode' not in self.model_spec.model_config:

            completion = self.client.messages.create(
                messages=prompt,
                system=system_message,
                model=self.model_spec.model_id,
                temperature=self.get_temperature(),
                max_tokens=self.get_max_tokens()
            )

            json_output = completion.model_dump_json()
            response = json.loads(json_output)
            response_text = completion.content[0].text

        else:
            # set thinking token budget to 4K
            # max_tokens should be higher than 4K -> so we set to 4K + get_max_tokens()
            completion = self.client.messages.create(
                messages=prompt,
                system=system_message,
                model=self.model_spec.model_id,
                max_tokens=4000 + self.get_max_tokens(),
                thinking={
                    "type": "enabled",
                    "budget_tokens": 4000
                },
            )

            json_output = completion.model_dump_json()
            response = json.loads(json_output)
            response_text = completion.content[1].text

        return prompt, response, response_text
