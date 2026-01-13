import logging
from typing import List, Dict, Tuple, Any
from retry import retry
import anthropic
import json
import base64
import httpx
import imghdr

import clemcore.backends as backends
from clemcore.backends.utils import ensure_messages_format, augment_response_object

logger = logging.getLogger(__name__)


class Anthropic(backends.RemoteBackend):
    """Backend class for accessing the Anthropic remote API."""

    def _make_api_client(self):
        return anthropic.Anthropic(api_key=self.key["api_key"])

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

    def __init__(self, client: anthropic.Anthropic, model_spec: backends.ModelSpec):
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
        return image_data, "image/" + str(image_type)

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
                        f"The backend {self.model_spec.model_id} does not support multimodal inputs!")
                    raise Exception(
                        f"The backend {self.model_spec.model_id} does not support multimodal inputs!")
                if 'multimodality' in self.model_spec.model_config:
                    if "image" in message.keys():

                        if not self.model_spec.model_config['multimodality']['multiple_images'] and len(
                                message['image']) > 1:
                            logger.info(
                                f"The backend {self.model_spec.model_id} does not support multiple images!")
                            raise Exception(
                                f"The backend {self.model_spec.model_id} does not support multiple images!")
                        else:
                            # encode each image
                            for image in message['image']:
                                if isinstance(image, str) and image.startswith("data:image/"):
                                    # data URL: extract media type and payload
                                    header, b64 = image.split(',', 1)
                                    image_type = header.split(';')[0].split(':', 1)[1]
                                    encoded_image_data = b64
                                else:
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

    @retry(tries=3, delay=10, logger=logger)
    @augment_response_object
    @ensure_messages_format
    def generate_response(self, messages: List[Dict]) -> Tuple[Any, Any, str]:
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
        gen_kwargs = dict(
            messages=prompt,
            system=system_message,
            model=self.model_spec.model_id,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        content_index = 0
        if 'thinking_mode' in self.model_spec.model_config:
            """
            Feature compatibility:
            - Thinking isn't compatible with temperature or top_k modifications as well as forced tool use.
            - When thinking is enabled, you can set top_p to values between 1 and 0.95.
            - You cannot pre-fill responses when thinking is enabled.
            - Changes to the thinking budget invalidate cached prompt prefixes that include messages. 
            - However, cached system prompts and tool definitions will continue to work when thinking parameters change.
            https://platform.claude.com/docs/en/build-with-claude/extended-thinking
            """
            # set thinking token budget to 4K
            # max_tokens should be higher than 4K -> so we set to 4K + get_max_tokens()
            content_index = 1
            gen_kwargs["temperature"] = 1.  # todo: we need to use self.gen_args for this (user should decide)
            gen_kwargs["max_tokens"] = 4000 + self.max_tokens # todo: we need to use self.gen_args for this
            gen_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 4000}
        completion = self.client.messages.create(**gen_kwargs)
        if completion.role != "assistant":  # safety check
            raise AttributeError("Response message role is " + completion.role + " but should be 'assistant'")
        response_text = completion.content[content_index].text
        response = completion.model_dump(mode="json")
        return prompt, response, response_text
