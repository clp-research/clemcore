import logging
from typing import List, Dict, Tuple, Any
from retry import retry
import json
import openai
import base64
import imghdr
import httpx

import clemcore.backends as backends
from clemcore.backends.utils import ensure_messages_format

logger = logging.getLogger(__name__)

NAME = "openai"


class OpenAI(backends.RemoteBackend):

    def _make_api_client(self):
        creds = backends.load_credentials(NAME)
        api_key = creds[NAME]["api_key"]
        organization = creds[NAME]["organisation"] if "organisation" in creds[NAME] else None
        return openai.OpenAI(api_key=api_key, organization=organization)

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        """Get an OpenAI model instance based on a model specification.
        Args:
            model_spec: A ModelSpec instance specifying the model.
        Returns:
            An OpenAI model instance based on the passed model specification.
        """
        return OpenAIModel(self.client, model_spec)


class OpenAIModel(backends.Model):
    """Model class accessing the OpenAI remote API."""
    def __init__(self, client: openai.OpenAI, model_spec: backends.ModelSpec):
        """
        Args:
            client: An OpenAI library OpenAI client class.
            model_spec: A ModelSpec instance specifying the model.
        """
        super().__init__(model_spec)
        self.client = client

    def encode_image(self, image_path):
        """Encode an image to allow sending it to the OpenAI remote API.
        Args:
            image_path: Path to the image to be encoded.
        Returns:
            A tuple with a bool, True if encoding was successful, False otherwise, the image encoded as base64 string
            and a string containing the image type.
        """
        if image_path.startswith('http'):
            image_bytes = httpx.get(image_path).content
            image_type = imghdr.what(None, image_bytes)
            return True, image_path, image_type
        with open(image_path, "rb") as image_file:
            image_type = imghdr.what(image_path)
            return False, base64.b64encode(image_file.read()).decode('utf-8'), 'image/'+str(image_type)

    def encode_messages(self, messages) -> list:
        """Encode a message history containing images to allow sending it to the OpenAI remote API.
        Args:
            messages: A message history. For example:
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        Returns:
            The message history list with encoded images.
        """
        encoded_messages = []

        for message in messages:
            if "image" not in message.keys():
                encoded_messages.append(message)
            else:
                this = {"role": message["role"],
                        "content": [
                            {
                                "type": "text",
                                "text": message["content"].replace(" <image> ", " ")
                            }
                        ]}

                if "image" in message.keys() and 'multimodality' not in self.model_spec.model_config:
                    logger.info(
                        f"The backend {self.model_spec.__getattribute__('model_id')} does not support multimodal inputs!")
                    raise Exception(
                        f"The backend {self.model_spec.__getattribute__('model_id')} does not support multimodal inputs!")

                if 'multimodality' in self.model_spec.model_config:
                    if "image" in message.keys():

                        if not self.model_spec['model_config']['multimodality']['multiple_images'] and len(message['image']) > 1:
                            logger.info(f"The backend {self.model_spec.__getattribute__('model_id')} does not support multiple images!")
                            raise Exception(f"The backend {self.model_spec.__getattribute__('model_id')} does not support multiple images!")
                        else:
                            # encode each image
                            for image in message['image']:
                                is_url, loaded, image_type = self.encode_image(image)
                                if is_url:
                                    this["content"].append(dict(type="image_url", image_url={
                                        "url": loaded
                                    }))
                                else:
                                    this["content"].append(dict(type="image_url", image_url={
                                        "url": f"data:{image_type};base64,{loaded}"
                                    }))
                encoded_messages.append(this)
        return encoded_messages

    @retry(tries=3, delay=90, logger=logger)
    @ensure_messages_format
    def generate_response(self, messages: List[Dict]) -> Tuple[str, Any, str]:
        """Request a generated response from the OpenAI remote API.
        Args:
            messages: A message history. For example:
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        Returns:
            The generated response message returned by the OpenAI remote API.
        """
        prompt = self.encode_messages(messages)

        if 'reasoning_model' in self.model_spec.model_config:
            api_response = self.client.chat.completions.create(model=self.model_spec.model_id,
                                                               messages=prompt,
                                                               temperature=1)
        else:
            api_response = self.client.chat.completions.create(model=self.model_spec.model_id,
                                                               messages=prompt,
                                                               temperature=self.get_temperature(),
                                                               max_tokens=self.get_max_tokens())
        message = api_response.choices[0].message
        if message.role != "assistant":  # safety check
            raise AttributeError("Response message role is " + message.role + " but should be 'assistant'")
        response_text = message.content.strip()
        response = json.loads(api_response.json())

        return prompt, response, response_text
