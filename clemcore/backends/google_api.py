import logging
from typing import List, Dict, Tuple, Any, Union
from retry import retry
from google import genai
from google.genai import types
import os
import requests
import uuid
import tempfile
import imghdr

import clemcore.backends as backends
from clemcore.backends.utils import ensure_messages_format, augment_response_object

logger = logging.getLogger(__name__)


class Google(backends.RemoteBackend):
    """Backend class for accessing the Google remote API."""

    def _make_api_client(self):
        return genai.Client(api_key=self.key["api_key"])

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        """Get a Google model instance based on a model specification.
        Args:
            model_spec: A ModelSpec instance specifying the model.
        Returns:
            A Google model instance based on the passed model specification.
        """
        return GoogleModel(self.client, model_spec)


class GoogleModel(backends.Model):
    """Model class accessing the Google remote API."""

    def __init__(self, client: genai.Client, model_spec: backends.ModelSpec):
        """
        Args:
            client: A Google genai Client class.
            model_spec: A ModelSpec instance specifying the model.
        """
        super().__init__(model_spec)
        self.client = client

    def download_image(self, image_url) -> Union[str, None]:
        """Download an image from a URL.
        Args:
            image_url: The URL to download the image from.
        Returns:
            The file path to the downloaded image or None if the image could not be downloaded.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            unique_name = str(uuid.uuid4()) + '.jpg'
            file_path = os.path.join(temp_dir, unique_name)
            with open(file_path, 'wb') as file:
                file.write(response.content)
            return file_path
        except requests.RequestException as e:
            print(f"Failed to download {image_url}: {e}")
            return None

    def upload_file(self, file_path, mime_type):
        """Uploads the given file to Gemini.
        See https://ai.google.dev/gemini-api/docs/prompting_with_media
        Args:
            file_path: Path to the file to upload.
            mime_type: The mime type of the file to upload.
        Returns:
            The uploaded file reference.
        """
        file_ref = self.client.files.upload(file=file_path, config={"mime_type": mime_type})
        return file_ref

    def encode_images(self, images):
        """Encode images and upload them to Gemini allow sending them to the Google remote API.
        Args:
            images: Paths to the images to be encoded.
        Returns:
            A list of the uploaded file references.
        """
        image_parts = []
        for image_path in images:
            if image_path.startswith('http'):
                image_path = self.download_image(image_path)
            image_type = imghdr.what(image_path)
            file_ref = self.upload_file(image_path, 'image/' + image_type)
            image_parts.append(file_ref)
        return image_parts

    def encode_messages(self, messages):
        """Encode a message history containing images to allow sending it to the Google remote API.
        Args:
            messages: A message history. For example:
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        Returns:
            A list of Content objects for the Google API.
        """
        encoded_messages = []

        for message in messages:
            if message['role'] == 'system':
                # System messages are handled separately via system_instruction
                continue
            if message['role'] in ('assistant', 'model'):
                encoded_messages.append(types.Content(role="model", parts=[types.Part(text=message["content"])]))
            if message['role'] == 'user':
                parts = [types.Part(text=message["content"])]
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
                            image_parts = self.encode_images(message['image'])
                            for img in image_parts:
                                parts.append(types.Part(file_data=img))
                encoded_messages.append(types.Content(role="user", parts=parts))

        return encoded_messages

    def extract_system_instruction(self, messages):
        """Extract system instruction from messages.
        Args:
            messages: A message history.
        Returns:
            The system instruction string or None.
        """
        for message in messages:
            if message['role'] == 'system':
                return message['content']
        return None

    @retry(tries=3, delay=10, logger=logger)
    @augment_response_object
    @ensure_messages_format
    def generate_response(self, messages: List[Dict]) -> Tuple[Any, Any, str]:
        """Request a generated response from the Google remote API.
        Args:
            messages: A message history. For example:
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        Returns:
            The generated response message returned by the Google remote API.
        """
        encoded_messages = self.encode_messages(messages)
        system_instruction = self.extract_system_instruction(messages)

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                ),
            ],
        )

        if system_instruction:
            config.system_instruction = system_instruction

        if 'thinking_mode' in self.model_spec.model_config:
            """
            Thinking mode for Gemini 2.5 models uses thinking_budget parameter.
            https://ai.google.dev/gemini-api/docs/thinking
            Note: Gemini-3 models are not yet supported out of the box here
            """
            # Increase max_output_tokens to account for thinking budget (similar to Anthropic)
            config.max_output_tokens = 4096 + self.max_tokens
            config.thinking_config = types.ThinkingConfig(thinking_budget=4096)

        result: types.GenerateContentResponse = self.client.models.generate_content(
            model=self.model_spec.model_id,
            contents=encoded_messages,
            config=config
        )
        response_role = result.candidates[0].content.role
        if response_role != "model":  # safety check
            # Note: Google uses 'model' instead of 'assistant' in the response, but this is fine,
            # because the user will form a message with role=assistant from the returned response_text
            raise AttributeError("Response message role is " + response_role + " but should be 'model'")

        response_text = (result.text or "").strip()

        if not response_text:
            logger.warning("Google API response message content is None or empty, returning empty string.")

        response = result.model_dump(mode="json")
        return encoded_messages, response, response_text
