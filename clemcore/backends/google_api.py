import logging
from typing import List, Dict, Tuple, Any, Union
from retry import retry
import google.generativeai as genai
import os
import requests
import uuid
import tempfile
import imghdr

import clemcore.backends as backends
from clemcore.backends.utils import ensure_messages_format

logger = logging.getLogger(__name__)

NAME = "google"


class Google(backends.Backend):
    """Backend class for accessing the Google remote API."""
    def __init__(self):
        creds = backends.load_credentials(NAME)
        genai.configure(api_key=creds[NAME]["api_key"])

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        """Get a Google model instance based on a model specification.
        Args:
            model_spec: A ModelSpec instance specifying the model.
        Returns:
            A Google model instance based on the passed model specification.
        """
        return GoogleModel(model_spec)


class GoogleModel(backends.Model):
    """Model class accessing the Google remote API."""
    def __init__(self, model_spec: backends.ModelSpec):
        """
        Args:
            model_spec: A ModelSpec instance specifying the model.
        """
        super().__init__(model_spec)

        self.model = genai.GenerativeModel(
            model_name=model_spec.model_id
        )

    def download_image(self, image_url) -> Union[str, None]:
        """Download an image from a URL.
        Args:
            image_url: The URL to download the image from.
        Returns:
            The file path to the downloaded image or None if the image could not be downloaded.
        """
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()

        try:
            # Send a GET request to the URL
            response = requests.get(image_url)
            response.raise_for_status()

            # Generate a unique file name
            unique_name = str(uuid.uuid4()) + '.jpg'
            file_path = os.path.join(temp_dir, unique_name)

            # Write the image content to a file
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
            The (URL of the) uploaded file.
        """
        file_url = genai.upload_file(file_path, mime_type=mime_type)
        return file_url

    def encode_images(self, images):
        """Encode images and upload them to Gemini allow sending them to the Google remote API.
        Args:
            images: Paths to the images to be encoded.
        Returns:
            A list of the (URLs of the) encoded and uploaded files.
        """
        image_parts = []

        for image_path in images:
            if image_path.startswith('http'):
                image_path = self.download_image(image_path)

            image_type = imghdr.what(image_path)
            # upload to Gemini server
            file_url = self.upload_file(image_path, 'image/'+image_type)
            image_parts.append(file_url)
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
            A tuple of the message history list with encoded images and the message history list for logging.
        """
        encoded_messages = []
        encoded_messages_for_logging = []

        for message in messages:
            if message['role'] == 'assistant':
                m = {"role": "model", "parts": [message["content"]]}
                m_for_logging = {"role": "model", "parts": [message["content"]]}
            elif message['role'] == 'user':
                m = {"role": "user", "parts": [message["content"]]}
                m_for_logging = {"role": "model", "parts": [message["content"]]}

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
                            image_parts = self.encode_images(message['image'])
                            for i in image_parts:
                                m["parts"].append(i)

                            # for logging purposes
                            m_for_logging["parts"].append(message["image"])

            encoded_messages.append(m)
            encoded_messages_for_logging.append(m_for_logging)
        return encoded_messages, encoded_messages_for_logging

    @retry(tries=10, delay=120, logger=logger)
    @ensure_messages_format
    def generate_response(self, messages: List[Dict]) -> Tuple[str, Any, str]:
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

        generation_config = {
            "temperature": self.get_temperature(),
            "max_output_tokens": self.get_max_tokens(),
            "response_mime_type": "text/plain",
        }

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
        ]

        encoded_messages, encoded_messages_for_logging = self.encode_messages(messages)

        response = self.model.generate_content(
            contents=encoded_messages,
            safety_settings=safety_settings,
            generation_config=generation_config)

        # print('Putting to sleep for 60 sec')
        # time.sleep(120)

        response_text = ''
        response_json = {}
        if response.parts:
            response_text = response.text.strip()
            response_json = {"text": response_text}

        if response_text == '':
            logger.error(
                f"The backend {self.model_spec.__getattribute__('model_id')} returned empty string!")

        return encoded_messages_for_logging, response_json, response_text
