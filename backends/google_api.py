from typing import List, Dict, Tuple, Any
from retry import retry
import google.generativeai as genai
import backends
from backends.utils import ensure_messages_format
import json
import imghdr

logger = backends.get_logger(__name__)

NAME = "google"


class Google(backends.Backend):

    def __init__(self):
        creds = backends.load_credentials(NAME)
        genai.configure(api_key=creds[NAME]["api_key"])

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        return GoogleModel(model_spec)


class GoogleModel(backends.Model):

    def __init__(self, model_spec: backends.ModelSpec):
        super().__init__(model_spec)

    def upload_file(self, file_path, mime_type):
        """Uploads the given file to Gemini.

        See https://ai.google.dev/gemini-api/docs/prompting_with_media
        """
        file_url = genai.upload_file(file_path, mime_type=mime_type)
        return file_url

    def encode_images(self, images):
        image_parts = []

        for image_path in images:
            if image_path.startswith('http'):
                image_parts.append(image_path)
            else:
                image_type = imghdr.what(image_path)
                # upload to Gemini server
                file_url = self.upload_file(image_path, 'image/'+image_type)
                image_parts.append(file_url)
        return image_parts

    def encode_messages(self, messages):
        encoded_messages = []
        encoded_messages_for_logging = []

        for message in messages:
            if message['role'] == 'assistant':
                m = {"role": "model", "parts": [message["content"]]}
                m_for_logging = {"role": "model", "parts": [message["content"]]}
            elif message['role'] == 'user':
                m = {"role": "user", "parts": [message["content"]]}
                m_for_logging = {"role": "model", "parts": [message["content"]]}

                if self.model_spec.has_attr('supports_images'):
                    if "image" in message.keys():

                        if not self.model_spec.has_attr('support_multiple_images') and len(message['image']) > 1:
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

    @retry(tries=3, delay=5, logger=logger)
    @ensure_messages_format
    def generate_response(self, messages: List[Dict]) -> Tuple[str, Any, str]:
        """
        :param messages: for example
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        :return: the continuation
        """

        encoded_messages, encoded_messages_for_logging = self.encode_messages(messages)

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

        model = genai.GenerativeModel(
            model_name=self.model_spec.model_id,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )

        response = model.generate_content(
            contents=encoded_messages,
            generation_config=generation_config)

        response_text = response.text
        response = {"text": response_text}

        return encoded_messages_for_logging, response, response_text
