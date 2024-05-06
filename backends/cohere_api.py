from typing import List, Dict, Tuple, Any
from retry import retry
import cohere
import backends
from backends.utils import ensure_messages_format
import json

logger = backends.get_logger(__name__)

NAME = "cohere"


class Cohere(backends.Backend):

    def __init__(self):
        creds = backends.load_credentials(NAME)
        self.client = cohere.Client(creds[NAME]["api_key"])

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        return CohereModel(self.client, model_spec)


class CohereModel(backends.Model):

    def __init__(self, client: cohere.Client, model_spec: backends.ModelSpec):
        super().__init__(model_spec)
        self.client = client

    @retry(tries=3, delay=0, logger=logger)
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
        chat_history = []

        # all other messages except the last one. It is passed to the API with the variable message.
        for message in messages[:-1]:

            if message['role'] == 'assistant':
                m = {"role": "CHATBOT", "message": message["content"]}
                chat_history.append(m)
            elif message['role'] == 'user':
                m = {"role": "USER", "message": message["content"]}
                chat_history.append(m)

        message = messages[-1]["content"]

        output = self.client.chat(
            message=message,
            model=self.model_spec.model_id,
            chat_history=chat_history,
            temperature=self.get_temperature(),
            max_tokens = self.get_max_tokens()
        )

        response_text = output.text
        prompt = json.dumps({"message": message, "chat_history": chat_history})

        response = output.__dict__
        response.pop('client')
        response.pop('token_count')

        return prompt, response, response_text
