from typing import List, Dict, Tuple, Any
from retry import retry
import cohere
import backends
import json

logger = backends.get_logger(__name__)

MODEL_COMMAND = "command"
MODEL_COMMAND_LIGHT = "command-light"
SUPPORTED_MODELS = [MODEL_COMMAND, MODEL_COMMAND_LIGHT]

NAME = "cohere"


class Cohere(backends.Backend):
    def __init__(self):
        creds = backends.load_credentials(NAME)
        self.client = cohere.Client(creds[NAME]["api_key"])
        self.temperature: float = -1.

    @retry(tries=3, delay=0, logger=logger)
    def generate_response(self, messages: List[Dict], model: str) -> Tuple[str, Any, str]:
        """
        :param messages: for example
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        :param model: model name
        :return: the continuation
        """
        assert 0.0 <= self.temperature <= 1.0, "Temperature must be in [0.,1.]"
        chat_history = []

        # all other messages except the last one. It is passed to the API with the variable message.
        for message in messages[:-1]:
            m = {"user_name": "", "text": ""}
            m["text"] = message["content"]
            if message['role'] == 'assistant':
                m["user_name"] = "Chatbot"
            elif message['role'] == 'user':
                m["user_name"] = "User"
            chat_history.append(m)

        message = messages[-1]["content"]

        output = self.client.chat(
            message=message,
            model=model,
            chat_history=chat_history
        )

        response_text = output.text
        prompt = json.dumps({"message": message, "chat_history": chat_history})

        response = output.__dict__
        response.pop('client')
        response.pop('token_count')

        return prompt, response, response_text

    def supports(self, model_name: str):
        return model_name in SUPPORTED_MODELS
