from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from typing import List, Dict, Tuple, Any
from retry import retry
import json
import backends

logger = backends.get_logger(__name__)

MEDIUM = "mistral-medium"
TINY = "mistral-tiny"
SMALL = "mistral-small"
SUPPORTED_MODELS = [MEDIUM, TINY, SMALL]

NAME = "mistral"

MAX_TOKENS = 100

class Mistral(backends.Backend):

    def __init__(self):
        creds = backends.load_credentials(NAME)
        self.client = MistralClient(api_key=creds[NAME]["api_key"])
        self.temperature: float = -1.

    def list_models(self):
        models = self.client.models.list()
        names = [item.id for item in models.data]
        names = sorted(names)
        return names

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
        :param model: chat-gpt for chat-completion, otherwise text completion
        :return: the continuation
        """
        assert 0.0 <= self.temperature <= 1.0, "Temperature must be in [0.,1.]"

        prompt = []
        for m in messages:
            prompt.append(ChatMessage(role=m['role'], content=m['content']))
        api_response = self.client.chat(model=model,
                                                      messages=prompt,
                                                      temperature=self.temperature,
                                                      max_tokens=MAX_TOKENS)
        message = api_response.choices[0].message
        if message.role != "assistant":  # safety check
            raise AttributeError("Response message role is " + message.role + " but should be 'assistant'")
        response_text = message.content.strip()
        response = json.loads(api_response.model_dump_json())

        return messages, response, response_text

    def supports(self, model_name: str):
        return model_name in SUPPORTED_MODELS
