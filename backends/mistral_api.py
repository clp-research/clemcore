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

    def __init__(self, model_spec: backends.ModelSpec):
        super().__init__(model_spec)
        creds = backends.load_credentials(NAME)
        self.client = MistralClient(api_key=creds[NAME]["api_key"])

    def list_models(self):
        models = self.client.models.list()
        names = [item.id for item in models.data]
        names = sorted(names)
        return names

    @retry(tries=3, delay=0, logger=logger)
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

        prompt = []
        for m in messages:
            prompt.append(ChatMessage(role=m['role'], content=m['content']))
        api_response = self.client.chat(model=self.model_spec.model_id,
                                                      messages=prompt,
                                                      temperature=self.get_temperature(),
                                                      max_tokens=MAX_TOKENS)
        message = api_response.choices[0].message
        if message.role != "assistant":  # safety check
            raise AttributeError("Response message role is " + message.role + " but should be 'assistant'")
        response_text = message.content.strip()
        response = json.loads(api_response.model_dump_json())

        return messages, response, response_text
