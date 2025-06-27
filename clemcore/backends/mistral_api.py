import logging
from mistralai import Mistral as MistralClient
from typing import List, Dict, Tuple, Any
from retry import retry
import json
import clemcore.backends as backends
from clemcore.backends.utils import ensure_messages_format

logger = logging.getLogger(__name__)

NAME = "mistral"


class Mistral(backends.Backend):
    """Backend class for accessing the Mistral remote API."""
    def __init__(self):
        creds = backends.load_credentials(NAME)
        self.client = MistralClient(api_key=creds[NAME]["api_key"])

    def list_models(self) -> list:
        """List models available on the Mistral remote API.
        Returns:
            A list containing names of models available on the Mistral remote API.
        """
        models = self.client.models.list()
        names = [item.id for item in models.data]
        names = sorted(names)
        return names

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        """Get a Mistral model instance based on a model specification.
        Args:
            model_spec: A ModelSpec instance specifying the model.
        Returns:
            A Mistral model instance based on the passed model specification.
        """
        return MistralModel(self.client, model_spec)


class MistralModel(backends.Model):
    """Model class accessing the Mistral remote API."""
    def __init__(self, client: MistralClient, model_spec: backends.ModelSpec):
        """
        Args:
            client: An Mistral library MistralClient class.
            model_spec: A ModelSpec instance specifying the model.
        """
        super().__init__(model_spec)
        self.client = client

    @retry(tries=3, delay=0, logger=logger)
    @ensure_messages_format
    def generate_response(self, messages: List[Dict]) -> Tuple[str, Any, str]:
        """Request a generated response from the Mistral remote API.
        Args:
            messages: A message history. For example:
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        Returns:
            The generated response message returned by the Mistral remote API.
        """
        api_response = self.client.chat.complete(model=self.model_spec.model_id,
                                                 messages=messages,
                                                 temperature=self.temperature,
                                                 max_tokens=self.max_tokens)
        message = api_response.choices[0].message
        if message.role != "assistant":  # safety check
            raise AttributeError("Response message role is " + message.role + " but should be 'assistant'")
        response_text = message.content.strip()
        response = json.loads(api_response.model_dump_json())

        return messages, response, response_text
