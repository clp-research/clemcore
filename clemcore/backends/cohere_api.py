import logging
from typing import List, Dict, Tuple, Any
from retry import retry
import cohere

import clemcore.backends as backends
from clemcore.backends.utils import ensure_messages_format, augment_response_object

logger = logging.getLogger(__name__)


class Cohere(backends.RemoteBackend):
    """Backend class for accessing the Cohere remote API."""

    def _make_api_client(self):
        return cohere.ClientV2(self.key["api_key"])

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        """Get a Cohere model instance based on a model specification.
        Args:
            model_spec: A ModelSpec instance specifying the model.
        Returns:
            A Cohere model instance based on the passed model specification.
        """
        return CohereModel(self.client, model_spec)


class CohereModel(backends.Model):
    """Model class accessing the Cohere remote API."""

    def __init__(self, client: cohere.ClientV2, model_spec: backends.ModelSpec):
        """
        Args:
            client: A Cohere library Client class.
            model_spec: A ModelSpec instance specifying the model.
        """
        super().__init__(model_spec)
        self.client = client

    @retry(tries=3, delay=10, logger=logger)
    @augment_response_object
    @ensure_messages_format
    def generate_response(self, messages: List[Dict]) -> Tuple[Any, Any, str]:
        """Request a generated response from the Cohere remote API.
        Args:
            messages: A message history. For example:
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        Returns:
            The generated response message returned by the Cohere remote API.
        """
        result: cohere.V2ChatResponse = self.client.chat(
            messages=messages,  # type: ignore[arg-type]
            model=self.model_spec.model_id,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        if result.message.role != "assistant":  # safety check
            raise AttributeError("Response message role is " + result.message.role + " but should be 'assistant'")
        response_text = result.message.content[0].text
        response = result.model_dump(mode="json")
        return messages, response, response_text
