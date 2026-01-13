import logging
import mistralai
from typing import List, Dict, Tuple, Any
from retry import retry
import clemcore.backends as backends
from clemcore.backends.utils import ensure_messages_format, augment_response_object

logger = logging.getLogger(__name__)


class Mistral(backends.RemoteBackend):
    """Backend class for accessing the Mistral remote API."""

    def _make_api_client(self):
        return mistralai.Mistral(api_key=self.key["api_key"])

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

    def __init__(self, client: mistralai.Mistral, model_spec: backends.ModelSpec):
        """
        Args:
            client: An Mistral library MistralClient class.
            model_spec: A ModelSpec instance specifying the model.
        """
        super().__init__(model_spec)
        self.client = client

    @retry(tries=3, delay=10, logger=logger)
    @augment_response_object
    @ensure_messages_format
    def generate_response(self, messages: List[Dict]) -> Tuple[Any, Any, str]:
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
        api_response: mistralai.ChatCompletionResponse = self.client.chat.complete(
            model=self.model_spec.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        message = api_response.choices[0].message
        if message.role != "assistant":  # safety check
            raise AttributeError(f"Response message role '{message.role}' but should be 'assistant'")
        response_text = (message.content or "").strip()
        response = api_response.model_dump(mode="json")
        return messages, response, response_text
