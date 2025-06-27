import logging
from typing import List, Dict, Tuple, Any
from retry import retry

import aleph_alpha_client
import anthropic
import clemcore.backends as backends
from clemcore.backends import ModelSpec, Model
from clemcore.backends.utils import ensure_messages_format

logger = logging.getLogger(__name__)

NAME = "alephalpha"


class AlephAlpha(backends.Backend):
    """Backend class for accessing the AlephAlpha remote API."""
    def __init__(self):
        creds = backends.load_credentials(NAME)
        self.client = aleph_alpha_client.Client(creds[NAME]["api_key"])

    def get_model_for(self, model_spec: ModelSpec) -> Model:
        """Get an AlephAlphaModel model instance based on a model specification.
        Args:
            model_spec: A ModelSpec instance specifying the model.
        Returns:
            An AlephAlphaModel model instance based on the passed model specification.
        """
        return AlephAlphaModel(self.client, model_spec)


class AlephAlphaModel(backends.Model):
    """Model class accessing the AlephAlpha remote API."""
    def __init__(self, client: aleph_alpha_client.Client, model_spec: ModelSpec):
        """
        Args:
            client: An AlephAlpha library Client class.
            model_spec: A ModelSpec instance specifying the model.
        """
        super().__init__(model_spec)
        self.client = client

    @retry(tries=3, delay=0, logger=logger)
    @ensure_messages_format
    def generate_response(self, messages: List[Dict]) -> Tuple[Any, Any, str]:
        """Request a generated response from the AlephAlpha remote API.
        Args:
            messages: A message history. For example:
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        Returns:
            The generated response message returned by the AlephAlpha remote API.
        """
        prompt_text = ''

        if 'control' in self.model_spec.model_id:
            for message in messages:
                content = message["content"]
                if message['role'] == 'assistant':
                    prompt_text += '### Response:' + content
                elif message['role'] == 'user':
                    prompt_text += '### Instruction:' + content
        else:

            for message in messages:
                if message['role'] == 'assistant':
                    prompt_text += f'{anthropic.AI_PROMPT} {message["content"]}'
                elif message['role'] == 'user':
                    prompt_text += f'{anthropic.HUMAN_PROMPT} {message["content"]}'

            prompt_text += anthropic.AI_PROMPT

        params = {
            "prompt": aleph_alpha_client.Prompt.from_text(prompt_text),
            "maximum_tokens": self.max_tokens,
            "stop_sequences": ['\n'],
            "temperature": self.temperature
        }

        request = aleph_alpha_client.CompletionRequest(**params)
        api_response = self.client.complete(request=request, model=self.model_spec.model_id)
        response = api_response.to_json()
        response_text = api_response.completions[0].completion.strip()

        prompt = params
        prompt['prompt'] = prompt_text

        return prompt, response, response_text
