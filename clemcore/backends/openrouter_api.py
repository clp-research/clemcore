import logging
from typing import List, Dict, Tuple, Any
from retry import retry
import json

from clemcore.backends.utils import ensure_messages_format, augment_response_object

import openai
import httpx

import clemcore.backends as backends

import clemcore.backends.openai_api as openai_api

logger = logging.getLogger(__name__)

NAME_DEPRECATED = "openrouter"  # for backwards compatibility: people have to adjust their key.json
NAME = "openrouter"


class OpenRouter(openai_api.OpenAI):
    """
    Backend class for accessing the OpenRouter remote API.

    Support for OpenRouter-specific request arguments.
    """

    def _make_api_client(self):
        try:
            creds = backends.load_credentials(NAME_DEPRECATED)
            _name = NAME_DEPRECATED
        except:
            creds = backends.load_credentials(NAME)  # new name: backend name and entry name match
            _name = NAME
        return openai.OpenAI(
            base_url = "https://openrouter.ai/api/v1",
            api_key = creds[_name]["api_key"],
            ### TO BE REVISED!!! (Famous last words...)
            ### The line below is needed because of
            ### issues with the certificates on our GPU server.
            http_client=httpx.Client(verify=False)
        )

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        """Get an OpenAI model instance based on a model specification.
        Args:
            model_spec: A ModelSpec instance specifying the model.
        Returns:
            An OpenAI model instance based on the passed model specification.
        """
        return OpenRouterModel(self.client, model_spec)


class OpenRouterModel(openai_api.OpenAIModel):
    """Model class accessing the OpenRouter remote API."""
    def __init__(self, client: openai.OpenAI, model_spec: backends.ModelSpec):
        """
        Args:
            client: An OpenAI library OpenAI client class.
            model_spec: A ModelSpec instance specifying the model.
        """
        super().__init__(client, model_spec)
        self.client = client

    @retry(tries=3, delay=90, logger=logger)
    @augment_response_object
    @ensure_messages_format
    def generate_response(self, messages: List[Dict]) -> Tuple[str, Any, str]:
        """Request a generated response from the OpenRouter remote API.
        Args:
            messages: A message history. For example:
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        Returns:
            The generated response message returned by the OpenRouter remote API.
        """
        prompt = self.encode_messages(messages)
        gen_kwargs = dict(model=self.model_spec.model_id,
                          messages=prompt,
                          temperature=self.temperature,
                          max_tokens=self.max_tokens)
        model_config = getattr(self.model_spec, "model_config", {})
        if 'reasoning_model' in model_config:
            if not self.temperature > 0:
                raise ValueError(f"For reasoning models temperature must be >0, but is {self.temperature}."
                                 f"Please use the -t option to set a temperature and try again.")
            # Note: For non-reasoning models max_tokens still accounts only for number of tokens (visible output)
            # sent to the user. However, for reasoning models the new max_completion_tokens must be used, which
            # accounts for both the reasoning tokens (which remain hidden on the openai backend) and the output.
            # https://platform.openai.com/docs/guides/reasoning/controlling-costs?api-mode=chat#controlling-costs
            logger.info("Ignoring max_tokens for reasoning models, because the argument is not supported")
            del gen_kwargs["max_tokens"]
        # Additional OpenRouter routing values are passed to the API as 'extra_body' in the request object
        # when using the OpenAI python API library
        # Pass model registry entry 'extra_body' to gen_kwargs
        if 'extra_body' in model_config:
            gen_kwargs['extra_body'] = model_config['extra_body']
        else:
            # Default to requesting fp8 quantization
            gen_kwargs['extra_body'] = {
                "provider": {
                    "quantizations": ["fp8"]
                }
            }
        api_response = self.client.chat.completions.create(**gen_kwargs)
        message = api_response.choices[0].message
        if message.role != "assistant":  # safety check
            raise AttributeError("Response message role is " + message.role + " but should be 'assistant'")
        response_text = message.content.strip()
        response = json.loads(api_response.json())

        return prompt, response, response_text