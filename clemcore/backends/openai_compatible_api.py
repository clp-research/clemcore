import logging

import openai
import httpx

import clemcore.backends as backends

import clemcore.backends.openai_api as openai_api

logger = logging.getLogger(__name__)


class GenericOpenAI(openai_api.OpenAI):
    """Generic backend class for accessing OpenAI-compatible remote APIs."""

    def __init__(self):
        super().__init__(key_name="openai_compatible")

    def _make_api_client(self):
        return openai.OpenAI(
            base_url=self.key["base_url"],
            api_key=self.key["api_key"],
            ### TO BE REVISED!!! (Famous last words...)
            ### The line below is needed because of
            ### issues with the certificates on our GPU server.
            http_client=httpx.Client(verify=False)
        )
