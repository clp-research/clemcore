import logging

import openai
import httpx

import clemcore.backends as backends

import clemcore.backends.openai_api as openai_api

logger = logging.getLogger(__name__)

NAME = "generic_openai_compatible"


class GenericOpenAI(openai_api.OpenAI):
    """Generic backend class for accessing OpenAI-compatible remote APIs."""

    def _make_api_client(self):
        creds = backends.load_credentials(NAME)
        return openai.OpenAI(
            base_url=creds[NAME]["base_url"],
            api_key=creds[NAME]["api_key"],
            ### TO BE REVISED!!! (Famous last words...)
            ### The line below is needed because of
            ### issues with the certificates on our GPU server.
            http_client=httpx.Client(verify=False)
        )
