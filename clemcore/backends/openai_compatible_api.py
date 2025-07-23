import logging

import openai
import httpx

import clemcore.backends as backends

import clemcore.backends.openai_api as openai_api

logger = logging.getLogger(__name__)

NAME_DEPRECATED = "generic_openai_compatible"  # for backwards compatibility: people have to adjust their key.json
NAME = "openai_compatible"


class GenericOpenAI(openai_api.OpenAI):
    """Generic backend class for accessing OpenAI-compatible remote APIs."""

    def _make_api_client(self):
        try:
            creds = backends.load_credentials(NAME_DEPRECATED)
            _name = NAME_DEPRECATED
        except:
            creds = backends.load_credentials(NAME)  # new name: backend name and entry name match
            _name = NAME
        return openai.OpenAI(
            base_url=creds[_name]["base_url"],
            api_key=creds[_name]["api_key"],
            ### TO BE REVISED!!! (Famous last words...)
            ### The line below is needed because of
            ### issues with the certificates on our GPU server.
            http_client=httpx.Client(verify=False)
        )
