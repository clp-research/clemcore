from typing import List, Dict, Tuple, Any
from retry import retry

import json
import openai
import backends
import httpx

logger = backends.get_logger(__name__)

MAX_TOKENS = 100

# For this backend, it makes less sense to talk about "supported models" than for others,
# because what is supported depends very much on where this is pointed to.
# E.g., if I run FastChat on my local machine, I may have very different models available
# than if this is pointed to FastChat running on our cluster.
# Also, what is supported depends on what the server that this is pointed to happens to be
# serving at that moment.
# But anyway, hopefull we'll soon have a different method for selecting backends. 2024-01-10
SUPPORTED_MODELS = ["fsc-vicuna-13b-v1.5", "fsc-vicuna-33b-v1.3", "fsc-vicuna-7b-v1.5",
                    "fsc-openchat-3.5-0106"]

NAME = "generic_openai_compatible"


class GenericOpenAI(backends.Backend):

    def __init__(self, model_spec: backends.ModelSpec):
        super().__init__(model_spec)
        creds = backends.load_credentials(NAME)
        self.client = openai.OpenAI(
            base_url=creds[NAME]["base_url"],
            api_key=creds[NAME]["api_key"],
            ### TO BE REVISED!!! (Famous last words...)
            ### The line below is needed because of
            ### issues with the certificates on our GPU server.
            http_client=httpx.Client(verify=False)
            )

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
        model_id  = self.model_spec.model_id
        if self.model_spec.model_id.startswith('fsc-') or self.model_spec.model_id.startswith('lcp-'):
            model_id = self.model_spec.model_id[4:]

        prompt = messages
        api_response = self.client.chat.completions.create(model=model_id, messages=prompt,
                                                           temperature=self.get_temperature(),
                                                           max_tokens=MAX_TOKENS)
        message = api_response.choices[0].message
        if message.role != "assistant":  # safety check
            raise AttributeError("Response message role is " + message.role + " but should be 'assistant'")
        response_text = message.content.strip()
        response = json.loads(api_response.json())

        return prompt, response, response_text
