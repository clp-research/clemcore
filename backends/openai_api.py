from typing import List, Dict, Tuple, Any
from retry import retry

import json
import openai
import backends

logger = backends.get_logger(__name__)

MODEL_GPT_4_1106_PREVIEW = "gpt-4-1106-preview"
MODEL_GPT_4_0613 = "gpt-4-0613"
MODEL_GPT_4_0314 = "gpt-4-0314"
MODEL_GPT_35_1106 = "gpt-3.5-turbo-1106"
MODEL_GPT_35_0613 = "gpt-3.5-turbo-0613"
MODEL_GPT_3 = "text-davinci-003"
SUPPORTED_MODELS = [MODEL_GPT_4_0314, MODEL_GPT_4_0613, MODEL_GPT_4_1106_PREVIEW, MODEL_GPT_35_1106, MODEL_GPT_35_0613, MODEL_GPT_3]

NAME = "openai"

MAX_TOKENS = 100   # 2024-01-10, das: Should this be hardcoded???

class OpenAI(backends.Backend):

    def __init__(self):
        creds = backends.load_credentials(NAME)
        if "organisation" in creds[NAME]:
            self.client = openai.OpenAI(
                api_key=creds[NAME]["api_key"],
                organization=creds[NAME]["organisation"]
                )
        else:
            self.client = openai.OpenAI(
                api_key=creds[NAME]["api_key"]
                )
        self.chat_models: List = ["gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-4-0314", "gpt-4-0613", "gpt-4-1106-preview"]
        self.temperature: float = -1.

    def list_models(self):
        models = self.client.models.list()
        names = [item.id for item in models.data]
        names = sorted(names)
        return names
        # [print(n) for n in names]   # 2024-01-10: what was this? a side effect-only method?

    @retry(tries=3, delay=0, logger=logger)
    def generate_response(self, messages: List[Dict], model: str) -> Tuple[str, Any, str]:
        """
        :param messages: for example
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        :param model: chat-gpt for chat-completion, otherwise text completion
        :return: the continuation
        """
        assert 0.0 <= self.temperature <= 1.0, "Temperature must be in [0.,1.]"
        if model in self.chat_models:
            # chat completion
            prompt = messages
            api_response = self.client.chat.completions.create(model=model,
                                                          messages=prompt,
                                                          temperature=self.temperature,
                                                          max_tokens=MAX_TOKENS)
            message = api_response.choices[0].message
            if message.role != "assistant":  # safety check
                raise AttributeError("Response message role is " + message.role + " but should be 'assistant'")
            response_text = message.content.strip()
            response = json.loads(api_response.json())

        else:  # default (text completion)
            prompt = "\n".join([message["content"] for message in messages])
            api_response = self.client.completions.create(model=model, prompt=prompt,
                                                     temperature=self.temperature, max_tokens=100)
            response = json.loads(api_response.json())
            response_text = api_response.choices[0].text.strip()
        return prompt, response, response_text

    def supports(self, model_name: str):
        return model_name in SUPPORTED_MODELS
