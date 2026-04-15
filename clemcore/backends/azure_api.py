import logging
from typing import List, Dict, Tuple, Any
from retry import retry
import json
import openai
import base64
import imghdr
import httpx

import clemcore.backends as backends
from clemcore.backends.utils import ensure_messages_format, augment_response_object
from anthropic import AnthropicFoundry

logger = logging.getLogger(__name__)

NAME = "azure"
GPT_KEY_NAME = "azure_gpt"
CLAUDE_KEY_NAME = "azure_claude"


class AzureBackend(backends.RemoteBackend):

    def __init__(self):
        key_registry = backends.KeyRegistry.from_json()

        self.key_name = NAME
        azure_key = key_registry.get_key_for(GPT_KEY_NAME)
        self.key = azure_key
        self.openai_client = openai.OpenAI(
            api_key=azure_key["api_key"],
            base_url=azure_key["base_url"]
        )

        claude_key = key_registry.get_key_for(CLAUDE_KEY_NAME)
        self.claude_client = AnthropicFoundry(
            api_key=claude_key["api_key"],
            base_url=claude_key["base_url"]
        )

        # RemoteBackend expects self.client; point it at the OpenAI client
        self.client = self.openai_client

    def _make_api_client(self):
        # Clients are initialised directly in __init__; this satisfies the abstract method.
        pass

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        if "claude" in model_spec.model_id.lower():
            return AzureClaudeModel(self.claude_client, model_spec)
        return AzureOpenAIModel(self.openai_client, model_spec)


# ---------------------------------------------------------------------------
# Shared image-encoding helper (identical logic for both model families)
# ---------------------------------------------------------------------------

def encode_image(image_path):
    if image_path.startswith('http'):
        image_bytes = httpx.get(image_path).content
        image_type = imghdr.what(None, image_bytes)
        return True, image_path, image_type
    with open(image_path, "rb") as image_file:
        image_type = imghdr.what(image_path)
        return False, base64.b64encode(image_file.read()).decode('utf-8'), 'image/' + str(image_type)


# ---------------------------------------------------------------------------
# GPT / OpenAI model
# ---------------------------------------------------------------------------

class AzureOpenAIModel(backends.Model):

    def __init__(self, client: openai.OpenAI, model_spec: backends.ModelSpec):
        super().__init__(model_spec)
        self.client = client

    def encode_messages(self, messages) -> list:
        encoded_messages = []
        for message in messages:
            if "image" not in message.keys():
                encoded_messages.append(message)
            else:
                this = {"role": message["role"],
                        "content": [{"type": "text",
                                     "text": message["content"].replace(" <image> ", " ")}]}

                if 'multimodality' not in self.model_spec.model_config:
                    logger.info(
                        f"The backend {self.model_spec.model_id} does not support multimodal inputs!")
                    raise Exception(
                        f"The backend {self.model_spec.model_id} does not support multimodal inputs!")

                if not self.model_spec['model_config']['multimodality']['multiple_images'] \
                        and len(message['image']) > 1:
                    logger.info(f"The backend {self.model_spec.model_id} does not support multiple images!")
                    raise Exception(f"The backend {self.model_spec.model_id} does not support multiple images!")

                for image in message['image']:
                    is_url, loaded, image_type = encode_image(image)
                    if is_url:
                        this["content"].append(dict(type="image_url", image_url={"url": loaded}))
                    else:
                        this["content"].append(dict(type="image_url",
                                                    image_url={"url": f"data:{image_type};base64,{loaded}"}))
                encoded_messages.append(this)
        return encoded_messages

    @retry(tries=3, delay=90, logger=logger)
    @augment_response_object
    @ensure_messages_format
    def generate_response(self, messages: List[Dict]) -> Tuple[str, Any, str]:
        prompt = self.encode_messages(messages)
        gen_kwargs = dict(model=self.model_spec.model_id,
                          messages=prompt,
                          temperature=self.temperature,
                          max_completion_tokens=self.max_tokens)
        model_config = getattr(self.model_spec, "model_config", {})
        if 'reasoning_model' in model_config:
            if 'reasoning_effort' in model_config:
                gen_kwargs['reasoning_effort'] = model_config['reasoning_effort']
            if not self.temperature > 0:
                raise ValueError(
                    f"For reasoning models temperature must be >0, but is {self.temperature}. "
                    f"Please use the -t option to set a temperature and try again.")
            del gen_kwargs["max_completion_tokens"]

        if self.model_spec["model_name"] == "upgpt-codex":
            api_response = self.client.responses.create(
                model=self.model_spec["model_id"], input=prompt)
            response_text = api_response.output_text.strip()
            response = json.loads(api_response.json())
        else:
            api_response = self.client.chat.completions.create(**gen_kwargs)
            message = api_response.choices[0].message
            if message.role != "assistant":
                raise AttributeError("Response message role is " + message.role + " but should be 'assistant'")
            response_text = message.content.strip()
            response = json.loads(api_response.json())

        return prompt, response, response_text


# ---------------------------------------------------------------------------
# Claude / Anthropic model
# ---------------------------------------------------------------------------

class AzureClaudeModel(backends.Model):

    def __init__(self, client: AnthropicFoundry, model_spec: backends.ModelSpec):
        super().__init__(model_spec)
        self.client = client

    def encode_messages(self, messages) -> list:
        encoded_messages = []
        for message in messages:
            if "image" not in message.keys():
                encoded_messages.append({"role": message["role"], "content": message["content"]})
            else:
                this = {"role": message["role"],
                        "content": [{"type": "text",
                                     "text": message["content"].replace(" <image> ", " ")}]}

                if 'multimodality' not in self.model_spec.model_config:
                    logger.info(
                        f"The backend {self.model_spec.model_id} does not support multimodal inputs!")
                    raise Exception(
                        f"The backend {self.model_spec.model_id} does not support multimodal inputs!")

                if not self.model_spec['model_config']['multimodality']['multiple_images'] \
                        and len(message['image']) > 1:
                    logger.info(f"The backend {self.model_spec.model_id} does not support multiple images!")
                    raise Exception(f"The backend {self.model_spec.model_id} does not support multiple images!")

                for image in message['image']:
                    is_url, loaded, image_type = encode_image(image)
                    if is_url:
                        this["content"].append(dict(type="image_url", image_url={"url": loaded}))
                    else:
                        this["content"].append(dict(type="image_url",
                                                    image_url={"url": f"data:{image_type};base64,{loaded}"}))
                encoded_messages.append(this)
        return encoded_messages

    @retry(tries=3, delay=90, logger=logger)
    @augment_response_object
    @ensure_messages_format
    def generate_response(self, messages: List[Dict]) -> Tuple[str, Any, str]:
        prompt = self.encode_messages(messages)
        gen_kwargs = dict(model=self.model_spec.model_id,
                          messages=prompt,
                          temperature=self.temperature,
                          max_tokens=self.max_tokens)
        content_index = 0
        model_config = getattr(self.model_spec, "model_config", {})
        if 'thinking_mode' in model_config:
            # Thinking mode constraints:
            # - Not compatible with temperature != 1 or top_k modifications
            # - top_p must be between 0.95 and 1 when thinking is enabled
            # - Cannot pre-fill responses
            # - Changes to thinking budget invalidate cached message prefixes
            # https://platform.claude.com/docs/en/build-with-claude/extended-thinking
            content_index = 1
            gen_kwargs["temperature"] = 1.
            thinking_budget = model_config['thinking_budget']
            gen_kwargs["max_tokens"] = thinking_budget + self.max_tokens
            gen_kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

        api_response = self.client.messages.create(**gen_kwargs)
        response_text = api_response.content[content_index].text
        response = api_response.model_dump(mode="json")
        return prompt, response, response_text
