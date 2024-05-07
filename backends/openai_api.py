from typing import List, Dict, Tuple, Any
from retry import retry

import json
import openai
import backends
import base64
from backends.utils import ensure_messages_format

logger = backends.get_logger(__name__)

NAME = "openai"


class OpenAI(backends.Backend):

    def __init__(self):
        creds = backends.load_credentials(NAME)
        api_key = creds[NAME]["api_key"]
        organization = creds[NAME]["organisation"] if "organisation" in creds[NAME] else None
        self.client = openai.OpenAI(api_key=api_key, organization=organization)

    def list_models(self):
        models = self.client.models.list()
        names = [item.id for item in models.data]
        names = sorted(names)
        return names
        # [print(n) for n in names]   # 2024-01-10: what was this? a side effect-only method?

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        return OpenAIModel(self.client, model_spec)


class OpenAIModel(backends.Model):

    def __init__(self, client: openai.OpenAI, model_spec: backends.ModelSpec):
        super().__init__(model_spec)
        self.client = client
        
    def encode_image(self, image_path):
        if image_path.startswith('http'):
            return True, image_path
        with open(image_path, "rb") as image_file:
            return False, base64.b64encode(image_file.read()).decode('utf-8')
        
    def apply_vision_format(self, messages):
        vision_messages = []
        for message in messages:
            vision_message = {"role": message["role"], 
                    "content": [
                        {
                            "type": "text",
                            "text": message["content"]
                        }
                ]}
            if "image" in message:
                if isinstance(message['image'], str):
                    message['image'] = [message['image']]
                for img in message['image']:
                    url, loaded = self.encode_image(img)
                    vision_message["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": loaded if url else f"data:image/jpeg;base64,{loaded}"
                        }
                    })
            vision_messages.append(vision_message)
        return vision_messages

    @retry(tries=3, delay=0, logger=logger)
    @ensure_messages_format
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
        if self.model_spec.has_attr("vision"):
            if self.model_spec["vision"]:
                messages = self.apply_vision_format(messages)
        prompt = messages
        api_response = self.client.chat.completions.create(model=self.model_spec.model_id,
                                                           messages=prompt,
                                                           temperature=self.get_temperature(),
                                                           max_tokens=self.get_max_tokens())
        message = api_response.choices[0].message
        if message.role != "assistant":  # safety check
            raise AttributeError("Response message role is " + message.role + " but should be 'assistant'")
        response_text = message.content.strip()
        response = json.loads(api_response.json())

        return prompt, response, response_text
