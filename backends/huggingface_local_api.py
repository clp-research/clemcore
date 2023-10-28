""" Backend using HuggingFace transformers & ungated models. Uses HF tokenizers instruct/chat templates for proper input format per model. """
from typing import List, Dict, Tuple, Any
import torch
import backends
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = backends.get_logger(__name__)

MODEL_MISTRAL_MISTRAL_7B_INSTRUCT_V0_1 = "Mistral-7B-Instruct-v0.1"
MODEL_RIIID_SHEEP_DUCK_LLAMA_2_70B_V1_1 = "sheep-duck-llama-2-70b-v1.1"
MODEL_RIIID_SHEEP_DUCK_LLAMA_2_13B = "sheep-duck-llama-2-13b"
MODEL_FALCON_7B_INSTRUCT = "falcon-7b-instruct"
MODEL_FALCON_40B_INSTRUCT = "falcon-40b-instruct"
MODEL_OPEN_ASSISTANT_12B = "oasst-sft-4-pythia-12b-epoch-3.5"
MODEL_KOALA_13B = "koala-13B-HF"
MODEL_VICUNA_13B = "Wizard-Vicuna-13B-Uncensored-HF"
MODEL_GOOGLE_FLAN_T5 = "flan-t5-xxl"
SUPPORTED_MODELS = [MODEL_MISTRAL_MISTRAL_7B_INSTRUCT_V0_1, MODEL_RIIID_SHEEP_DUCK_LLAMA_2_70B_V1_1, MODEL_RIIID_SHEEP_DUCK_LLAMA_2_13B, MODEL_FALCON_7B_INSTRUCT, MODEL_OPEN_ASSISTANT_12B, MODEL_KOALA_13B, MODEL_VICUNA_13B]

NAME = "huggingface"

# models that come with proper tokenizer chat template:
PREMADE_CHAT_TEMPLATE = ["Mistral-7B-Instruct-v0.1"]
# models to apply Orca-Hashes template to:
ORCA_HASH = ["sheep-duck-llama-2-70b-v1.1", "sheep-duck-llama-2-13b"]
# jinja template for Orca-Hashes format:
orca_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '### User:\\n' + message['content'] + '\\n\\n' }}{% elif message['role'] == 'system' %}{{ '### System:\\n' + message['content'] + '\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ '### Assistant:\\n' + message['content'] + '\\n\\n' }}{% endif %}{% if loop.last %}{{ '### Assistant:\\n' }}{% endif %}{% endfor %}"
VICUNA = ["Wizard-Vicuna-13B-Uncensored-HF"]
# jinja template for Vicuna 1.1 format:
vicuna_1_1_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] + '\\n' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + '</s>\\n' }}{% endif %}{% if loop.last %}{{ 'ASSISTANT: ' }}{% endif %}{% endfor %}"
KOALA = ["koala-13B-HF"]
# jinja template for Koala format:
koala_template = "{{ 'BEGINNING OF CONVERSATION: ' }}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] + ' ' }}{% elif message['role'] == 'assistant' %}{{ 'GPT: ' + message['content'] + ' ' }}{% endif %}{% if loop.last %}{{ 'GPT:' }}{% endif %}{% endfor %}"
OASST = ["oasst-sft-4-pythia-12b-epoch-3.5"]
# jinja template for OpenAssist format:
oasst_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|prompter|>' + message['content'] + '<|endoftext|>' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>' + message['content'] + '<|endoftext|>' }}{% endif %}{% if loop.last %}{{ '<|assistant|>' }}{% endif %}{% endfor %}"
FALCON = ["falcon-7b-instruct", "falcon-40b-instruct"]
# jinja template for (minimal) Falcon format:
falcon_template = "{% for message in messages %}{{ message.content }}{{ eos_token }}{% endfor %}"
# templates currently have 'generation prompt' hardcoded
# doesn't matter for clembench, but once added, templates can be pushed to HF and this block can be reduced


class HuggingfaceLocal2(backends.Backend):
    def __init__(self):
        self.temperature: float = -1.
        self.model_loaded = False

    def load_model(self, model_name):
        logger.info(f'Start loading huggingface model: {model_name}')

        CACHE_DIR = 'huggingface_cache'

        if model_name in ["Mistral-7B-Instruct-v0.1"]:  # mistralai models
            hf_user_prefix = "mistralai/"
        elif model_name in ["sheep-duck-llama-2-70b-v1.1", "sheep-duck-llama-2-13b"]:  # Riiid models
            hf_user_prefix = "Riiid/"
        elif model_name in ["falcon-7b-instruct", "falcon-40b-instruct"]:  # tiiuae models
            hf_user_prefix = "tiiuae/"
        elif model_name in ["oasst-sft-4-pythia-12b-epoch-3.5"]:  # OpenAssistant models
            hf_user_prefix = "OpenAssistant/"
        elif model_name in ["koala-13B-HF", "Wizard-Vicuna-13B-Uncensored-HF"]:  # TheBloke models
            hf_user_prefix = "TheBloke/"
        elif model_name in ["flan-t5-xxl"]:  # Google models
            hf_user_prefix = "google/"

        hf_model_str = f"{hf_user_prefix}{model_name}"

        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_str, device_map="auto", cache_dir=CACHE_DIR)
        # apply proper chat template:
        if model_name not in PREMADE_CHAT_TEMPLATE:
            if model_name in ORCA_HASH:
                self.tokenizer.chat_template = orca_template
            elif model_name in FALCON:
                self.tokenizer.chat_template = falcon_template
            elif model_name in OASST:
                self.tokenizer.chat_template = oasst_template
            elif model_name in KOALA:
                self.tokenizer.chat_template = koala_template
            elif model_name in VICUNA:
                self.tokenizer.chat_template = vicuna_1_1_template

        # TODO: remove quantization from falcon
        if model_name in FALCON:
            self.model = AutoModelForCausalLM.from_pretrained(
                hf_model_str, device_map="auto", load_in_8bit=True, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                trust_remote_code=True, cache_dir=CACHE_DIR
            )
        elif model_name in OASST:
            self.model = AutoModelForCausalLM.from_pretrained(
                hf_model_str, device_map="auto", torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(hf_model_str, device_map="auto", cache_dir=CACHE_DIR)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model_loaded = True

    def generate_response(self, messages: List[Dict], model: str,
                          max_new_tokens: int = 100, return_full_text: bool = False) -> Tuple[Any, Any, str]:
        """
        :param messages: for example
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        :param model: model name
        :return: the continuation
        """
        assert 0.0 <= self.temperature <= 1.0, "Temperature must be in [0.,1.]"

        # load the model to the memory
        if not self.model_loaded:
            self.load_model(model)
            logger.info(f"Finished loading huggingface model: {model}")

        # greedy decoding:
        do_sample: bool = False
        if self.temperature > 0.0:
            do_sample = True

        # apply chat template & tokenize
        prompt_tokens = self.tokenizer.apply_chat_template(messages, return_tensors="pt")

        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        prompt = {"inputs": prompt_text, "max_new_tokens": max_new_tokens,
                  "temperature": self.temperature, "return_full_text": return_full_text}

        model_output_ids = self.model.generate(prompt_tokens,
                            temperature=self.temperature,
                            max_new_tokens=max_new_tokens,
                            return_full_text=return_full_text,
                            do_sample=do_sample)

        model_output = self.tokenizer.batch_decode(model_output_ids, skip_special_tokens=True)

        # TODO: Check outputs to see if cleanup below is needed
        # most of this should not be needed with return_full_text=False and skip_special_tokens=True
        # response_text = generated_text.replace(prompt_text, '')\
        response_text = model_output.replace(prompt_text, '')\
            .replace('<pad>', '')\
            .replace('<s>', '')\
            .replace('</s>','')\
            .replace('<|endoftext|>', '')\
            .replace('<|assistant|>','').strip()

        response = {'response': model_output}
        return prompt, response, response_text

    def supports(self, model_name: str):
        return model_name in SUPPORTED_MODELS
