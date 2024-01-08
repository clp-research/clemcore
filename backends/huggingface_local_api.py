""" Backend using HuggingFace transformers & ungated models. Uses HF tokenizers instruct/chat templates for proper input format per model. """
from typing import List, Dict, Tuple, Any
import torch
import backends

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import copy

logger = backends.get_logger(__name__)

MODEL_MISTRAL_7B_INSTRUCT_V0_1 = "Mistral-7B-Instruct-v0.1"
MODEL_RIIID_SHEEP_DUCK_LLAMA_2_70B_V1_1 = "sheep-duck-llama-2-70b-v1.1"
MODEL_RIIID_SHEEP_DUCK_LLAMA_2_13B = "sheep-duck-llama-2-13b"
MODEL_FALCON_7B_INSTRUCT = "falcon-7b-instruct"
MODEL_FALCON_40B_INSTRUCT = "falcon-40b-instruct"
MODEL_OPEN_ASSISTANT_12B = "oasst-sft-4-pythia-12b-epoch-3.5"
MODEL_KOALA_13B = "koala-13B-HF"
MODEL_WIZARD_VICUNA_13B = "Wizard-Vicuna-13B-Uncensored-HF"
MODEL_GOOGLE_FLAN_T5 = "flan-t5-xxl"
MODEL_WIZARDLM_70B_V1 = "WizardLM-70b-v1.0"
MODEL_WIZARDLM_13B_V1_2 = "WizardLM-13b-v1.2"
MODEL_LMSYS_VICUNA_7B = "vicuna-7b-v1.5"
MODEL_LMSYS_VICUNA_13B = "vicuna-13b-v1.5"
MODEL_LMSYS_VICUNA_33B = "vicuna-33b-v1.3"
MODEL_GPT4ALL_13B_SNOOZY = "gpt4all-13b-snoozy"
MODEL_CODELLAMA_34B_I = "CodeLlama-34b-Instruct-hf"
MODEL_ZEPHYR_7B_ALPHA = "zephyr-7b-alpha"
MODEL_ZEPHYR_7B_BETA = "zephyr-7b-beta"
MODEL_OPENCHAT_3_5 = "openchat_3.5"
MODEL_YI_34B_CHAT = "Yi-34B-Chat"
MODEL_ORCA_2_13B = "Orca-2-13b"
MODEL_DEEPSEEK_7B_CHAT = "deepseek-llm-7b-chat"
MODEL_DEEPSEEK_67B_CHAT = "deepseek-llm-67b-chat"
MODEL_TULU_2_DPO_7B = "tulu-2-dpo-7b"
MODEL_TULU_2_DPO_70B = "tulu-2-dpo-70b"
MODEL_MIXTRAL_8X7B_INSTRUCT_V0_1 = "Mixtral-8x7B-Instruct-v0.1"
MODEL_SUS_CHAT_34B = "SUS-Chat-34B"


SUPPORTED_MODELS = [MODEL_MISTRAL_7B_INSTRUCT_V0_1, MODEL_RIIID_SHEEP_DUCK_LLAMA_2_70B_V1_1,
                    MODEL_RIIID_SHEEP_DUCK_LLAMA_2_13B, MODEL_FALCON_7B_INSTRUCT, MODEL_OPEN_ASSISTANT_12B,
                    MODEL_KOALA_13B, MODEL_WIZARD_VICUNA_13B, MODEL_WIZARDLM_70B_V1, MODEL_WIZARDLM_13B_V1_2,
                    MODEL_LMSYS_VICUNA_13B, MODEL_LMSYS_VICUNA_33B, MODEL_LMSYS_VICUNA_7B, MODEL_GPT4ALL_13B_SNOOZY,
                    MODEL_CODELLAMA_34B_I, MODEL_ZEPHYR_7B_ALPHA, MODEL_ZEPHYR_7B_BETA, MODEL_OPENCHAT_3_5,
                    MODEL_YI_34B_CHAT, MODEL_DEEPSEEK_7B_CHAT, MODEL_DEEPSEEK_67B_CHAT, MODEL_TULU_2_DPO_7B,
                    MODEL_TULU_2_DPO_70B, MODEL_MIXTRAL_8X7B_INSTRUCT_V0_1, MODEL_SUS_CHAT_34B]


NAME = "huggingface"

# models that come with proper tokenizer chat template:
PREMADE_CHAT_TEMPLATE = [MODEL_MISTRAL_7B_INSTRUCT_V0_1, MODEL_CODELLAMA_34B_I, MODEL_ZEPHYR_7B_ALPHA,
                         MODEL_ZEPHYR_7B_BETA, MODEL_MIXTRAL_8X7B_INSTRUCT_V0_1]

# models to apply Orca-Hashes template to:
ORCA_HASH = [MODEL_RIIID_SHEEP_DUCK_LLAMA_2_70B_V1_1, MODEL_RIIID_SHEEP_DUCK_LLAMA_2_13B]
# jinja template for Orca-Hashes format:
orca_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '### User:\\n' + message['content'] + '\\n\\n' }}{% elif message['role'] == 'system' %}{{ '### System:\\n' + message['content'] + '\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ '### Assistant:\\n' + message['content'] + '\\n\\n' }}{% endif %}{% if loop.last %}{{ '### Assistant:\\n' }}{% endif %}{% endfor %}"
VICUNA = [MODEL_WIZARD_VICUNA_13B, MODEL_WIZARDLM_70B_V1, MODEL_WIZARDLM_13B_V1_2, MODEL_LMSYS_VICUNA_13B,
          MODEL_LMSYS_VICUNA_33B, MODEL_LMSYS_VICUNA_7B, MODEL_GPT4ALL_13B_SNOOZY]
# jinja template for Vicuna 1.1 format:
vicuna_1_1_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] + '\\n' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + '</s>\\n' }}{% endif %}{% if loop.last %}{{ 'ASSISTANT:' }}{% endif %}{% endfor %}"
KOALA = [MODEL_KOALA_13B]
# jinja template for Koala format:
koala_template = "{{ 'BEGINNING OF CONVERSATION: ' }}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] + ' ' }}{% elif message['role'] == 'assistant' %}{{ 'GPT: ' + message['content'] + ' ' }}{% endif %}{% if loop.last %}{{ 'GPT:' }}{% endif %}{% endfor %}"
OASST = [MODEL_OPEN_ASSISTANT_12B]
# jinja template for OpenAssist format:
oasst_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|prompter|>' + message['content'] + '<|endoftext|>' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>' + message['content'] + '<|endoftext|>' }}{% endif %}{% if loop.last %}{{ '<|assistant|>' }}{% endif %}{% endfor %}"
FALCON = [MODEL_FALCON_7B_INSTRUCT, MODEL_FALCON_40B_INSTRUCT]
# jinja template for assumed Falcon format:
falcon_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] + '\\n' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + '\\n' }}{% endif %}{% if loop.last %}{{ 'ASSISTANT:' }}{% endif %}{% endfor %}"
# Falcon template based on https://huggingface.co/tiiuae/falcon-7b-instruct/discussions/1#64708b0a3df93fddece002a4
OPENCHAT = [MODEL_OPENCHAT_3_5]
# jinja template for openchat format:
openchat_template = "{{ bos_token }}{% for message in messages %}{{ 'GPT4 Correct ' + message['role'].title() + ': ' + message['content'] + '<|end_of_turn|>'}}{% endfor %}GPT4 Correct Assistant:"
CHATML = [MODEL_YI_34B_CHAT, MODEL_ORCA_2_13B]
# jinja template for chatml format:
chatml_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = true %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
TULU = [MODEL_TULU_2_DPO_7B, MODEL_TULU_2_DPO_70B]
# jinja template for tulu format:
tulu_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
DEEPSEEK = [MODEL_DEEPSEEK_7B_CHAT, MODEL_DEEPSEEK_67B_CHAT]
# jinja template for deepseek format:
deepseek_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = true %}{% endif %}{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ message['content'] + '\n\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
SUSTECH = [MODEL_SUS_CHAT_34B]
sus_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '### Human: ' + message['content'] + '\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ '### Assistant: ' + message['content'] }}{% endif %}{% if loop.last %}{{ '### Assistant: ' }}{% endif %}{% endfor %}"


# templates currently have 'generation prompt' hardcoded
# doesn't matter for clembench, but once added, templates can be pushed to HF and this block can be reduced
# newer versions of transformers/tokenizers are supposed to properly handle the generation prompt argument
# but transformers==4.34.0 does not support this feature (at least not reliably)

# due to issues with differences between fast and slow HF tokenizer classes, some models require the 'slow' class/arg
SLOW_TOKENIZER = [MODEL_YI_34B_CHAT, MODEL_ORCA_2_13B, MODEL_SUS_CHAT_34B]


class HuggingfaceLocal(backends.Backend):
    def __init__(self):
        self.temperature: float = -1.
        self.model_loaded = False

    def load_model(self, model_name):
        logger.info(f'Start loading huggingface model: {model_name}')

        # model cache handling
        root_data_path = os.path.join(os.path.abspath(os.sep), "data")
        # check if root/data exists:
        if not os.path.isdir(root_data_path):
            logger.info(f"{root_data_path} does not exist, creating directory.")
            # create root/data:
            os.mkdir(root_data_path)
        CACHE_DIR = os.path.join(root_data_path, "huggingface_cache")

        if model_name in [MODEL_MISTRAL_7B_INSTRUCT_V0_1, MODEL_MIXTRAL_8X7B_INSTRUCT_V0_1]:  # mistralai models
            hf_user_prefix = "mistralai/"
        elif model_name in [MODEL_RIIID_SHEEP_DUCK_LLAMA_2_70B_V1_1,
                            MODEL_RIIID_SHEEP_DUCK_LLAMA_2_13B]:  # Riiid models
            hf_user_prefix = "Riiid/"
        elif model_name in [MODEL_FALCON_7B_INSTRUCT, MODEL_FALCON_40B_INSTRUCT]:  # tiiuae models
            hf_user_prefix = "tiiuae/"
        elif model_name in [MODEL_OPEN_ASSISTANT_12B]:  # OpenAssistant models
            hf_user_prefix = "OpenAssistant/"
        elif model_name in [MODEL_KOALA_13B, MODEL_WIZARD_VICUNA_13B]:  # TheBloke models
            hf_user_prefix = "TheBloke/"
        elif model_name in [MODEL_GOOGLE_FLAN_T5]:  # Google models
            hf_user_prefix = "google/"
        elif model_name in [MODEL_WIZARDLM_70B_V1, MODEL_WIZARDLM_13B_V1_2]:  # WizardLM models
            hf_user_prefix = "WizardLM/"
        elif model_name in [MODEL_LMSYS_VICUNA_7B, MODEL_LMSYS_VICUNA_13B, MODEL_LMSYS_VICUNA_33B]:  # lmsys models
            hf_user_prefix = "lmsys/"
        elif model_name in [MODEL_GPT4ALL_13B_SNOOZY]:  # nomic-ai models
            hf_user_prefix = "nomic-ai/"
        elif model_name in [MODEL_CODELLAMA_34B_I]:  # codellama models
            hf_user_prefix = "codellama/"
        elif model_name in [MODEL_ZEPHYR_7B_ALPHA, MODEL_ZEPHYR_7B_BETA]:  # HuggingFaceH4 models
            hf_user_prefix = "HuggingFaceH4/"
        elif model_name in [MODEL_OPENCHAT_3_5]:  # openchat models
            hf_user_prefix = "openchat/"
        elif model_name in [MODEL_YI_34B_CHAT]:  # 01-ai models
            hf_user_prefix = "01-ai/"
        elif model_name in [MODEL_ORCA_2_13B]:  # microsoft models
            hf_user_prefix = "microsoft/"
        elif model_name in [MODEL_DEEPSEEK_7B_CHAT, MODEL_DEEPSEEK_67B_CHAT]:  # deepseek-ai models
            hf_user_prefix = "deepseek-ai/"
        elif model_name in [MODEL_TULU_2_DPO_7B, MODEL_TULU_2_DPO_70B]:  # allenai models
            hf_user_prefix = "allenai/"
        elif model_name in [MODEL_SUS_CHAT_34B]:  # SUSTech models
            hf_user_prefix = "SUSTech/"

        hf_model_str = f"{hf_user_prefix}{model_name}"

        # use 'slow' tokenizer for models that require it:
        if model_name in SLOW_TOKENIZER:
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_str, device_map="auto", torch_dtype="auto",
                                                           cache_dir=CACHE_DIR, verbose=False, use_fast=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_str, device_map="auto", torch_dtype="auto",
                                                           cache_dir=CACHE_DIR, verbose=False)

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
            elif model_name in OPENCHAT:
                self.tokenizer.chat_template = openchat_template
            elif model_name in CHATML:
                self.tokenizer.chat_template = chatml_template
            elif model_name in TULU:
                self.tokenizer.chat_template = tulu_template
            elif model_name in DEEPSEEK:
                self.tokenizer.chat_template = deepseek_template
            elif model_name in SUSTECH:
                self.tokenizer.chat_template = sus_template


        # load all models using their default configuration:
        self.model = AutoModelForCausalLM.from_pretrained(hf_model_str, device_map="auto", torch_dtype="auto",
                                                          cache_dir=CACHE_DIR)

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
        :param max_new_tokens: How many tokens to generate ('at most', but no stop sequence is defined).
        :param return_full_text: If True, whole input context is returned.
        :return: the continuation
        """
        assert 0.0 <= self.temperature <= 1.0, "Temperature must be in [0.,1.]"

        # load the model to the memory
        if not self.model_loaded:
            self.load_model(model)
            logger.info(f"Finished loading huggingface model: {model}")
            logger.info(f"Model device map: {self.model.hf_device_map}")

        # log current given messages list:
        # logger.info(f"Raw messages passed: {messages}")

        # deepcopy messages to prevent reference issues:
        current_messages = copy.deepcopy(messages)

        # cull empty system message:
        if current_messages[0]['role'] == "system":
            if not current_messages[0]['content']:
                del current_messages[0]

        # flatten consecutive user messages:
        for msg_idx, message in enumerate(current_messages):
            if msg_idx > 0 and message['role'] == "user" and current_messages[msg_idx - 1]['role'] == "user":
                current_messages[msg_idx - 1]['content'] += f" {message['content']}"
                del current_messages[msg_idx]
            elif msg_idx > 0 and message['role'] == "assistant" and current_messages[msg_idx - 1]['role'] == "assistant":
                current_messages[msg_idx - 1]['content'] += f" {message['content']}"
                del current_messages[msg_idx]

        # log current flattened messages list:
        # logger.info(f"Flattened messages: {current_messages}")

        # apply chat template & tokenize:
        prompt_tokens = self.tokenizer.apply_chat_template(current_messages, return_tensors="pt")
        prompt_tokens = prompt_tokens.to(self.device)

        prompt_text = self.tokenizer.batch_decode(prompt_tokens)[0]
        prompt = {"inputs": prompt_text, "max_new_tokens": max_new_tokens,
                  "temperature": self.temperature, "return_full_text": return_full_text}

        # greedy decoding:
        do_sample: bool = False
        if self.temperature > 0.0:
            do_sample = True

        # test to check if temperature is properly set on this Backend object:
        # logger.info(f"Currently used temperature for this instance of HuggingfaceLocal: {self.temperature}")

        if do_sample:
            model_output_ids = self.model.generate(
                prompt_tokens,
                temperature=self.temperature,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample
            )
        else:
            model_output_ids = self.model.generate(
                prompt_tokens,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample
            )

        model_output = self.tokenizer.batch_decode(model_output_ids)[0]

        response = {'response': model_output}

        # cull input context; equivalent to transformers.pipeline method:
        if not return_full_text:
            response_text = model_output.replace(prompt_text, '').strip()

            # handle Yi decoded output mismatch:
            if model == MODEL_YI_34B_CHAT:
                response_text = model_output.rsplit("assistant\n", maxsplit=1)[1]

            # remove llama2 EOS token at the end of output:
            if response_text[-4:len(response_text)] == "</s>":
                response_text = response_text[:-4]
            # remove openchat EOS token at the end of output:
            if response_text[-15:len(response_text)] == "<|end_of_turn|>":
                response_text = response_text[:-15]
            # remove ChatML EOS token at the end of output:
            if response_text[-10:len(response_text)] == "<|im_end|>":
                response_text = response_text[:-10]
            # remove DeepSeek EOS token at the end of output:
            if response_text[-19:len(response_text)] == "<｜end▁of▁sentence｜>":
                response_text = response_text[:-19]
            # remove SUS EOS token at the end of output:
            if response_text[-13:len(response_text)] == "<|endoftext|>":
                response_text = response_text[:-13]

        else:
            response_text = model_output.strip()

        return prompt, response, response_text

    def supports(self, model_name: str):
        return model_name in SUPPORTED_MODELS
