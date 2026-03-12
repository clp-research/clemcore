"""Backend using HuggingFace transformers models.
Uses HF tokenizers instruct/chat templates for proper input format per model.
"""
import logging
from typing import List, Dict, Tuple, Any
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoProcessor, BitsAndBytesConfig, PreTrainedTokenizerBase, PreTrainedModel
from transformers.generation.utils import GenerateOutput
from transformers.image_utils import load_image
from peft import PeftModel
from jinja2 import TemplateError
from jinja2 import Environment as Jinja2Env
from jinja2 import meta as jinja2_meta

import clemcore.backends as backends
from clemcore.backends.key_registry import KeyRegistry
from clemcore.backends.utils import ensure_alternating_roles, ensure_messages_format, augment_response_object, \
    ContextExceededError

logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("clemcore.cli")

FALLBACK_CONTEXT_SIZE = 256
_MAX_TOKENIZER_CONTEXT_GUARD = 1_000_000  # guard against very large sentinel values


def _parse_context_size(value: Any) -> int | None:
    """Parse context size from model registry values like 8192 or '128k'."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return None
        match = re.fullmatch(r"(\d+)([km])?", text)
        if not match:
            return None
        number = int(match.group(1))
        suffix = match.group(2)
        if suffix == "k":
            return number * 1024
        if suffix == "m":
            return number * 1024 * 1024
        return number
    return None


def _context_size_from_config(auto_config, model_spec: "backends.ModelSpec", tokenizer=None) -> int:
    """Resolve context size from AutoConfig, falling back to registry entry, tokenizer metadata, or a hard default.
    Args:
        auto_config: The AutoConfig instance for the model.
        model_spec: The ModelSpec for the model; checked for a top-level 'context_size' entry.
        tokenizer: Optional tokenizer; its model_max_length is used as a secondary fallback.
    Returns:
        Context size as int.
    """
    if hasattr(auto_config, 'max_position_embeddings'):
        return auto_config.max_position_embeddings
    if hasattr(auto_config, 'n_positions'):
        return auto_config.n_positions
    context_size = _parse_context_size(getattr(model_spec, "context_size", None))
    if context_size is None and tokenizer is not None:
        tokenizer_max = getattr(tokenizer, "model_max_length", None)
        if isinstance(tokenizer_max, int) and tokenizer_max < _MAX_TOKENIZER_CONTEXT_GUARD:
            context_size = tokenizer_max
    return context_size or FALLBACK_CONTEXT_SIZE


def check_chat_template_kwargs(chat_template: str, chat_template_kwargs: dict,
                               ignore_variables: tuple = ("messages", "content",
                                                          "add_generation_prompt", "raise_exception")):
    """Checks a model's chat template for passable variables and compares which those in a chat template kwargs dict.
    Chat template kwargs dicts are directly imported from model registry JSON files, and the template rendering simply
    ignores kwargs that do not match variables in the template, so without checking, incompatible/typo'd chat template
    kwargs will fail silently.
    Outcomes are simply logged to not interfere with benchmarking - in case of unexpected outputs, check logs.
    Args:
        chat_template: The 'raw' chat template string from the used tokenizer instance.
        chat_template_kwargs: The chat template kwargs dict imported from the model registry JSON file.
        ignore_variables: Default variables that are either always passed or already handled by transformers/tokenizers
            chat template rendering. Might be useful for in-depth debugging, hence not completely hardcoded.
    """
    # create simple jinja2 environment to allow for template parsing:
    chat_template_test_jinja2env = Jinja2Env()
    # parse the raw chat template to allow for template variable checking:
    parsed_chat_template = chat_template_test_jinja2env.parse(chat_template)
    # extract all variables in chat template:
    chat_template_vars = jinja2_meta.find_undeclared_variables(parsed_chat_template)
    # filter out ignored variables:
    filtered_chat_template_vars = [var for var in chat_template_vars if var not in ignore_variables]
    # get vars that are in chat template, but not covered in the kwargs dict:
    chat_template_vars_not_in_kwargs = [var for var in filtered_chat_template_vars
                                        if var not in chat_template_kwargs.keys()]
    # get chat template kwargs that are not in chat template:
    kwargs_not_in_chat_template_vars = [var for var in chat_template_kwargs.keys()
                                        if var not in filtered_chat_template_vars]
    # log the outcome:
    logger.info(f"Non-default chat template variables not in model entry chat template kwargs: {chat_template_vars_not_in_kwargs}")
    logger.info(f"Model entry chat template kwargs not found in chat template variables: {kwargs_not_in_chat_template_vars}")


def load_config_and_tokenizer(model_spec: backends.ModelSpec) -> Tuple[PreTrainedTokenizerBase, AutoConfig, int]:
    """Load a HuggingFace model's standard config and tokenizer, and get context token limit from config.
    If the model config does not contain the context limit, it is set to 256 as fallback. Does not load the model
    weights, allowing for prototyping on non-GPU systems.
    Args:
        model_spec: The ModelSpec for the model.
    Returns:
        Tokenizer, model config and context token limit (int).
    """
    logger.info(f'Loading huggingface model config and tokenizer: {model_spec.model_name}')

    use_api_key = False
    api_key = None
    if 'requires_api_key' in model_spec.model_config:
        if model_spec['model_config']['requires_api_key']:
            # load HF API key:
            key = KeyRegistry.from_json().get_key_for("huggingface")
            api_key = key["api_key"]
            use_api_key = True
        else:
            requires_api_key_info = (f"{model_spec['model_name']} registry setting has requires_api_key, "
                                     f"but it is not 'true'. Please check the model entry.")
            stdout_logger.info(requires_api_key_info)
            logger.info(requires_api_key_info)

    hf_model_str = model_spec['huggingface_id']

    if use_api_key:
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            hf_model_str,
            token=api_key,
        )
    else:
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            hf_model_str,
        )

    # apply proper chat template:
    if not model_spec['model_config']['premade_chat_template']:
        if 'custom_chat_template' in model_spec.model_config:
            tokenizer.chat_template = model_spec['model_config']['custom_chat_template']
        else:
            logger.info(
                f"No custom chat template for {model_spec.model_name} found in model settings from model registry "
                f"while model has no pre-made template! Generic template will be used, likely leading to "
                f"bad results.")

    if use_api_key:
        auto_config = AutoConfig.from_pretrained(hf_model_str, token=api_key)
    else:
        auto_config = AutoConfig.from_pretrained(hf_model_str)

    # get context token limit for model:
    context_size = _context_size_from_config(auto_config, model_spec, tokenizer)

    # Decoder-only models (e.g., GPT, LLaMA) often don't define a pad token explicitly,
    # since they use causal attention over the entire left-context during generation.
    # To avoid warnings from Transformers when padding is used, we set the pad token
    # to the EOS token if it's not already defined.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Many models do not reliably set `padding_side` in their tokenizer configs,
    # especially decoder-only models where left-padding is needed for correct generation.
    # We first check for an explicit setting in `model_config`, and fall back to
    # automatic detection based on the model architecture.
    padding_side = model_spec.model_config.get("padding_side", None)
    if padding_side is None:
        stdout_logger.warning("No 'padding_side' configured in 'model_config' for %s", model_spec.model_name)
        is_encoder_decoder = getattr(auto_config, 'is_encoder_decoder', False)
        decoder_enabled = getattr(auto_config, 'is_decoder', None)

        model_type = getattr(auto_config, 'model_type', '')
        encoder_only_types = {"bert", "roberta", "albert", "electra", "distilbert",
                              "camembert", "xlm-roberta", "deberta", "deberta-v2",
                              "ernie", "funnel", "layoutlm", "xlm"}
        is_encoder = model_type in encoder_only_types or (is_encoder_decoder and not decoder_enabled)
        is_decoder = not is_encoder or (is_encoder_decoder and decoder_enabled)

        if is_encoder:
            tokenizer.padding_side = "right"
            stdout_logger.warning(
                "Model %s is encoder-only. Encoder-only models "
                "do not support text generation and may not work with this benchmark. ", model_spec.model_name)
            stdout_logger.warning("Deriving padding_side=%s from model architecture (encoder)",
                                  tokenizer.padding_side)
        elif is_decoder:
            tokenizer.padding_side = "left"
            stdout_logger.warning("Deriving padding_side=%s from model architecture (decoder, encoder-decoder=%s)",
                                  tokenizer.padding_side, is_encoder_decoder)
    else:
        padding_side = padding_side.lower()
        if padding_side not in ("left", "right"):
            raise ValueError(f"Invalid 'padding_side={padding_side}' configured in 'model_config' "
                             f"for {model_spec.model_name}. Must be 'left' or 'right'.")
        tokenizer.padding_side = padding_side

    return tokenizer, auto_config, context_size


def load_model(model_spec: backends.ModelSpec) -> PreTrainedModel | PeftModel:
    """Load Huggingface model weights, into VRAM if available.
    Weights are distributed over all available GPUs for maximum speed - make sure to limit the available GPUs using
    environment variables if only a subset is to be used.
    Args:
        model_spec: The ModelSpec for the model.
    Returns:
        The transformers model class instance of the loaded model.
    """
    logger.info(f'Start loading huggingface model weights: {model_spec.model_name}')

    model_args = dict(device_map="auto", torch_dtype="auto")
    bnb_kwargs = {}
    if "load_in_8bit" in model_spec.model_config:
        bnb_kwargs["load_in_8bit"] = model_spec.model_config["load_in_8bit"]
    if "load_in_4bit" in model_spec.model_config:
        bnb_kwargs["load_in_4bit"] = model_spec.model_config["load_in_4bit"]
    if bnb_kwargs:
        model_args["quantization_config"] = BitsAndBytesConfig(**bnb_kwargs)
    if 'requires_api_key' in model_spec.model_config and model_spec['model_config']['requires_api_key']:
        # load HF API key:
        key = KeyRegistry.from_json().get_key_for("huggingface")
        model_args["token"] = key["api_key"]
    if model_spec.model_config.get('trust_remote_code'):
        model_args['trust_remote_code'] = True
    if 'attn_implementation' in model_spec.model_config:
        model_args['attn_implementation'] = model_spec.model_config['attn_implementation']

    hf_model_str = model_spec['huggingface_id']
    model = AutoModelForCausalLM.from_pretrained(hf_model_str, **model_args)

    if "peft_model" in model_spec.model_config:
        adapter_model = model_spec.model_config["peft_model"]  # can be a path or name
        stdout_logger.info(f"Load PeftModel adapters from {adapter_model}")
        model = PeftModel.from_pretrained(model, adapter_model)

    logger.info(f"Finished loading huggingface model: {model_spec.model_name}")
    # Some model classes (e.g., certain Qwen variants) don't expose hf_device_map.
    # Guard to avoid AttributeError during logging.
    if hasattr(model, "hf_device_map"):
        logger.info(f"Model device map: {model.hf_device_map}")
    else:
        logger.info("Model device map: <unavailable>")

    return model


class HuggingfaceLocal(backends.Backend):
    """Model/backend handler class for locally-run Huggingface models."""

    def __init__(self):
        super().__init__()

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        """Return HuggingfaceLocalMultimodalModel for multimodal models, HuggingfaceLocalModel otherwise.
        Args:
            model_spec: The ModelSpec for the model.
        Returns:
            The Model class instance of the model.
        """
        torch.set_num_threads(1)
        if model_spec.model_config.get('multimodal', False):
            return HuggingfaceLocalMultimodalModel(model_spec)
        return HuggingfaceLocalModel(model_spec)


class HuggingfaceLocalModel(backends.BatchGenerativeModel):
    """Class for loaded HuggingFace transformers text-only models ready for generation."""

    def __init__(self, model_spec: backends.ModelSpec):
        """
        Args:
            model_spec: A ModelSpec instance specifying the model.
        """
        super().__init__(model_spec)
        # fail-fast
        self.tokenizer, self.config, self.context_size = load_config_and_tokenizer(model_spec)
        self.model: PreTrainedModel = load_model(model_spec)

        # validate and store chat_template_kwargs via setter (runs check_chat_template_kwargs)
        self.chat_template_kwargs = model_spec.model_config.get("chat_template_kwargs", {})

        # check if model's generation_config has pad_token_id set:
        if not self.model.generation_config.pad_token_id:
            # set pad_token_id to tokenizer's eos_token_id to prevent excessive warnings:
            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id

        self.device = next(self.model.parameters()).device

    @property
    def chat_template_kwargs(self) -> dict:
        return dict(self._chat_template_kwargs)

    @chat_template_kwargs.setter
    def chat_template_kwargs(self, value: dict):
        if value and not isinstance(value, dict):
            raise ValueError("chat_template_kwargs must be a dict")
        self._chat_template_kwargs = dict(value) if value else {}
        if self._chat_template_kwargs:
            check_chat_template_kwargs(self.tokenizer.chat_template, self._chat_template_kwargs)

    @augment_response_object
    @ensure_messages_format
    def generate_response(self, messages: List[Dict]) -> Tuple[Any, Any, str]:
        """
        Public method for single response generation.

        Wraps the batch response method internally to reuse batch logic.

        Args:
            messages (List[Dict]): List of message dicts.

        Returns:
            Tuple[Any, Any, str]: Single response tuple (prompt, response_object, response_text).
        """
        batch_messages = [messages]  # Wrap single message list into batch
        # Call batch method without decorators to avoid double invocation of decorators
        results = self._generate_batch_response(batch_messages)

        return results[0]  # Unpack single result to maintain original API

    @augment_response_object
    @ensure_messages_format
    def generate_batch_response(self, batch_messages: List[List[Dict]]) -> List[Tuple[Any, Any, str]]:
        """
        Public method for batch response generation.

        Args:
            batch_messages (List[List[Dict]]): Batch of message lists.

        Returns:
            List[Tuple[Any, Any, str]]: List of response tuples.
        """
        return self._generate_batch_response(batch_messages)

    def _generate_batch_response(self, batch_messages: List[List[Dict]]) -> List[Tuple[Any, Any, str]]:
        """
        Core batch response implementation without decorators.

        Args:
            batch_messages (List[List[Dict]]): Batch of message lists,
                assumed to be properly formatted.

        Returns:
            List[Tuple[Any, Any, str]]: List of response tuples (prompt, response_object, response_text).

        Note:
            Intended for internal use only. Use public decorated methods
            for normal calls to ensure formatting and metadata.
        """
        # We want to avoid the following warning given by Huggingface:
        # > The attention mask is not set and cannot be inferred from input because pad token is same as eos token.
        # This is mainly due to the fact that we set the pad_token to be the eos_token on generate()
        # when such a token is not specified, e.g., for LLama3 models. This causes the problem that
        # Huggingface cannot know anymore where the inputs end, and potentially the model attends to
        # parts of the inputs that are actually padded. However, this is usually not a problem for single
        # item batches, because here, the padding is not necessary anyway. In the following we first apply
        # the chat template and then use the tokenizer to receive the proper masks, also feasible for batches.

        # Bypassing CoT requires appending a message with empty CoT to the history, which is then completed by the model
        # As this is incompatible with the add_generation_prompt argument, it's handled separately here
        if 'cot_bypass' in self.model_spec.model_config and self.model_spec.model_config['cot_bypass']:
            # Add last message containing CoT bypass string content to each message history in batch
            for message_history in batch_messages:
                message_history.append({"role": "assistant", "content": self.model_spec.model_config['cot_bypass']})
            # Render each chat in the batch (list of messages) to a string prompt to continue after CoT bypass
            rendered_chats = self.tokenizer.apply_chat_template(
                batch_messages,
                continue_final_message=True,  # continue after CoT bypass
                tokenize=False,  # get back the rendered string
                **self._chat_template_kwargs
            )
        elif 'cot_effort' in self.model_spec.model_config and self.model_spec.model_config['cot_effort']:
            # Render each chat in the batch (list of messages) to a string prompt with generation prompt
            # including setting CoT effort to value defined in model registry entry
            # NOTE: Currently this is custom code to handle gpt-oss models! Other models that have CoT effort setting
            # training might not pass the value to the model and thus template in the same way. Using this for those
            # models will likely lead to errors!
            chat_template_kwargs = self.chat_template_kwargs
            if "reasoning_effort" not in chat_template_kwargs:
                chat_template_kwargs["reasoning_effort"] = self.model_spec.model_config['cot_effort']
            rendered_chats = self.tokenizer.apply_chat_template(
                batch_messages,
                add_generation_prompt=True,  # append assistant prompt
                tokenize=False,  # get back the rendered string
                **chat_template_kwargs
            )
        else:
            # Render each chat in the batch (list of messages) to a string prompt with generation prompt
            rendered_chats = self.tokenizer.apply_chat_template(
                batch_messages,
                add_generation_prompt=True,  # append assistant prompt
                tokenize=False,  # get back the rendered string
                **self._chat_template_kwargs
            )

        # The rendered chat (with system message already removed before) will, for example, look like:
        # <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWho won the world series in 2020?<|eot_id|>
        # Followed by the added generation prompt: <|start_header_id|>assistant<|end_header_id|>\n\n
        # Tokenize all chats at once with padding for batch
        encoding_dict = self.tokenizer(
            rendered_chats,
            add_special_tokens=False,  # <|begin_of_text|>/BOT token already added above
            return_tensors="pt",
            padding=True,  # pad to the longest sequence (necessary for batching)
            return_attention_mask=True  # with 1's up to sample length, followed by 0's
        )
        prompt_token_ids = encoding_dict["input_ids"].to(self.device)
        attention_mask = encoding_dict["attention_mask"].to(self.device)

        # Check context limit for each input in the batch
        assert_context_limits(self, prompt_token_ids)

        # Prepare generation arguments: by default assume greedy decoding (set values to None to avoid warnings)
        gen_args = {
            "do_sample": False,
            "temperature": None,  # avoid warning
            "top_p": None,  # avoid warning
            "max_new_tokens": self.max_tokens,
            "attention_mask": attention_mask,
            "return_dict_in_generate": True,
        }
        if self.temperature > 0.0:
            gen_args["do_sample"] = True
            gen_args["top_p"] = getattr(self.model.generation_config, "top_p", None)  # look in config for default value
            gen_args["temperature"] = self.temperature

        # Let CoT-output models generate to their context limit to assure CoT+final answer completion
        if 'cot_output' in self.model_spec.model_config and self.model_spec.model_config['cot_output']:
            gen_args["max_new_tokens"] = self.context_size

        # Put the model into evaluation mode e.g., disable dropout and configure batch norm etc.
        if self.model.training:
            stdout_logger.info("Model is in training mode; switching to eval mode for generation.")
            self.model.eval()

        # Generate outputs for the whole batch (Note: model.generate() is decorated with torch.no_grad() !)
        generation_output: GenerateOutput = self.model.generate(prompt_token_ids, **gen_args)

        # Decode all outputs and prompts
        model_outputs = self.tokenizer.decode(generation_output.sequences)
        prompt_texts = self.tokenizer.decode(prompt_token_ids)

        prompts, response_texts, responses = split_and_clean_batch_outputs(self,
                                                                           model_outputs,
                                                                           prompt_texts)
        return list(zip(prompts, responses, response_texts))


class HuggingfaceLocalMultimodalModel(backends.Model):
    """Class for loaded HuggingFace vision-language models ready for generation.
    Batch generation is not supported for multimodal models.
    CoT-specific features (cot_bypass, cot_output, etc.) are not applicable to VLMs.
    """

    def __init__(self, model_spec: backends.ModelSpec):
        """
        Args:
            model_spec: A ModelSpec instance specifying the model.
        """
        super().__init__(model_spec)
        self.processor = HuggingfaceLocalMultimodalModel.load_processor(model_spec)
        self.model: PreTrainedModel = load_model(model_spec)
        self.device = next(self.model.parameters()).device

        # context size from AutoConfig
        trust_rc = model_spec.model_config.get('trust_remote_code', False)
        auto_config = AutoConfig.from_pretrained(
            model_spec.huggingface_id,
            trust_remote_code=trust_rc
        )
        self.context_size = _context_size_from_config(auto_config, model_spec)

    @augment_response_object
    @ensure_messages_format
    def generate_response(self, messages: List[Dict]) -> Tuple[Any, Any, str]:
        """Generate a response for multimodal (vision-language) input."""
        hf_messages, images = HuggingfaceLocalMultimodalModel._unpack_images_from_messages(messages)
        text = self.processor.apply_chat_template(
            hf_messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(
            text=text,
            images=images if images else None,
            return_tensors="pt"
        ).to(self.device)

        # Context limit check on the tokenized input
        assert_context_limits(self, inputs['input_ids'])

        gen_args = {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "max_new_tokens": self.max_tokens,
            "return_dict_in_generate": True,
        }
        if self.temperature > 0.0:
            gen_args["do_sample"] = True
            gen_args["temperature"] = self.temperature
            gen_args["top_p"] = getattr(self.model.generation_config, "top_p", None)

        if self.model.training:
            stdout_logger.info("Model is in training mode; switching to eval mode for generation.")
            self.model.eval()

        input_len = inputs['input_ids'].shape[-1]
        generation_output: GenerateOutput = self.model.generate(**inputs, **gen_args)

        # Decode only the newly generated tokens
        new_tokens = generation_output.sequences[0][input_len:]
        response_text = self.processor.decode(new_tokens, skip_special_tokens=True).strip()

        prompt = {"inputs": text, "max_new_tokens": self.max_tokens, "temperature": self.temperature}
        response = {"response": self.processor.decode(generation_output.sequences[0], skip_special_tokens=False)}
        return prompt, response, response_text

    @staticmethod
    def load_processor(model_spec: backends.ModelSpec):
        """Load AutoProcessor for multimodal models."""
        logger.info(f'Loading processor for multimodal model: {model_spec.model_name}')
        hf_model_str = model_spec['huggingface_id']
        kwargs = {}
        if model_spec.model_config.get('trust_remote_code'):
            kwargs['trust_remote_code'] = True
        if model_spec.model_config.get('requires_api_key'):
            key = KeyRegistry.from_json().get_key_for("huggingface")
            kwargs['token'] = key['api_key']
        return AutoProcessor.from_pretrained(hf_model_str, **kwargs)

    @staticmethod
    def _unpack_images_from_messages(messages: List[Dict]) -> Tuple[List[Dict], List]:
        """Convert clemcore messages to HF processor format, separating embedded images into a list.

        Clemcore embeds images directly in messages as a string path/URL (or list of them) under the
        'image' key. HF processors expect images as a flat list alongside messages where image slots
        are marked with {"type": "image"} content entries. This method performs both transformations.
        """
        hf_messages = []
        images = []
        for msg in messages:
            content = []
            if 'image' in msg:
                image_field = msg['image']
                if isinstance(image_field, str):
                    content.append({"type": "image"})
                    images.append(load_image(image_field))
                elif isinstance(image_field, list):
                    for img in image_field:
                        content.append({"type": "image"})
                        images.append(load_image(img))
            content.append({"type": "text", "text": msg['content']})
            hf_messages.append({"role": msg['role'], "content": content})
        return hf_messages, images

def split_and_clean_batch_outputs(
        model: HuggingfaceLocalModel,
        model_outputs: List[str],
        prompt_texts: List[str]
) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, str]]]:
    """
    Processes a batch of raw model output strings by removing input prompts,
    trimming any configured output prefixes, and cleaning up end-of-sequence tokens.

    Args:
        model: The HuggingfaceLocalModel instance containing model configuration and settings.
        model_outputs: List of raw generated output strings from the model (batch).
        prompt_texts: List of prompt strings corresponding to each model output in the batch.

    Returns:
        Tuple of three lists (prompts, response_texts, responses):
        - prompts: List of dicts with prompt information (inputs, max_new_tokens, temperature, etc.).
        - response_texts: List of cleaned response strings, with prompts removed and special tokens trimmed.
        - responses: List of dicts containing the raw model output strings under the key 'response'.
    """
    prompts = []
    responses = []
    response_texts = []

    for model_output, prompt_text in zip(model_outputs, prompt_texts):
        # Remove prompt from output
        response_text = model_output.replace(prompt_text, '').strip()
        # Apply model-specific output_split_prefix if present
        if 'output_split_prefix' in model.model_spec.model_config:
            prefix = model.model_spec.model_config['output_split_prefix']
            if prefix in response_text:
                response_text = response_text.rsplit(prefix, maxsplit=1)[1]
        # Remove batch processing padding tokens
        if response_text.startswith(model.tokenizer.pad_token) or response_text.endswith(model.tokenizer.pad_token):
            response_text = response_text.replace(model.tokenizer.pad_token, "")
        # Remove EOS tokens and potential trailing tokens from response
        eos_to_cull = model.model_spec.model_config['eos_to_cull']  # This is a regEx to handle inconsistent outputs
        response_text = re.sub(eos_to_cull, "", response_text)

        # Check for CoT output and split if present
        if 'cot_output' in model.model_spec.model_config and model.model_spec.model_config['cot_output']:
            cot_content, response_text = split_and_clean_cot_output(response_text, model)

        # Prompt and response info for recording raw model inputs and outputs
        prompt_info = {
            "inputs": prompt_text,
            "max_new_tokens": model.max_tokens,
            "temperature": model.temperature
        }
        response_info = {'response': model_output}
        # Add cot_content content to response_info
        if 'cot_output' in model.model_spec.model_config and model.model_spec.model_config['cot_output']:
            response_info['cot_content'] = cot_content

        prompts.append(prompt_info)
        responses.append(response_info)
        response_texts.append(response_text)
    return prompts, response_texts, responses


def split_and_clean_cot_output(response_text: str, model: HuggingfaceLocalModel) -> Tuple[str, str]:
    """
    Splits a CoT-output model's response into cot_content and final answer.
    Final answers are cut to the token sequence allowed by the max_tokens value set for the model/benchmark run due to
    fairness concerns.
    CoT tags, stored in the model's registry entry, cover more than just relevant special tokens to assure broad
    applicability through string splitting. For example, the CoT end tag for gpt-oss models is
    '<|end|><|start|>assistant<|channel|>final<|message|>' (the relevant part being '<|channel|>final'), as it includes
    the non-special-token string 'final' between special tokens, with the *entire tag string* demarcating the beginning
    of the final answer, instead of a simple single special token like for example DeepSeek's '</thinking>'.

    Args:
        response_text: The response text, without input prompt, but including all special tokens and tags.
        model: The HuggingfaceLocalModel instance containing model configuration and settings.
    Returns:
        Tuple of two strings:
        - cot_content: The cleaned CoT/thinking/reasoning/cot_content content.
        - answer: The cleaned final answer content.
    """
    # Cull CoT start tag if model has it defined
    if 'cot_start_tag' in model.model_spec.model_config and model.model_spec.model_config['cot_start_tag']:
        response_text = response_text.replace(model.model_spec.model_config['cot_start_tag'], "")
    # Split response text at CoT end tag
    # split_cot_response = response_text.split(model.model_spec.model_config['cot_end_tag'])
    split_cot_response = re.split(model.model_spec.model_config['cot_end_tag'], response_text)
    cot_content = split_cot_response[0]
    # Handle empty CoT outputs
    if len(split_cot_response) >= 2:
        answer = split_cot_response[-1]
    else:
        answer = ""
    # Retokenize and count CoT and final answer tokens
    # tokenized_cot_content = model.tokenizer(cot_content)
    # n_cot_content_tokens = len(tokenized_cot_content)
    tokenized_answer = model.tokenizer(answer)
    tokenized_answer = tokenized_answer.input_ids
    n_answer_tokens = len(tokenized_answer)
    # Cut answer tokens to max_tokens value if they exceed it
    if n_answer_tokens > model.max_tokens:
        logger.info(f"CoT final answer token count {n_answer_tokens} exceeds max_tokens {model.max_tokens}, "
                    f"cutting off excess tokens.")
        tokenized_answer = tokenized_answer[:model.max_tokens]
    # Decode retokenized and potentially cut answer
    answer = model.tokenizer.decode(tokenized_answer)
    # Strip answer to assure proper clemgame parsing
    answer = answer.strip()

    return cot_content, answer


def assert_context_limits(model: HuggingfaceLocalModel, prompt_token_ids):
    for i in range(prompt_token_ids.size(0)):
        context_check = _check_context_limit(
            model.context_size,
            prompt_token_ids[i],
            max_new_tokens=model.max_tokens
        )
        if not context_check[0]:
            logger.info(f"Context token limit for {model.model_spec.model_name} exceeded on batch index {i}: "
                        f"{context_check[1]}/{context_check[3]}")
            raise ContextExceededError(
                f"Context token limit for {model.model_spec.model_name} exceeded at batch index {i}",
                tokens_used=context_check[1],
                tokens_left=context_check[2],
                context_size=context_check[3]
            )


def _check_context_limit(context_size, prompt_tokens, max_new_tokens: int = 100) -> Tuple[bool, int, int, int]:
    """Internal context limit check to run in generate_response.
    Args:
        prompt_tokens: List of prompt token IDs.
        max_new_tokens: How many tokens to generate ('at most', but no stop sequence is defined).
    Returns:
        Tuple with
            Bool: True if context limit is not exceeded, False if too many tokens
            Number of tokens for the given messages and maximum new tokens
            Number of tokens of 'context space left'
            Total context token limit
    """
    prompt_size = len(prompt_tokens)
    tokens_used = prompt_size + max_new_tokens  # context includes tokens to be generated
    tokens_left = context_size - tokens_used
    fits = tokens_used <= context_size
    return fits, tokens_used, tokens_left, context_size


def check_messages(messages: List[Dict], model_spec: backends.ModelSpec) -> bool:
    """Message checking for clemgame development.
    This checks if the model's chat template accepts the given messages as passed, before the standard flattening done
    for generation. This allows clemgame developers to construct message lists that are sound as-is and are not affected
    by the indiscriminate flattening of the generation method. Deliberately verbose.
    Args:
        model_spec: The ModelSpec for the model.
        messages: for example
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
                {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                {"role": "user", "content": "Where was it played?"}
            ]
    Returns:
        True if messages are sound as-is, False if messages are not compatible with the model's template.
    """
    tokenizer, _, _ = load_config_and_tokenizer(model_spec)

    # bool for message acceptance:
    messages_accepted: bool = True

    # check for system message:
    has_system_message: bool = False
    if messages[0]['role'] == "system":
        print("System message detected.")
        has_system_message = True
        if not messages[0]['content']:
            print(f"Initial system message is empty. It will be removed when generating responses.")
        else:
            print(f"Initial system message has content! It will not be removed when generating responses. This "
                  f"will lead to issues with models that do not allow system messages.")
        """
        print("Checking model system message compatibility...")
        # unfortunately Mistral models, which do not accept system message, currently do not raise a distinct 
        # exception for this...
        try:
            self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        except TemplateError:
            print("The model's chat template does not allow for system message!")
            messages_accepted = False
        """

    # check for message order:
    starts_with_assistant: bool = False
    double_user: bool = False
    double_assistant: bool = False
    ends_with_assistant: bool = False

    for msg_idx, message in enumerate(messages):
        if not has_system_message:
            if msg_idx == 0 and message['role'] == "assistant":
                starts_with_assistant = True
        else:
            if msg_idx == 1 and message['role'] == "assistant":
                starts_with_assistant = True
        if msg_idx > 0 and message['role'] == "user" and messages[msg_idx - 1]['role'] == "user":
            double_user = True
        elif msg_idx > 0 and message['role'] == "assistant" and messages[msg_idx - 1]['role'] == "assistant":
            double_assistant = True
    if messages[-1]['role'] == "assistant":
        ends_with_assistant = True

    if starts_with_assistant or double_user or double_assistant or ends_with_assistant:
        print("Message order issue(s) found:")
        if starts_with_assistant:
            print("First message has role:'assistant'.")
        if double_user:
            print("Messages contain consecutive user messages.")
        if double_assistant:
            print("Messages contain consecutive assistant messages.")
        if ends_with_assistant:
            print("Last message has role:'assistant'.")

    # proper check of chat template application:
    try:
        tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    except TemplateError:
        print(f"The {model_spec.model_name} chat template does not accept these messages! "
              f"Cleaning applied before generation might still allow these messages, but is indiscriminate and "
              f"might lead to unintended generation inputs.")
        messages_accepted = False
    else:
        print(
            f"The {model_spec.model_name} chat template accepts these messages. Cleaning before generation is still "
            f"applied to these messages, which is indiscriminate and might lead to unintended generation inputs.")

    return messages_accepted


def check_context_limit(messages: List[Dict], model_spec: backends.ModelSpec,
                        max_new_tokens: int = 100, clean_messages: bool = False,
                        verbose: bool = True) -> Tuple[bool, int, int, int]:
    """Externally-callable context limit check for clemgame development.
    Args:
        messages: for example
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
                {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                {"role": "user", "content": "Where was it played?"}
            ]
        model_spec: The ModelSpec for the model.
        max_new_tokens: How many tokens to generate ('at most', but no stop sequence is defined).
        clean_messages: If True, the standard cleaning method for message lists will be applied.
        verbose: If True, prettyprint token counts.
    Returns:
        Tuple with
            Bool: True if context limit is not exceeded, False if too many tokens
            Number of tokens for the given messages and maximum new tokens
            Number of tokens of 'context space left'
            Total context token limit
    """
    tokenizer, _, context_size = load_config_and_tokenizer(model_spec)

    # optional messages processing:
    if clean_messages:
        current_messages = ensure_alternating_roles(messages)
    else:
        current_messages = messages
    # the actual tokens, including chat format:
    # transformers >=5: apply_chat_template returns BatchEncoding when tokenize=True
    prompt_tokens = tokenizer.apply_chat_template(current_messages, add_generation_prompt=True)
    if hasattr(prompt_tokens, 'input_ids'):
        prompt_tokens = prompt_tokens.input_ids
    context_check_tuple = _check_context_limit(context_size, prompt_tokens, max_new_tokens=max_new_tokens)
    tokens_used = context_check_tuple[1]
    tokens_left = context_check_tuple[2]
    if verbose:
        print(f"{tokens_used} input tokens, {tokens_left} tokens of {context_size} left.")
    fits = context_check_tuple[0]
    return fits, tokens_used, tokens_left, context_size
