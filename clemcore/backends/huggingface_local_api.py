"""Backend using HuggingFace transformers models.
Uses HF tokenizers instruct/chat templates for proper input format per model.
"""
import logging
from typing import List, Dict, Tuple, Any
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, PreTrainedTokenizerBase, PreTrainedModel
from transformers.generation.utils import GenerateOutput
from peft import PeftModel
from jinja2 import TemplateError

import clemcore.backends as backends
from clemcore.backends.key_registry import KeyRegistry
from clemcore.backends.utils import ensure_alternating_roles, ensure_messages_format, augment_response_object, \
    ContextExceededError

logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("clemcore.cli")

FALLBACK_CONTEXT_SIZE = 256


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
            print(requires_api_key_info)
            logger.info(requires_api_key_info)

    hf_model_str = model_spec['huggingface_id']

    # use 'slow' tokenizer for models that require it:
    if 'slow_tokenizer' in model_spec.model_config:
        if model_spec['model_config']['slow_tokenizer']:
            tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                hf_model_str,
                device_map="auto",
                torch_dtype="auto",
                verbose=False,
                use_fast=False
            )
        else:
            tokenizer = None
            slow_tokenizer_info = (f"{model_spec['model_name']} registry setting has slow_tokenizer, "
                                   f"but it is not 'true'. Please check the model entry.")
            print(slow_tokenizer_info)
            logger.info(slow_tokenizer_info)
    elif use_api_key:
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            hf_model_str,
            token=api_key,
            device_map="auto",
            torch_dtype="auto",
            verbose=False
        )
    else:
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            hf_model_str,
            device_map="auto",
            torch_dtype="auto",
            verbose=False
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
    if hasattr(auto_config, 'max_position_embeddings'):  # this is the standard attribute used by most
        context_size = auto_config.max_position_embeddings
    elif hasattr(auto_config, 'n_positions'):  # some models may have their context size under this attribute
        context_size = auto_config.n_positions
    else:  # few models, especially older ones, might not have their context size in the config
        context_size = FALLBACK_CONTEXT_SIZE

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
        tokenizer.padding_side = "left" if auto_config.is_decoder and not auto_config.is_encoder_decoder else "right"
        stdout_logger.warning("Derive padding_size=%s from model architecture (decoder=%s, encoder-decoder=%s)",
                              tokenizer.padding_side, auto_config.is_decoder, auto_config.is_encoder_decoder)
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
    if "load_in_8bit" in model_spec.model_config:
        model_args["load_in_8bit"] = model_spec.model_config["load_in_8bit"]
    if "load_in_4bit" in model_spec.model_config:
        model_args["load_in_4bit"] = model_spec.model_config["load_in_4bit"]
    if 'requires_api_key' in model_spec.model_config and model_spec['model_config']['requires_api_key']:
        # load HF API key:
        key = KeyRegistry.from_json().get_key_for("huggingface")
        model_args["token"] = key["api_key"]

    hf_model_str = model_spec['huggingface_id']
    model = AutoModelForCausalLM.from_pretrained(hf_model_str, **model_args)

    if "peft_model" in model_spec.model_config:
        adapter_model = model_spec.model_config["peft_model"]  # can be a path or name
        stdout_logger.info(f"Load PeftModel adapters from {adapter_model}")
        model = PeftModel.from_pretrained(model, adapter_model)

    logger.info(f"Finished loading huggingface model: {model_spec.model_name}")
    logger.info(f"Model device map: {model.hf_device_map}")

    return model


class HuggingfaceLocal(backends.Backend):
    """Model/backend handler class for locally-run Huggingface models."""

    def __init__(self):
        super().__init__()

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        """Get a HuggingFaceLocalModel instance with the passed model and settings.
        Will load all required data for using the model upon initialization.
        Args:
            model_spec: The ModelSpec for the model.
        Returns:
            The Model class instance of the model.
        """
        torch.set_num_threads(1)
        return HuggingfaceLocalModel(model_spec)


class HuggingfaceLocalModel(backends.BatchGenerativeModel):
    """Class for loaded HuggingFace transformers models ready for generation."""

    def __init__(self, model_spec: backends.ModelSpec):
        """
        Args:
            model_spec: A ModelSpec instance specifying the model.
        """
        super().__init__(model_spec)
        # fail-fast
        self.tokenizer, self.config, self.context_size = load_config_and_tokenizer(model_spec)
        self.model: PreTrainedModel = load_model(model_spec)

        # check if model's generation_config has pad_token_id set:
        if not self.model.generation_config.pad_token_id:
            # set pad_token_id to tokenizer's eos_token_id to prevent excessive warnings:
            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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
                tokenize=False  # get back the rendered string
            )
        elif 'cot_effort' in self.model_spec.model_config and self.model_spec.model_config['cot_effort']:
            # Render each chat in the batch (list of messages) to a string prompt with generation prompt
            # including setting CoT effort to value defined in model registry entry
            # NOTE: Currently this is custom code to handle gpt-oss models! Other models that have CoT effort setting
            # training might not pass the value to the model and thus template in the same way. Using this for those
            # models will likely lead to errors!
            rendered_chats = self.tokenizer.apply_chat_template(
                batch_messages,
                add_generation_prompt=True,  # append assistant prompt
                tokenize=False,  # get back the rendered string
                reasoning_effort=self.model_spec.model_config['cot_effort']  # use string from model config
            )
        else:
            # Render each chat in the batch (list of messages) to a string prompt with generation prompt
            rendered_chats = self.tokenizer.apply_chat_template(
                batch_messages,
                add_generation_prompt=True,  # append assistant prompt
                tokenize=False  # get back the rendered string
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
        self.model.eval()

        # Generate outputs for the whole batch (Note: model.generate() is decorated with torch.no_grad() !)
        generation_output: GenerateOutput = self.model.generate(prompt_token_ids, **gen_args)

        # Decode all outputs and prompts
        model_outputs = self.tokenizer.batch_decode(generation_output.sequences)
        prompt_texts = self.tokenizer.batch_decode(prompt_token_ids)

        prompts, response_texts, responses = split_and_clean_batch_outputs(self,
                                                                           model_outputs,
                                                                           prompt_texts)
        return list(zip(prompts, responses, response_texts))


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
    prompt_tokens = tokenizer.apply_chat_template(current_messages, add_generation_prompt=True)
    context_check_tuple = _check_context_limit(context_size, prompt_tokens, max_new_tokens=max_new_tokens)
    tokens_used = context_check_tuple[1]
    tokens_left = context_check_tuple[2]
    if verbose:
        print(f"{tokens_used} input tokens, {tokens_left} tokens of {context_size} left.")
    fits = context_check_tuple[0]
    return fits, tokens_used, tokens_left, context_size
