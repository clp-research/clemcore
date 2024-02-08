import unittest

import backends
from backends.huggingface_local_api import check_messages, check_context_limit

MODEL_SPEC = backends.ModelSpec(**{
    "model_name": "Mistral-7B-Instruct-v0.1",
    "backend": "huggingface_local",
    "huggingface_id": "mistralai/Mistral-7B-Instruct-v0.1",
    "premade_chat_template": True,
    "eos_to_cull": "</s>"
})


class HuggingfaceLocalTestCase(unittest.TestCase):

    def test_proper_minimal_messages(self):
        minimal_messages = [
            {"role": "user", "content": "What is your favourite condiment?"},
            {"role": "assistant", "content": "Lard!"},
            {"role": "user", "content": "Do you have mayonnaise recipes?"}
        ]
        # check proper minimal messages with Mistral-7B-Instruct-v0.1:
        print("Minimal messages:")
        check_messages(minimal_messages, MODEL_SPEC)

    def test_improper_double_user_messages(self):
        double_user_messages = [
            {"role": "user", "content": "Hello there!"},
            {"role": "user", "content": "What is your favourite condiment?"},
            {"role": "assistant", "content": "Lard!"},
            {"role": "user", "content": "Do you have mayonnaise recipes?"}
        ]
        # check improper double user messages with Mistral-7B-Instruct-v0.1:
        print("Double user messages:")
        check_messages(double_user_messages, MODEL_SPEC)

    def test_improper_first_assistant_message(self):
        first_assistant_messages = [
            {"role": "assistant", "content": "Hello there!"},
            {"role": "user", "content": "What is your favourite condiment?"},
            {"role": "assistant", "content": "Lard!"},
            {"role": "user", "content": "Do you have mayonnaise recipes?"}
        ]
        # check improper first assistant message with Mistral-7B-Instruct-v0.1:
        print("First message role assistant:")
        check_messages(first_assistant_messages, MODEL_SPEC)

    def test_system_message(self):
        system_messages = [
            {"role": "system", "content": "You love all kinds of fat."},
            {"role": "user", "content": "What is your favourite condiment?"},
            {"role": "assistant", "content": "Lard!"},
            {"role": "user", "content": "Do you have mayonnaise recipes?"}
        ]
        # check system message with Mistral-7B-Instruct-v0.1:
        print("System message:")
        check_messages(system_messages, MODEL_SPEC)

    def test_empty_system_message(self):
        empty_system_messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What is your favourite condiment?"},
            {"role": "assistant", "content": "Lard!"},
            {"role": "user", "content": "Do you have mayonnaise recipes?"}
        ]
        # check empty system message with Mistral-7B-Instruct-v0.1:
        print("Empty system message:")
        check_messages(empty_system_messages, MODEL_SPEC)

    def test_context_limit_check_with_minimal_messages(self):
        minimal_messages = [
            {"role": "user", "content": "What is your favourite condiment?"},
            {"role": "assistant", "content": "Lard!"},
            {"role": "user", "content": "Do you have mayonnaise recipes?"}
        ]
        # check minimal messages with Mistral-7B-Instruct-v0.1:
        print("Minimal messages context check with Mistral-7B-Instruct-v0.1:")
        minimal_context_check_tuple = check_context_limit(minimal_messages, MODEL_SPEC)
        print(f"Minimal messages context check output: {minimal_context_check_tuple}")

    def test_context_limit_check_with_many_messages(self):
        excessive_messages = list()
        for _ in range(2000):
            excessive_messages.append({"role": "user", "content": "What is your favourite condiment?"})
            excessive_messages.append({"role": "assistant", "content": "Lard!"})
        excessive_messages.append({"role": "user", "content": "Do you have mayonnaise recipes?"})
        # check excessive messages with Mistral-7B-Instruct-v0.1:
        print("Excessive messages context check with Mistral-7B-Instruct-v0.1:")
        excessive_context_check_tuple = check_context_limit(excessive_messages, MODEL_SPEC)
        print(f"Excessive messages context check output: {excessive_context_check_tuple}")
        """Note: Mistral-7B-Instruct-v0.1 has an official context limit of 32768, and while the context limit checks might 
        pass, using the full context of models with large limits like this is likely to use a great amount of memory (VRAM) 
        which can lead to CUDA Out-Of-Memory errors that are not only hard to handle, but can also incapacitate shared 
        hardware until it is manually reset. Please test for this while developing clemgames to prevent hardware outages 
        when the full set of clemgames is run by others."""


if __name__ == '__main__':
    unittest.main()
