import unittest

from clemcore.backends import ModelRegistry, BackendRegistry
from clemcore.backends.utils import ensure_alternating_roles


class UtilsTestCase(unittest.TestCase):

    def test_ensure_alternating_roles_with_empty_system_removed_if_empty(self):
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Initial Prompt"},
            {"role": "assistant", "content": "Turn 1"},
        ]
        _messages = ensure_alternating_roles(messages)
        self.assertEqual(messages, messages)
        self.assertEqual(_messages, [
            {"role": "user", "content": "Initial Prompt"},
            {"role": "assistant", "content": "Turn 1"},
        ])

    def test_ensure_alternating_roles_with_empty_system_keeps_system_if_wanted(self):
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Initial Prompt"},
            {"role": "assistant", "content": "Turn 1"},
        ]
        _messages = ensure_alternating_roles(messages, cull_system_message=False)
        self.assertEqual(messages, messages)
        self.assertEqual(_messages, messages)

    def test_ensure_alternating_roles_with_system_keeps_system(self):
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Initial Prompt"},
            {"role": "assistant", "content": "Turn 1"},
        ]
        _messages = ensure_alternating_roles(messages)
        self.assertEqual(messages, messages)
        self.assertEqual(_messages, messages)

    def test_ensure_alternating_roles_with_doubled_assistant(self):
        messages = [
            {"role": "user", "content": "Initial Prompt"},
            {"role": "assistant", "content": "Turn 1"},
            {"role": "assistant", "content": "Response 1"}
        ]
        _messages = ensure_alternating_roles(messages)
        self.assertEqual(messages, messages)
        self.assertEqual(_messages, [
            {"role": "user", "content": "Initial Prompt"},
            {"role": "assistant", "content": "Turn 1\n\nResponse 1"}
        ]
                         )

    def test_ensure_alternating_roles_with_doubled_user(self):
        messages = [
            {"role": "user", "content": "Initial Prompt"},
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Response 1"}
        ]
        _messages = ensure_alternating_roles(messages)
        self.assertEqual(messages, messages)
        self.assertEqual(_messages, [
            {"role": "user", "content": "Initial Prompt\n\nTurn 1"},
            {"role": "assistant", "content": "Response 1"}
        ]
                         )

    def test_ensure_alternating_roles_with_triple_user(self):
        messages = [
            {"role": "user", "content": "Initial Prompt"},
            {"role": "user", "content": "Turn 1a"},
            {"role": "user", "content": "Turn 1b"},
            {"role": "assistant", "content": "Response 1"}
        ]
        _messages = ensure_alternating_roles(messages)
        self.assertEqual(messages, messages)
        self.assertEqual(_messages, [
            {"role": "user", "content": "Initial Prompt\n\nTurn 1a\n\nTurn 1b"},
            {"role": "assistant", "content": "Response 1"}
        ]
                         )

    def test_ensure_alternating_roles_with_doubled_both(self):
        messages = [
            {"role": "user", "content": "Initial Prompt"},
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "assistant", "content": "Turn 2"}
        ]
        _messages = ensure_alternating_roles(messages)
        self.assertEqual(messages, messages)
        self.assertEqual(_messages, [
            {"role": "user", "content": "Initial Prompt\n\nTurn 1"},
            {"role": "assistant", "content": "Response 1\n\nTurn 2"}
        ]
                         )


MODELS_LIST = [
    {
        "model_name": "model1",
        "backend": "huggingface_local",
        "model_id": "model_id1",
        "custom_chat_template": "custom_chat_template",
        "eos_to_cull": "<|end_of_turn|>"
    }, {
        "model_name": "model2",
        "backend": "huggingface_local",
        "model_id": "model_id2",
        "custom_chat_template": "custom_chat_template",
        "eos_to_cull": "<|end_of_turn|>"
    }, {
        "model_name": "model1",
        "backend": "openai",
        "model_id": "model_id3",
        "custom_chat_template": "custom_chat_template",
        "eos_to_cull": "<|end_of_turn|>"
    }
]


class ModelTestCase(unittest.TestCase):
    def test_get_backend_for_model1(self):
        model_registry = ModelRegistry().register_from_list(MODELS_LIST)
        model_spec = model_registry.get_first_model_spec_that_unify_with("model1")
        assert model_spec.backend == "huggingface_local"

    def test_get_backend_for_model2(self):
        model_registry = ModelRegistry().register_from_list(MODELS_LIST)
        model_spec = model_registry.get_first_model_spec_that_unify_with("model2")
        assert model_spec.backend == "huggingface_local"

    def test_get_backend_for_model1_other(self):
        model_registry = ModelRegistry().register_from_list(MODELS_LIST)
        model_spec = model_registry.get_first_model_spec_that_unify_with(dict(model_name="model1", backend="openai"))
        assert model_spec.backend == "openai"


if __name__ == '__main__':
    unittest.main()
