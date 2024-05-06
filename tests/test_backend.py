import unittest

from backends import get_model_for, load_model_registry
from backends.utils import ensure_alternating_roles


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


class ModelTestCase(unittest.TestCase):
    def test_get_backend_for_model1(self):
        load_model_registry("test-registry.json")
        model = get_model_for("model1")
        assert model.model_spec.backend == "huggingface_local"

    def test_get_backend_for_model2(self):
        load_model_registry("test-registry.json")
        model = get_model_for("model2")
        assert model.model_spec.backend == "huggingface_local"

    def test_get_backend_for_model1_other(self):
        load_model_registry("test-registry.json")
        model = get_model_for(dict(model_name="model1", backend="openai"))
        assert model.model_spec.backend == "openai"


if __name__ == '__main__':
    unittest.main()
