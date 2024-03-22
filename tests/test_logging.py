import unittest

from backends import get_model_for, load_model_registry


class TabooTestCase(unittest.TestCase):

    def test_get_model_for_huggingface_local_logs_infos(self):
        load_model_registry()
        get_model_for("llama-2-7b-chat-hf")
