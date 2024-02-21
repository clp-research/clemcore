import unittest

from backends import get_model_for, load_model_registry


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
