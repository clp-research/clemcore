"""Integration tests for model registry entries.

Tests that all models in model_registry.json (except huggingface, vllm, openrouter, llamacpp)
can be loaded and their backends instantiated.
"""
import inspect
import unittest
from pathlib import Path

from clemcore.backends import ModelRegistry

# Backends to exclude from integration testing (require local GPU or specific setup)
EXCLUDED_BACKENDS = {
    "huggingface_local",
    "huggingface_multimodal",
    "vllm",
    "openrouter",
    "llamacpp",
    "slurk"
}


def get_class_package_path(_class):
    class_file = inspect.getfile(_class)
    package_dir = Path(class_file).parent
    if "site-packages" in str(package_dir):
        print("WARNING: Loading registry from INSTALLED package, not local source!")
    return package_dir


def get_local_model_specs():
    package_path = get_class_package_path(ModelRegistry)
    registry = ModelRegistry.from_directory(package_path)
    return registry.model_specs


def get_testable_model_specs():
    """Get model specs that should be tested (excluding certain backends)."""
    model_specs = get_local_model_specs()
    return [spec for spec in model_specs if spec.backend not in EXCLUDED_BACKENDS]


class TestPackagedModelRegistry(unittest.TestCase):
    """Test that model registry entries can be loaded and resolved."""

    @classmethod
    def setUpClass(cls):
        """Load model specs and backend registry once for all tests."""
        cls.model_specs = get_testable_model_specs()

    def test_model_specs_loaded(self):
        """Verify that we have model specs to test."""
        self.assertGreater(len(self.model_specs), 0, "Should have at least one model spec to test")

    def test_model_registry_unification(self):
        """Test that all models can be resolved via ModelRegistry unification."""
        registry = ModelRegistry().register_from_list(
            [spec.to_dict() for spec in self.model_specs]
        )
        for spec in self.model_specs:
            with self.subTest(model_name=spec.model_name):
                unified = registry.get_first_model_spec_that_unify_with(spec.model_name)
                self.assertIsNotNone(unified)
                self.assertEqual(unified.model_name, spec.model_name)
                self.assertTrue(unified.has_backend())


class TestModelSpecValidation(unittest.TestCase):
    """Test that model specs have required fields."""

    @classmethod
    def setUpClass(cls):
        cls.model_specs = get_testable_model_specs()

    def test_all_specs_have_model_name(self):
        """All specs should have a model_name."""
        for spec in self.model_specs:
            with self.subTest(spec=str(spec)):
                self.assertTrue(hasattr(spec, "model_name"))
                self.assertIsNotNone(spec.model_name)

    def test_all_specs_have_backend(self):
        """All specs should have a backend."""
        for spec in self.model_specs:
            with self.subTest(model_name=spec.model_name):
                self.assertTrue(spec.has_backend())
                self.assertIsNotNone(spec.backend)

    def test_all_specs_have_model_id_or_equivalent(self):
        """All specs should have model_id or huggingface_id."""
        for spec in self.model_specs:
            with self.subTest(model_name=spec.model_name):
                has_model_id = hasattr(spec, "model_id") and spec.model_id
                has_hf_id = hasattr(spec, "huggingface_id") and spec.huggingface_id
                # slurk backend doesn't require model_id
                if spec.backend == "slurk":
                    continue
                self.assertTrue(
                    has_model_id or has_hf_id,
                    f"Model '{spec.model_name}' should have model_id or huggingface_id"
                )


class TestExcludedBackends(unittest.TestCase):
    """Verify that excluded backends are correctly filtered out."""

    def test_excluded_backends_not_in_testable_specs(self):
        """Verify excluded backends are filtered from testable specs."""
        testable_specs = get_testable_model_specs()
        testable_backends = {spec.backend for spec in testable_specs}

        for excluded in EXCLUDED_BACKENDS:
            self.assertNotIn(
                excluded,
                testable_backends,
                f"Backend '{excluded}' should be excluded from tests"
            )

    def test_all_specs_loaded_includes_excluded(self):
        """Verify that the full registry includes excluded backends."""
        all_specs = get_local_model_specs()
        all_backends = {spec.backend for spec in all_specs}

        # At least some excluded backends should exist in the full registry
        found_excluded = EXCLUDED_BACKENDS.intersection(all_backends)
        self.assertGreater(
            len(found_excluded), 0,
            "Should find at least one excluded backend in full registry"
        )


if __name__ == "__main__":
    unittest.main()
