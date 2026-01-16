import unittest

from clemcore.backends.backend_registry import (
    is_backend_file,
    to_backend_name,
    is_backend,
    Backend,
    BackendRegistry,
    HumanModelBackend,
    CustomResponseModelBackend,
)
from clemcore.backends import ModelSpec


class BackendUtilsTestCase(unittest.TestCase):

    def test_is_backend_file_true(self):
        """Test that _api.py files are recognized as backend files."""
        self.assertTrue(is_backend_file("openai_api.py"))
        self.assertTrue(is_backend_file("anthropic_api.py"))
        self.assertTrue(is_backend_file("my_custom_api.py"))

    def test_is_backend_file_false(self):
        """Test that non-backend files are not recognized."""
        self.assertFalse(is_backend_file("openai.py"))
        self.assertFalse(is_backend_file("api.py"))
        self.assertFalse(is_backend_file("backend.py"))
        self.assertFalse(is_backend_file("openai_api.txt"))
        self.assertFalse(is_backend_file(""))

    def test_to_backend_name(self):
        """Test conversion from filename to backend name."""
        self.assertEqual(to_backend_name("openai_api.py"), "openai")
        self.assertEqual(to_backend_name("anthropic_api.py"), "anthropic")
        self.assertEqual(to_backend_name("my_custom_backend_api.py"), "my_custom_backend")

    def test_is_backend_with_class(self):
        """Test is_backend with Backend subclass."""
        class MyBackend(Backend):
            def get_model_for(self, model_spec):
                pass

        self.assertTrue(is_backend(MyBackend))

    def test_is_backend_with_base_class(self):
        """Test that Backend base class itself is not recognized."""
        self.assertFalse(is_backend(Backend))

    def test_is_backend_with_non_backend(self):
        """Test is_backend with non-Backend classes."""
        self.assertFalse(is_backend(str))
        self.assertFalse(is_backend(int))
        self.assertFalse(is_backend(object))

    def test_is_backend_with_instance(self):
        """Test that instances are not recognized as backends."""
        self.assertFalse(is_backend("string"))
        self.assertFalse(is_backend(123))


class BackendRegistryTestCase(unittest.TestCase):

    def test_registry_len(self):
        """Test registry length."""
        registry = BackendRegistry([
            {"backend": "openai", "file_name": "openai_api.py", "file_path": "/path", "lookup_source": "test"}
        ])
        # +2 for internal human and programmatic backends
        self.assertEqual(len(registry), 3)

    def test_registry_iter(self):
        """Test iterating over registry."""
        backend_files = [
            {"backend": "openai", "file_name": "openai_api.py", "file_path": "/path", "lookup_source": "test"}
        ]
        registry = BackendRegistry(backend_files)
        backends = list(registry)
        backend_names = [b["backend"] for b in backends]
        self.assertIn("openai", backend_names)
        self.assertIn("_player_human", backend_names)
        self.assertIn("_player_programmed", backend_names)

    def test_registry_is_supported_true(self):
        """Test that registered backends are supported."""
        backend_files = [
            {"backend": "openai", "file_name": "openai_api.py", "file_path": "/path", "lookup_source": "test"}
        ]
        registry = BackendRegistry(backend_files)
        self.assertTrue(registry.is_supported("openai"))
        self.assertTrue(registry.is_supported("_player_human"))
        self.assertTrue(registry.is_supported("_player_programmed"))

    def test_registry_is_supported_false(self):
        """Test that unregistered backends are not supported."""
        registry = BackendRegistry([])
        self.assertFalse(registry.is_supported("nonexistent"))

    def test_registry_get_first_file_matching(self):
        """Test getting first matching backend file."""
        backend_files = [
            {"backend": "openai", "file_name": "openai_api.py", "file_path": "/path/1", "lookup_source": "cwd"},
            {"backend": "openai", "file_name": "openai_api.py", "file_path": "/path/2", "lookup_source": "packaged"}
        ]
        registry = BackendRegistry(backend_files)
        match = registry.get_first_file_matching("openai")
        self.assertEqual(match["file_path"], "/path/1")

    def test_registry_get_first_file_matching_not_found(self):
        """Test that ValueError is raised for unknown backend."""
        registry = BackendRegistry([])
        with self.assertRaises(ValueError) as context:
            registry.get_first_file_matching("nonexistent")
        self.assertIn("nonexistent", str(context.exception))

    def test_registry_get_backend_for_human(self):
        """Test getting human model backend."""
        registry = BackendRegistry([])
        backend = registry.get_backend_for("_player_human")
        self.assertIsInstance(backend, HumanModelBackend)

    def test_registry_get_backend_for_programmed(self):
        """Test getting programmatic model backend."""
        registry = BackendRegistry([])
        backend = registry.get_backend_for("_player_programmed")
        self.assertIsInstance(backend, CustomResponseModelBackend)

    def test_registry_from_packaged_finds_backends(self):
        """Test that from_packaged_and_cwd_files finds packaged backends."""
        registry = BackendRegistry.from_packaged_and_cwd_files()
        # Should find at least some packaged backends
        self.assertGreater(len(registry), 2)  # More than just human and programmed
        # Check that common backends are found
        backend_names = [b["backend"] for b in registry]
        # At least some of these should be present
        common_backends = {"openai", "anthropic", "google", "mistral", "cohere"}
        found = common_backends.intersection(set(backend_names))
        self.assertGreater(len(found), 0, "Should find at least one common backend")


class HumanModelBackendTestCase(unittest.TestCase):

    def test_get_model_for_human(self):
        """Test getting model for human player."""
        backend = HumanModelBackend()
        model_spec = ModelSpec(model_name="human", backend="_player_human")
        model = backend.get_model_for(model_spec)
        self.assertIsNotNone(model)

    def test_get_model_for_non_human_raises(self):
        """Test that getting model for non-human spec raises."""
        backend = HumanModelBackend()
        model_spec = ModelSpec(model_name="gpt-4", backend="openai")
        with self.assertRaises(ValueError):
            backend.get_model_for(model_spec)


class CustomResponseModelBackendTestCase(unittest.TestCase):

    def test_get_model_for_programmatic(self):
        """Test getting model for programmatic player."""
        backend = CustomResponseModelBackend()
        model_spec = ModelSpec(model_name="programmatic", backend="_player_programmed")
        model = backend.get_model_for(model_spec)
        self.assertIsNotNone(model)

    def test_get_model_for_non_programmatic_raises(self):
        """Test that getting model for non-programmatic spec raises."""
        backend = CustomResponseModelBackend()
        model_spec = ModelSpec(model_name="gpt-4", backend="openai")
        with self.assertRaises(ValueError):
            backend.get_model_for(model_spec)


class BackendStrRepresentationTestCase(unittest.TestCase):

    def test_backend_str(self):
        """Test Backend string representation."""
        backend = HumanModelBackend()
        self.assertEqual(str(backend), "HumanModelBackend")

    def test_backend_repr(self):
        """Test Backend repr."""
        backend = CustomResponseModelBackend()
        self.assertEqual(repr(backend), "CustomResponseModelBackend")


if __name__ == '__main__':
    unittest.main()
