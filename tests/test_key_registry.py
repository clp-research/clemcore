import json
import os
import tempfile
import unittest
from pathlib import Path

from clemcore.backends.key_registry import Key, KeyRegistry


class KeyTestCase(unittest.TestCase):

    def test_key_creation(self):
        """Test basic key creation."""
        key = Key(api_key="test_key_123")
        self.assertEqual(key.api_key, "test_key_123")

    def test_key_with_extra_fields(self):
        """Test key creation with extra fields."""
        key = Key(api_key="test_key", organization="my_org", base_url="https://api.example.com")
        self.assertEqual(key.api_key, "test_key")
        self.assertEqual(key.organization, "my_org")
        self.assertEqual(key.base_url, "https://api.example.com")

    def test_key_has_api_key_true(self):
        """Test has_api_key with valid key."""
        key = Key(api_key="valid_key")
        self.assertTrue(key.has_api_key())

    def test_key_has_api_key_false_none(self):
        """Test has_api_key with None."""
        key = Key(api_key=None)
        self.assertFalse(key.has_api_key())

    def test_key_has_api_key_false_empty(self):
        """Test has_api_key with empty string."""
        key = Key(api_key="")
        self.assertFalse(key.has_api_key())

    def test_key_has_api_key_false_whitespace(self):
        """Test has_api_key with whitespace only."""
        key = Key(api_key="   ")
        self.assertFalse(key.has_api_key())

    def test_key_dict_like_access(self):
        """Test dict-like access to key attributes."""
        key = Key(api_key="test", custom_field="value")
        self.assertEqual(key["api_key"], "test")
        self.assertEqual(key["custom_field"], "value")

    def test_key_iteration(self):
        """Test iterating over key attributes."""
        key = Key(api_key="test", other="value")
        keys = list(key)
        self.assertIn("api_key", keys)
        self.assertIn("other", keys)

    def test_key_len(self):
        """Test len of key."""
        key = Key(api_key="test", field1="a", field2="b")
        self.assertEqual(len(key), 3)

    def test_key_to_json_masks_long_key(self):
        """Test JSON serialization masks long API keys."""
        key = Key(api_key="sk-1234567890abcdef")
        json_str = key.to_json(mask_secrets=True)
        parsed = json.loads(json_str)
        self.assertNotEqual(parsed["api_key"], "sk-1234567890abcdef")
        self.assertIn("...", parsed["api_key"])
        self.assertTrue(parsed["api_key"].startswith("sk-1"))
        self.assertTrue(parsed["api_key"].endswith("cdef"))

    def test_key_to_json_masks_short_key(self):
        """Test JSON serialization masks short API keys."""
        key = Key(api_key="short")
        json_str = key.to_json(mask_secrets=True)
        parsed = json.loads(json_str)
        self.assertIn("****", parsed["api_key"])

    def test_key_to_json_shows_missing(self):
        """Test JSON serialization shows missing for invalid keys."""
        key = Key(api_key="")
        json_str = key.to_json(mask_secrets=True)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["api_key"], "[MISSING]")

    def test_key_to_json_no_mask(self):
        """Test JSON serialization without masking."""
        key = Key(api_key="secret_key")
        json_str = key.to_json(mask_secrets=False)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["api_key"], "secret_key")


class KeyRegistryTestCase(unittest.TestCase):

    def test_empty_registry(self):
        """Test creating empty key registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = KeyRegistry(Path(tmpdir) / "key.json", {})
            self.assertEqual(len(registry), 0)

    def test_registry_with_keys(self):
        """Test creating registry with initial keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            keys = {
                "openai": {"api_key": "sk-123"},
                "anthropic": {"api_key": "sk-456"}
            }
            registry = KeyRegistry(Path(tmpdir) / "key.json", keys)
            self.assertEqual(len(registry), 2)

    def test_registry_get_key_for(self):
        """Test getting a key for a backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            keys = {"openai": {"api_key": "sk-openai"}}
            registry = KeyRegistry(Path(tmpdir) / "key.json", keys)
            key = registry.get_key_for("openai")
            self.assertEqual(key.api_key, "sk-openai")

    def test_registry_get_key_for_missing_raises(self):
        """Test that getting missing key raises AssertionError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = KeyRegistry(Path(tmpdir) / "key.json", {})
            with self.assertRaises(AssertionError):
                registry.get_key_for("nonexistent")

    def test_registry_set_key_for_new(self):
        """Test setting a new key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = KeyRegistry(Path(tmpdir) / "key.json", {})
            registry.set_key_for("openai", {"api_key": "new_key"})
            self.assertEqual(len(registry), 1)
            self.assertEqual(registry["openai"].api_key, "new_key")

    def test_registry_set_key_for_update(self):
        """Test updating an existing key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            keys = {"openai": {"api_key": "old_key"}}
            registry = KeyRegistry(Path(tmpdir) / "key.json", keys)
            registry.set_key_for("openai", {"api_key": "new_key"})
            self.assertEqual(registry["openai"].api_key, "new_key")

    def test_registry_set_key_for_reset(self):
        """Test resetting a key completely."""
        with tempfile.TemporaryDirectory() as tmpdir:
            keys = {"openai": {"api_key": "old", "extra": "field"}}
            registry = KeyRegistry(Path(tmpdir) / "key.json", keys)
            registry.set_key_for("openai", {"api_key": "new"}, reset=True)
            self.assertEqual(registry["openai"].api_key, "new")
            self.assertFalse(hasattr(registry["openai"], "extra"))

    def test_registry_persist(self):
        """Test persisting registry to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            key_file = Path(tmpdir) / "key.json"
            keys = {"openai": {"api_key": "persist_test"}}
            registry = KeyRegistry(key_file, keys)
            registry.persist()

            # Verify file was written
            self.assertTrue(key_file.exists())
            with open(key_file) as f:
                saved = json.load(f)
            self.assertEqual(saved["openai"]["api_key"], "persist_test")

    def test_registry_from_json(self):
        """Test loading registry from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            key_file = Path(tmpdir) / "key.json"
            key_data = {"openai": {"api_key": "loaded_key"}}
            with open(key_file, "w") as f:
                json.dump(key_data, f)

            # Change to temp directory to test cwd lookup
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                registry = KeyRegistry.from_json()
                self.assertEqual(registry["openai"].api_key, "loaded_key")
            finally:
                os.chdir(old_cwd)

    def test_registry_from_json_fallback_empty(self):
        """Test that loading from non-existent file returns empty registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                # No key.json exists, should return empty registry
                registry = KeyRegistry.from_json(fallback=False)
                self.assertEqual(len(registry), 0)
            finally:
                os.chdir(old_cwd)

    def test_registry_iteration(self):
        """Test iterating over registry keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            keys = {"openai": {"api_key": "a"}, "anthropic": {"api_key": "b"}}
            registry = KeyRegistry(Path(tmpdir) / "key.json", keys)
            backend_names = list(registry)
            self.assertIn("openai", backend_names)
            self.assertIn("anthropic", backend_names)

    def test_registry_contains(self):
        """Test checking if backend is in registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            keys = {"openai": {"api_key": "a"}}
            registry = KeyRegistry(Path(tmpdir) / "key.json", keys)
            self.assertIn("openai", registry)
            self.assertNotIn("anthropic", registry)


if __name__ == '__main__':
    unittest.main()
