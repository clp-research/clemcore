"""Tests that all eos_to_cull entries in model_registry.json are valid regex patterns.

Backends such as huggingface_local and llama.cpp use re.sub() to strip EOS tokens
from model outputs. Special regex characters in EOS strings must be properly escaped,
otherwise re.sub() either silently produces wrong output or raises an error.
"""
import json
import re
import unittest
from pathlib import Path

REGISTRY_PATH = Path(__file__).parent.parent / "clemcore" / "backends" / "model_registry.json"


class TestModelRegistryEos(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            cls.registry = json.load(f)
        cls.entries_with_eos = [
            entry for entry in cls.registry
            if "eos_to_cull" in entry.get("model_config", {})
        ]

    def test_eos_entries_exist(self):
        """Sanity check — registry has at least some eos_to_cull entries."""
        self.assertGreater(len(self.entries_with_eos), 0)

    def test_eos_to_cull_compiles_as_regex(self):
        """Every eos_to_cull value must be a valid Python regex."""
        errors = []
        for entry in self.entries_with_eos:
            eos = entry["model_config"]["eos_to_cull"]
            try:
                re.compile(eos)
            except re.error as e:
                errors.append(f"{entry['model_name']}: invalid regex {repr(eos)} — {e}")
        self.assertFalse(errors, "Invalid eos_to_cull regex patterns found:\n" + "\n".join(errors))

    def test_eos_to_cull_no_unescaped_specials(self):
        """Warn about unescaped regex special characters that are likely mistakes.

        Characters like | ( ) [ ] { } are almost never intentional in EOS tokens
        and must be escaped with \\ in the JSON string.
        """
        # Characters that are virtually never intentional raw in an EOS token
        suspicious = set(r"[](){}|")
        errors = []
        for entry in self.entries_with_eos:
            eos = entry["model_config"]["eos_to_cull"]
            for idx, char in enumerate(eos):
                if char in suspicious:
                    prev = eos[idx - 1] if idx > 0 else ""
                    if prev != "\\":
                        errors.append(
                            f"{entry['model_name']}: unescaped '{char}' at position {idx} "
                            f"in eos_to_cull {repr(eos)}"
                        )
        self.assertFalse(errors, "Potentially unescaped regex specials in eos_to_cull:\n" + "\n".join(errors))


if __name__ == "__main__":
    unittest.main()
