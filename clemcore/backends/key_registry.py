import json
import logging
from pathlib import Path

module_logger = logging.getLogger(__name__)


class KeyRegistry:

    def __init__(self, key_file: Path, keys: dict):
        self._key_file = key_file
        self._keys = keys or {}

    def __contains__(self, item):
        return item in self._keys

    def __len__(self):
        return len(self._keys)

    def __iter__(self):
        return iter(self._keys)

    def get_key_for(self, backend_name: str) -> dict:
        assert backend_name in self, f"No '{backend_name}' in {self._key_file}. See README."
        return self._keys[backend_name]

    @classmethod
    def from_json(cls, file_name: str = "key.json") -> "KeyRegistry":
        """
        Look up key.json in the following locations:
        (1) Lookup in the current working directory (relative to script execution)
        (2) Lookup in the users home .clemcore folder
        If keys are found in the (1) then (2) is ignored.
        Returns: key registry backed by the json file
        """
        key_file = Path.cwd() / file_name
        try:
            with open(key_file) as f:
                return cls(key_file, json.load(f))
        except Exception as e:
            module_logger.info(f"Loading key file from {key_file} (cwd) failed: %s.", e)
        try:
            key_file = Path.home() / ".clemcore" / file_name
            with open(key_file) as f:
                return cls(key_file, json.load(f))
        except Exception as e:
            module_logger.info(f"Loading key file from {key_file} (home) failed: %s", e)
        module_logger.warning(f"Fallback to empty key.json at {key_file}.")
        return cls(key_file, {})
