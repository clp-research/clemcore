from collections.abc import Mapping
import json
import logging
from pathlib import Path

module_logger = logging.getLogger(__name__)


class Key(Mapping):

    def __init__(self, api_key: str = None, **kwargs):
        self.api_key = api_key  # the secret
        self.__dict__.update(kwargs)

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def has_api_key(self):
        return isinstance(self.api_key, str) and bool(self.api_key.strip())

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True)

    def __repr__(self):
        return f"Key(api_key={self.api_key!r}, extra={list(self.keys())})"


class KeyRegistry(Mapping):

    def __init__(self, key_file_path: Path, keys: Mapping = None):
        self._key_file_path = key_file_path
        self._keys = {}
        if keys:
            self._keys = {backend_name: Key(**entry) for backend_name, entry in keys.items()}

    @property
    def key_file_path(self):
        return self._key_file_path

    def __getitem__(self, item):
        return self._keys[item]

    def __len__(self):
        return len(self._keys)

    def __iter__(self):
        return iter(self._keys)

    def get_key_for(self, backend_name: str) -> Key:
        assert backend_name in self, f"No '{backend_name}' in {self._key_file_path}. See README."
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
        key_file_path = Path.cwd() / file_name
        try:
            with open(key_file_path) as f:
                return cls(key_file_path, json.load(f))
        except Exception as e:
            module_logger.info(f"Loading key file from {key_file_path} (cwd) failed: %s.", e)
        try:
            key_file_path = Path.home() / ".clemcore" / file_name
            with open(key_file_path) as f:
                return cls(key_file_path, json.load(f))
        except Exception as e:
            module_logger.info(f"Loading key file from {key_file_path} (home) failed: %s", e)
        module_logger.warning(f"Fallback to empty key.json at {key_file_path}.")
        return cls(key_file_path, {})
