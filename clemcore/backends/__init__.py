import json
import os
from typing import Dict, List
from clemcore.backends.model_registry import (
    ModelSpec,
    ModelRegistry,
    Model,
    HumanModel,
    CustomResponseModel,
    BatchGenerativeModel
)
from clemcore.backends.backend_registry import Backend, RemoteBackend, BackendRegistry

__all_ = [
    "Model",
    "BatchGenerativeModel",
    "ModelSpec",
    "ModelRegistry",
    "HumanModel",
    "CustomResponseModel",
    "Backend",
    "RemoteBackend",
    "BackendRegistry"
]


def load_credentials(backend, file_name="key.json") -> Dict:
    """Load login credentials and API keys from JSON file.
    Args:
        backend: Name of the backend/API provider to load key for.
        file_name: Name of the key file.
    Returns:
        Dictionary with {backend: {api_key: key}}.
    """

    key_file = os.path.join(os.getcwd(), file_name)
    try:  # first, optional location at cwd
        with open(key_file) as f:
            creds = json.load(f)
    except:  # second, look into user home app dir
        key_file = os.path.join(os.path.expanduser("~"), ".clemcore", file_name)
        with open(key_file) as f:
            creds = json.load(f)
    assert backend in creds, f"No '{backend}' in {file_name}. See README."
    assert "api_key" in creds[backend], f"No 'api_key' in {file_name}. See README."
    return creds


def load_models(model_names: List[str]):
    ...
