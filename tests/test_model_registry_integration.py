"""Integration tests for models in model_registry.json.

Tests that all registered models (except huggingface, vllm, openrouter, llamacpp)
can be loaded via the ModelRegistry and successfully generate responses.

Run with: pytest -m api tests/test_model_registry_integration.py -v
"""
import inspect
from pathlib import Path

import pytest

from clemcore.backends import ModelRegistry
from clemcore.backends.backend_registry import BackendRegistry

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


def get_testable_model_registry():
    package_path = get_class_package_path(ModelRegistry)
    return ModelRegistry.from_directory(package_path).where(lambda spec: spec.backend not in EXCLUDED_BACKENDS)


def get_testable_model_specs():
    return get_testable_model_registry().model_specs


def get_model_ids():
    """Generate test IDs from model names for pytest parametrization."""
    return get_testable_model_registry().select("model_name")


@pytest.fixture(scope="module")
def model_registry():
    """Create a ModelRegistry with specs from model_registry.json."""
    return get_testable_model_registry()


@pytest.fixture(scope="module")
def backend_registry():
    """Create a BackendRegistry from packaged backends."""
    package_path = get_class_package_path(BackendRegistry)
    return BackendRegistry.from_directory(package_path)


@pytest.mark.api
@pytest.mark.parametrize("model_spec", get_testable_model_specs(), ids=get_model_ids())
def test_model_generate_response(model_spec, model_registry, backend_registry):
    """Test that a model can generate a response."""
    backend = backend_registry.get_backend_for(model_spec.backend)
    model = backend.get_model_for(model_spec)
    model.set_gen_arg("temperature", 0)
    model.set_gen_arg("max_tokens", 100)

    messages = [{"role": "user", "content": "Say hello in exactly 3 words."}]
    prompt, response, response_text = model.generate_response(messages)

    assert prompt is not None, f"Prompt should not be None for {model_spec.model_name}"
    assert response is not None, f"Response should not be None for {model_spec.model_name}"
    assert response_text is not None, f"Response text should not be None for {model_spec.model_name}"
    assert isinstance(response_text, str)
    assert len(response_text) > 0, f"Response text should not be empty for {model_spec.model_name}"
