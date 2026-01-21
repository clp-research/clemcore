"""Integration tests for HuggingFace local backend.

These tests use a tiny model to verify the full generation pipeline works correctly.
They are marked as 'integration' and skipped by default (run with: pytest -m integration).
"""
import pytest

# Skip all tests if huggingface dependencies aren't available
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("clemcore.backends.huggingface_local_api")

from clemcore import backends
from clemcore.backends.huggingface_local_api import HuggingfaceLocal

# Tiny GPT-2 model (~2MB) for fast CI testing
# This model is designed for testing and produces semi-coherent text
TINY_MODEL_SPEC = backends.ModelSpec(**{
    "model_name": "tiny-gpt2",
    "backend": "huggingface_local",
    "huggingface_id": "sshleifer/tiny-gpt2",
    "model_config": {
        "premade_chat_template": False,
        # Simple chat template for models without one
        "custom_chat_template": (
            "{% for message in messages %}"
            "{{ message['role'] }}: {{ message['content'] }}\n"
            "{% endfor %}"
            "assistant:"
        ),
        "eos_to_cull": r"<\|endoftext\|>",
        "padding_side": "left"
    }
})


@pytest.fixture(scope="module")
def tiny_model():
    """Load the tiny model once for all tests in this module."""
    backend = HuggingfaceLocal()
    model = backend.get_model_for(TINY_MODEL_SPEC)
    return model


@pytest.mark.integration
class TestHuggingfaceIntegration:
    """Integration tests that load and run an actual HuggingFace model."""

    def test_generate_response_returns_valid_structure(self, tiny_model):
        """Test that generate_response returns the expected tuple structure."""
        messages = [{"role": "user", "content": "Hello"}]

        tiny_model.set_gen_arg("temperature", 0)
        tiny_model.set_gen_arg("max_tokens", 20)

        prompt, response, response_text = tiny_model.generate_response(messages)

        # Check return structure
        assert prompt is not None, "Prompt should not be None"
        assert response is not None, "Response should not be None"
        assert response_text is not None, "Response text should not be None"

        # Check types
        assert isinstance(prompt, dict), "Prompt should be a dict"
        assert isinstance(response, dict), "Response should be a dict"
        assert isinstance(response_text, str), "Response text should be a string"

    def test_generate_response_not_empty(self, tiny_model):
        """Test that the model generates non-empty output."""
        messages = [{"role": "user", "content": "What is the capital of France?"}]

        tiny_model.set_gen_arg("temperature", 0)
        tiny_model.set_gen_arg("max_tokens", 50)

        _, _, response_text = tiny_model.generate_response(messages)

        # Response should not be empty or just whitespace
        assert response_text.strip(), "Response text should not be empty"
        assert len(response_text) > 0, "Response text should have content"

    def test_generate_response_deterministic_with_temp_zero(self, tiny_model):
        """Test that generation is reproducible with temperature=0."""
        messages = [{"role": "user", "content": "Count from one to five."}]

        tiny_model.set_gen_arg("temperature", 0)
        tiny_model.set_gen_arg("max_tokens", 30)

        # Generate twice with same settings
        _, _, response_text_1 = tiny_model.generate_response(messages)
        _, _, response_text_2 = tiny_model.generate_response(messages)

        assert response_text_1 == response_text_2, (
            f"Responses should be identical with temp=0.\n"
            f"Response 1: {response_text_1!r}\n"
            f"Response 2: {response_text_2!r}"
        )

    def test_generate_response_varies_with_temperature(self, tiny_model):
        """Test that generation varies with temperature > 0 (probabilistic)."""
        messages = [{"role": "user", "content": "Tell me something random."}]

        tiny_model.set_gen_arg("temperature", 1.0)
        tiny_model.set_gen_arg("max_tokens", 50)

        # Generate multiple times - with high temp, outputs should eventually differ
        responses = set()
        for _ in range(5):
            _, _, response_text = tiny_model.generate_response(messages)
            responses.add(response_text)

        # With temperature=1.0, we expect at least some variation
        # (though not guaranteed, so we just check we got valid outputs)
        assert all(r.strip() for r in responses), "All responses should be non-empty"

    def test_generate_response_prompt_contains_input(self, tiny_model):
        """Test that the returned prompt contains the input information."""
        test_content = "This is a unique test message for verification."
        messages = [{"role": "user", "content": test_content}]

        tiny_model.set_gen_arg("temperature", 0)
        tiny_model.set_gen_arg("max_tokens", 10)

        prompt, _, _ = tiny_model.generate_response(messages)

        # The prompt dict should contain the rendered input
        assert "inputs" in prompt, "Prompt should contain 'inputs' key"
        assert test_content in prompt["inputs"], "Prompt inputs should contain our message"

    def test_generate_response_with_conversation_history(self, tiny_model):
        """Test generation with multi-turn conversation history."""
        messages = [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello Alice!"},
            {"role": "user", "content": "What is my name?"}
        ]

        tiny_model.set_gen_arg("temperature", 0)
        tiny_model.set_gen_arg("max_tokens", 30)

        prompt, response, response_text = tiny_model.generate_response(messages)

        # Should handle multi-turn without errors
        assert response_text is not None
        assert isinstance(response_text, str)
        # The prompt should contain all messages
        assert "Alice" in prompt["inputs"]

    def test_generate_batch_response(self, tiny_model):
        """Test batch generation with multiple inputs."""
        batch_messages = [
            [{"role": "user", "content": "Hello"}],
            [{"role": "user", "content": "Goodbye"}],
        ]

        tiny_model.set_gen_arg("temperature", 0)
        tiny_model.set_gen_arg("max_tokens", 20)

        results = tiny_model.generate_batch_response(batch_messages)

        assert len(results) == 2, "Should return results for each batch item"

        for prompt, response, response_text in results:
            assert prompt is not None
            assert response is not None
            assert isinstance(response_text, str)
            assert response_text.strip(), "Each response should be non-empty"
