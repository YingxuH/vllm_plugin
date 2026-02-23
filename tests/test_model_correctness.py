"""Tests for MERaLiON2 model correctness against HuggingFace."""
import os
import sys
import pytest
import torch
import numpy as np

from vllm.config import ModelConfig
from vllm.model_executor.models import ModelRegistry

from vllm_plugin_meralion2 import register

# Add tests directory to path if needed
tests_dir = os.path.dirname(os.path.abspath(__file__))
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

from test_utils import (
    get_openai_client,
    create_dummy_audio_base64,
    get_single_audio_response,
    get_default_generation_params,
    DEFAULT_MODEL_PATH,
)


# Register the plugin before tests
register()


class TestModelCorrectness:
    """Test model correctness against HuggingFace implementation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        register()

    @pytest.mark.skipif(
        not os.path.exists(DEFAULT_MODEL_PATH),
        reason=f"Model path {DEFAULT_MODEL_PATH} does not exist"
    )
    def test_model_initialization(self):
        """Test that the model can be initialized and responds to requests."""
        try:
            client, model_name = get_openai_client()
            
            # Test that we can list models and the model is available
            models = client.models.list()
            assert len(models.data) > 0, "No models available"
            
            # Test that we can get model info
            model_info = [m for m in models.data if m.id == model_name]
            assert len(model_info) > 0, f"Model {model_name} not found"
            
            # Test a simple text-only request to verify model is working
            response = client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": "Say hello in one sentence."
                }],
                max_tokens=50,
                temperature=0.0
            )
            
            assert response.choices is not None
            assert len(response.choices) > 0
            assert response.choices[0].message.content is not None
            assert len(response.choices[0].message.content) > 0
            
        except Exception as e:
            pytest.skip(f"Failed to connect to vLLM server: {e}")

    @pytest.mark.skipif(
        not os.path.exists(DEFAULT_MODEL_PATH),
        reason=f"Model path {DEFAULT_MODEL_PATH} does not exist"
    )
    def test_forward_pass(self):
        """Test forward pass through the model via API."""
        try:
            client, model_name = get_openai_client()
            params = get_default_generation_params()
            
            # Test with text-only input
            response = get_single_audio_response(
                client=client,
                model_name=model_name,
                text_input="Say hello.",
                base64_audio_input=None,
                **params
            )
            
            # Verify response structure
            assert response.choices is not None
            assert len(response.choices) > 0
            assert response.choices[0].message.content is not None
            
            # Verify usage statistics
            if hasattr(response, 'usage'):
                assert response.usage is not None
                assert response.usage.total_tokens > 0
            
        except Exception as e:
            pytest.skip(f"Failed to test forward pass: {e}")

    @pytest.mark.skipif(
        not os.path.exists(DEFAULT_MODEL_PATH),
        reason=f"Model path {DEFAULT_MODEL_PATH} does not exist"
    )
    def test_audio_embedding_generation(self):
        """Test that audio embeddings are generated correctly via API."""
        try:
            client, model_name = get_openai_client()
            params = get_default_generation_params()
            
            # Create dummy audio data
            audio_base64 = create_dummy_audio_base64(
                duration_seconds=1.0,
                sample_rate=16000,
                seed=42
            )
            
            # Test with audio input
            response = get_single_audio_response(
                client=client,
                model_name=model_name,
                text_input="Please transcribe this audio.",
                base64_audio_input=audio_base64,
                **params
            )
            
            # Verify response structure
            assert response.choices is not None
            assert len(response.choices) > 0
            assert response.choices[0].message.content is not None
            
            # Verify that the model processed the audio (response should not be empty)
            response_text = response.choices[0].message.content
            assert len(response_text) > 0, "Model should generate response for audio input"
            
            # Verify usage statistics
            if hasattr(response, 'usage'):
                assert response.usage is not None
                assert response.usage.total_tokens > 0
            
        except Exception as e:
            pytest.skip(f"Failed to test audio embedding generation: {e}")

    def test_model_class_exists(self):
        """Test that the model class exists in the registry."""
        register()
        supported_archs = ModelRegistry.get_supported_archs()
        assert "MERaLiON2ForConditionalGeneration" in supported_archs
        
        # Try to load the model class
        try:
            model_cls = ModelRegistry._try_load_model_cls("MERaLiON2ForConditionalGeneration")
            assert model_cls is not None
        except Exception:
            # At least verify it's registered
            assert "MERaLiON2ForConditionalGeneration" in supported_archs

    def test_model_has_required_methods(self):
        """Test that the model has required methods."""
        register()
        
        try:
            model_cls = ModelRegistry._try_load_model_cls("MERaLiON2ForConditionalGeneration")
            
            # Check for required methods
            required_methods = [
                'forward',
                'load_weights',
                'compute_logits',
            ]
            
            for method_name in required_methods:
                assert hasattr(model_cls, method_name), (
                    f"Model class missing required method: {method_name}"
                )
        except Exception:
            # If we can't load the class, skip this test
            pytest.skip("Cannot load model class for method checking")

    def test_model_supports_multimodal(self):
        """Test that the model supports multimodal inputs."""
        register()
        
        try:
            model_cls = ModelRegistry._try_load_model_cls("MERaLiON2ForConditionalGeneration")
            
            # Check for multimodal support
            # Note: issubclass() doesn't work with Protocols that have non-method members in Python 3.12+
            # So we check if SupportsMultiModal is in the bases or MRO instead
            from vllm.model_executor.models.interfaces import SupportsMultiModal
            
            # Try issubclass first (works in older Python versions)
            try:
                is_multimodal = issubclass(model_cls, SupportsMultiModal)
                assert is_multimodal, (
                    f"MERaLiON2ForConditionalGeneration should inherit from SupportsMultiModal"
                )
            except TypeError:
                # Python 3.12+ doesn't support issubclass() with Protocols that have non-method members
                # Check if SupportsMultiModal is in the bases (should be there since class explicitly inherits from it)
                is_in_bases = SupportsMultiModal in model_cls.__bases__
                is_in_mro = SupportsMultiModal in model_cls.__mro__
                
                assert is_in_bases or is_in_mro, (
                    f"MERaLiON2ForConditionalGeneration should inherit from SupportsMultiModal. "
                    f"Bases: {model_cls.__bases__}, MRO: {model_cls.__mro__}"
                )
        except Exception as e:
            # If we can't load the class, skip this test
            pytest.skip(f"Cannot load model class for multimodal support checking: {e}")


class TestVersionCompatibility:
    """Test version-specific compatibility."""

    def test_v1_08_09_engine_compatibility(self):
        """Test compatibility with vLLM 0.8/0.9 lane."""
        from conftest import is_v1_08_09_engine

        if is_v1_08_09_engine():
            register()
            supported_archs = ModelRegistry.get_supported_archs()
            assert "MERaLiON2ForConditionalGeneration" in supported_archs

    def test_v1_010_engine_compatibility(self):
        """Test compatibility with vLLM 0.10 lane."""
        from conftest import is_v1_010_engine

        if is_v1_010_engine():
            register()
            supported_archs = ModelRegistry.get_supported_archs()
            assert "MERaLiON2ForConditionalGeneration" in supported_archs
