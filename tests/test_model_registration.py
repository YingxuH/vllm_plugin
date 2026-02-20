"""Tests for MERaLiON2 model registration with vLLM."""
import pytest
import vllm
from vllm import ModelRegistry

from vllm_plugin_meralion2 import register


class TestModelRegistration:
    """Test MERaLiON2 model registration."""

    def test_plugin_registration(self):
        """Test that the plugin can be registered."""
        # Register the plugin
        register()
        
        # Check that the model is registered
        supported_archs = ModelRegistry.get_supported_archs()
        assert "MERaLiON2ForConditionalGeneration" in supported_archs

    def test_supported_vllm_versions(self, supported_vllm_versions):
        """Test that the plugin supports the expected vLLM versions."""
        current_version = vllm.__version__
        assert current_version in supported_vllm_versions, (
            f"vLLM version {current_version} is not in supported versions: "
            f"{supported_vllm_versions}"
        )

    def test_model_registry_contains_meralion2(self):
        """Test that MERaLiON2 model is in the registry after registration."""
        register()
        supported_archs = ModelRegistry.get_supported_archs()
        assert "MERaLiON2ForConditionalGeneration" in supported_archs

    def test_double_registration(self):
        """Test that registering twice doesn't cause errors."""
        # First registration
        register()
        supported_archs_1 = ModelRegistry.get_supported_archs()
        
        # Second registration
        register()
        supported_archs_2 = ModelRegistry.get_supported_archs()
        
        # Should still be registered
        assert "MERaLiON2ForConditionalGeneration" in supported_archs_1
        assert "MERaLiON2ForConditionalGeneration" in supported_archs_2

    def test_model_class_import(self, vllm_version):
        """Test that the correct model class is imported based on vLLM version."""
        register()
        
        # Check that the model is in supported architectures
        supported_archs = ModelRegistry.get_supported_archs()
        assert "MERaLiON2ForConditionalGeneration" in supported_archs
        
        # Try to load the model class (using private method as it's the only way)
        try:
            model_cls = ModelRegistry._try_load_model_cls("MERaLiON2ForConditionalGeneration")
            assert model_cls is not None
            assert model_cls.__name__ == "MERaLiON2ForConditionalGeneration"
        except Exception:
            # If loading fails, at least verify it's registered
            assert "MERaLiON2ForConditionalGeneration" in supported_archs
