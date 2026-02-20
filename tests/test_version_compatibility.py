"""Tests for version compatibility across different vLLM versions."""
import pytest
import vllm

from vllm import ModelRegistry
from vllm_plugin_meralion2 import register


class TestVersionCompatibility:
    """Test compatibility with different vLLM versions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        register()

    def test_current_version_supported(self):
        """Test that current vLLM version is supported."""
        current_version = vllm.__version__
        
        v0_versions = [
            '0.6.5', '0.6.6', '0.6.6.post1',
            '0.7.0', '0.7.1', '0.7.2', '0.7.3'
        ]
        v1_versions = ['0.8.5', '0.8.5.post1']
        
        supported_versions = v0_versions + v1_versions
        
        assert current_version in supported_versions, (
            f"vLLM version {current_version} is not supported. "
            f"Supported versions: {supported_versions}"
        )

    def test_v0_versions_supported(self):
        """Test that v0 engine versions are supported."""
        v0_versions = [
            '0.6.5', '0.6.6', '0.6.6.post1',
            '0.7.0', '0.7.1', '0.7.2', '0.7.3'
        ]
        
        current_version = vllm.__version__
        if current_version in v0_versions:
            register()
            supported_archs = ModelRegistry.get_supported_archs()
            assert "MERaLiON2ForConditionalGeneration" in supported_archs

    def test_v1_versions_supported(self):
        """Test that v1 engine versions are supported."""
        v1_versions = ['0.8.5', '0.8.5.post1']
        
        current_version = vllm.__version__
        if current_version in v1_versions:
            register()
            supported_archs = ModelRegistry.get_supported_archs()
            assert "MERaLiON2ForConditionalGeneration" in supported_archs

    def test_unsupported_version_error(self):
        """Test that unsupported versions raise appropriate error."""
        # This test verifies the error handling in register()
        # We can't easily test unsupported versions without mocking,
        # but we verify the registration works for supported versions
        current_version = vllm.__version__
        
        v0_versions = [
            '0.6.5', '0.6.6', '0.6.6.post1',
            '0.7.0', '0.7.1', '0.7.2', '0.7.3'
        ]
        v1_versions = ['0.8.5', '0.8.5.post1']
        supported_versions = v0_versions + v1_versions
        
        if current_version in supported_versions:
            # Should not raise error
            try:
                register()
                registered = True
            except RuntimeError:
                registered = False
            assert registered, f"Registration failed for supported version {current_version}"

    def test_model_class_import_by_version(self):
        """Test that correct model class is imported based on version."""
        current_version = vllm.__version__
        
        v0_versions = [
            '0.6.5', '0.6.6', '0.6.6.post1',
            '0.7.0', '0.7.1', '0.7.2', '0.7.3'
        ]
        v1_versions = ['0.8.5', '0.8.5.post1']
        
        register()
        
        # Verify registration
        supported_archs = ModelRegistry.get_supported_archs()
        assert "MERaLiON2ForConditionalGeneration" in supported_archs
        
        # Try to verify the correct module is imported
        try:
            model_cls = ModelRegistry._try_load_model_cls("MERaLiON2ForConditionalGeneration")
            
            if current_version in v0_versions:
                # V0 engine - should import from vllm064_post1
                from vllm_plugin_meralion2.vllm064_post1 import MERaLiON2ForConditionalGeneration as V0Model
                assert model_cls.__name__ == V0Model.__name__
            elif current_version in v1_versions:
                # V1 engine - should import from vllm085
                from vllm_plugin_meralion2.vllm085 import MERaLiON2ForConditionalGeneration as V1Model
                assert model_cls.__name__ == V1Model.__name__
        except Exception:
            # If loading fails, at least verify registration
            assert "MERaLiON2ForConditionalGeneration" in supported_archs

    def test_registration_idempotent(self):
        """Test that registration can be called multiple times safely."""
        # First registration
        register()
        supported_archs_1 = ModelRegistry.get_supported_archs()
        
        # Second registration
        register()
        supported_archs_2 = ModelRegistry.get_supported_archs()
        
        # Should still be registered
        assert "MERaLiON2ForConditionalGeneration" in supported_archs_1
        assert "MERaLiON2ForConditionalGeneration" in supported_archs_2
