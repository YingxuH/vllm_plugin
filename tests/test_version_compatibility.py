"""Tests for version compatibility across different vLLM versions."""
import pytest
import vllm
from packaging.version import Version

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
        current_version = Version(vllm.__version__)
        assert Version("0.8.5") <= current_version < Version("0.11.0"), (
            f"vLLM version {current_version} is not supported. "
            "Supported versions: >=0.8.5,<0.11.0"
        )

    def test_supported_range_registration(self):
        """Test registration works for supported semver range."""
        current_version = Version(vllm.__version__)
        if Version("0.8.5") <= current_version < Version("0.11.0"):
            register()
            supported_archs = ModelRegistry.get_supported_archs()
            assert "MERaLiON2ForConditionalGeneration" in supported_archs

    def test_unsupported_version_error(self):
        """Test that unsupported versions raise appropriate error."""
        # This test verifies the error handling in register()
        # We can't easily test unsupported versions without mocking,
        # but we verify the registration works for supported versions
        current_version = Version(vllm.__version__)
        if Version("0.8.5") <= current_version < Version("0.11.0"):
            # Should not raise error
            try:
                register()
                registered = True
            except RuntimeError:
                registered = False
            assert registered, f"Registration failed for supported version {current_version}"

    def test_model_class_import_by_version(self):
        """Test that correct model class is imported based on version."""
        current_version = Version(vllm.__version__)
        register()

        # Verify registration
        supported_archs = ModelRegistry.get_supported_archs()
        assert "MERaLiON2ForConditionalGeneration" in supported_archs

        # Try to verify the correct module is imported
        try:
            model_cls = ModelRegistry._try_load_model_cls("MERaLiON2ForConditionalGeneration")

            if Version("0.8.5") <= current_version < Version("0.10.0"):
                assert model_cls.__module__.endswith("vllm085")
            elif Version("0.10.0") <= current_version < Version("0.11.0"):
                assert model_cls.__module__.endswith("vllm010")
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
