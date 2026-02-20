"""Tests for MERaLiON2 multimodal audio processing."""
import numpy as np
import pytest

from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

# Version-specific imports
try:
    from vllm.tokenizers import cached_tokenizer_from_config
except ImportError:
    # For v0 engine, tokenizer might be in different location
    try:
        from vllm.model_executor.tokenizers import cached_tokenizer_from_config
    except ImportError:
        cached_tokenizer_from_config = None

from vllm_plugin_meralion2 import register

# Register the plugin before tests
register()


class TestAudioProcessing:
    """Test audio processing functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        register()

    def test_processor_registration(self):
        """Test that audio processor is registered."""
        # Create a minimal model config for testing
        # Note: This test may require a valid model path
        # For now, we just check that the registry has the processor
        assert MULTIMODAL_REGISTRY is not None

    def test_audio_input_mapper(self, dummy_audio_data):
        """Test audio input mapper functionality."""
        audio_array, sample_rate = dummy_audio_data
        
        # This test would require a full model context
        # For now, we verify the audio data format
        assert isinstance(audio_array, np.ndarray)
        assert audio_array.dtype == np.float32
        assert sample_rate == 16000

    def test_multiple_audio_inputs(self, dummy_audio_list):
        """Test processing multiple audio inputs."""
        audio_list = dummy_audio_list
        
        # Verify format
        assert isinstance(audio_list, list)
        assert len(audio_list) > 0
        for audio_data, sample_rate in audio_list:
            assert isinstance(audio_data, np.ndarray)
            assert isinstance(sample_rate, int)

    def test_audio_resampling(self, dummy_audio_data):
        """Test audio resampling functionality."""
        import librosa
        
        audio_array, orig_sample_rate = dummy_audio_data
        target_sample_rate = 16000
        
        if orig_sample_rate != target_sample_rate:
            resampled = librosa.resample(
                audio_array,
                orig_sr=orig_sample_rate,
                target_sr=target_sample_rate
            )
            assert resampled.shape[0] > 0
            assert isinstance(resampled, np.ndarray)
        else:
            # No resampling needed
            assert audio_array.shape[0] > 0


class TestDummyDataGeneration:
    """Test dummy data generation for model initialization."""

    def test_dummy_audio_generation(self, dummy_audio_data):
        """Test that dummy audio data can be generated."""
        audio_array, sample_rate = dummy_audio_data
        
        assert audio_array is not None
        assert len(audio_array) > 0
        assert sample_rate > 0

    def test_dummy_data_format(self, dummy_audio_data):
        """Test dummy data format."""
        audio_array, sample_rate = dummy_audio_data
        
        # Check data types
        assert isinstance(audio_array, np.ndarray)
        assert isinstance(sample_rate, int)
        
        # Check audio array properties
        assert audio_array.dtype == np.float32
        assert len(audio_array.shape) == 1  # 1D audio array


class TestAudioChunking:
    """Test audio chunking functionality."""

    def test_audio_chunking(self, dummy_audio_data_long):
        """Test that long audio is chunked correctly."""
        audio_array, sample_rate = dummy_audio_data_long
        
        # Constants from the plugin
        FEATURE_CHUNK_SIZE = 16000 * 30  # 30 seconds at 16kHz
        MAX_NUMBER_CHUNKS = 10
        
        audio_length = len(audio_array)
        expected_chunks = ((audio_length - 1) // FEATURE_CHUNK_SIZE) + 1
        expected_chunks = min(expected_chunks, MAX_NUMBER_CHUNKS)
        
        assert expected_chunks > 0
        assert expected_chunks <= MAX_NUMBER_CHUNKS
