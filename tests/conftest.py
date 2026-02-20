"""Shared fixtures for MERaLiON2 vLLM plugin tests."""
import numpy as np
import pytest
from typing import Generator

# Audio test fixtures
@pytest.fixture
def dummy_audio_data() -> tuple[np.ndarray, int]:
    """Generate dummy audio data for testing.
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    # Generate 1 second of audio at 16kHz
    sample_rate = 16000
    duration = 1.0
    num_samples = int(sample_rate * duration)
    audio_array = np.random.randn(num_samples).astype(np.float32)
    return audio_array, sample_rate


@pytest.fixture
def dummy_audio_data_long() -> tuple[np.ndarray, int]:
    """Generate longer dummy audio data for testing (30 seconds).
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    sample_rate = 16000
    duration = 30.0
    num_samples = int(sample_rate * duration)
    audio_array = np.random.randn(num_samples).astype(np.float32)
    return audio_array, sample_rate


@pytest.fixture
def dummy_audio_list() -> list[tuple[np.ndarray, int]]:
    """Generate a list of dummy audio data for testing.
    
    Returns:
        List of (audio_array, sample_rate) tuples
    """
    return [
        (np.random.randn(16000).astype(np.float32), 16000),
        (np.random.randn(8000).astype(np.float32), 16000),
    ]


def get_vllm_version() -> str:
    """Get the current vLLM version."""
    import vllm
    return vllm.__version__


def is_v0_engine() -> bool:
    """Check if using v0 engine (0.6.5-0.7.3)."""
    version = get_vllm_version()
    v0_versions = ['0.6.5', '0.6.6', '0.6.6.post1', '0.7.0', '0.7.1', '0.7.2', '0.7.3']
    return version in v0_versions


def is_v1_engine() -> bool:
    """Check if using v1 engine (0.8.5-0.8.5.post1)."""
    version = get_vllm_version()
    v1_versions = ['0.8.5', '0.8.5.post1']
    return version in v1_versions


@pytest.fixture
def vllm_version() -> str:
    """Get the current vLLM version."""
    return get_vllm_version()


@pytest.fixture
def supported_vllm_versions() -> list[str]:
    """Get list of supported vLLM versions."""
    return [
        '0.6.5', '0.6.6', '0.6.6.post1',
        '0.7.0', '0.7.1', '0.7.2', '0.7.3',
        '0.8.5', '0.8.5.post1'
    ]
