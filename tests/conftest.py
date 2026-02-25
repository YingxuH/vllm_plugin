"""Shared fixtures for MERaLiON2 vLLM plugin tests."""
import numpy as np
import pytest
from packaging.version import Version

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
    """Legacy helper retained for backwards compatibility in tests."""
    return False


def is_v1_engine() -> bool:
    """Check if using supported V1 range for this plugin."""
    version = Version(get_vllm_version())
    return Version("0.8.5") <= version < Version("0.11.0")


def is_v1_08_09_engine() -> bool:
    """Check if using vLLM 0.8/0.9 compatibility lane."""
    version = Version(get_vllm_version())
    return Version("0.8.5") <= version < Version("0.10.0")


def is_v1_010_engine() -> bool:
    """Check if using vLLM 0.10 compatibility lane."""
    version = Version(get_vllm_version())
    return Version("0.10.0") <= version < Version("0.11.0")


@pytest.fixture
def vllm_version() -> str:
    """Get the current vLLM version."""
    return get_vllm_version()


@pytest.fixture
def supported_vllm_versions() -> list[str]:
    """Get supported vLLM range represented as a list entry."""
    return [">=0.8.5,<0.11.0"]
