"""Test utilities for MERaLiON2 vLLM plugin tests."""
import os
import base64
import numpy as np
from typing import Optional, Tuple, List
from openai import OpenAI

# Model path from serve_test_meralion2_general.sh
DEFAULT_MODEL_PATH = "/workspace/MERaLiON_local/MERaLiON-2-10B"
DEFAULT_BASE_URL = os.getenv("VLLM_TEST_BASE_URL", "http://localhost:8063/v1")
DEFAULT_API_KEY = os.getenv("VLLM_TEST_API_KEY", "EMPTY")


def get_openai_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None
) -> Tuple[OpenAI, str]:
    """Get OpenAI client and model name.
    
    Args:
        base_url: Base URL for the vLLM server. Defaults to DEFAULT_BASE_URL.
        api_key: API key. Defaults to DEFAULT_API_KEY.
    
    Returns:
        Tuple of (client, model_name).
    
    Raises:
        ConnectionError: If unable to connect to the server.
    """
    base_url = base_url or DEFAULT_BASE_URL
    api_key = api_key or DEFAULT_API_KEY
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    
    try:
        models = client.models.list()
        if not models.data:
            raise ConnectionError(f"No models available at {base_url}")
        model_name = models.data[0].id
        return client, model_name
    except Exception as e:
        raise ConnectionError(f"Failed to connect to vLLM server at {base_url}: {e}")


def create_audio_base64(audio_array: np.ndarray, sample_rate: int = 16000) -> str:
    """Convert audio array to base64 encoded string.
    
    Args:
        audio_array: Audio data as numpy array (float32, 1D).
        sample_rate: Sample rate of the audio.
    
    Returns:
        Base64 encoded audio string.
    
    Raises:
        ImportError: If soundfile is not available.
    """
    import io
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError(
            "soundfile is required for audio encoding. "
            "Install it with: pip install soundfile"
        )
    
    # Ensure audio is float32 and 1D
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)
    if len(audio_array.shape) > 1:
        audio_array = audio_array.flatten()
    
    # Write to in-memory buffer
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format='OGG')
    buffer.seek(0)
    
    # Encode to base64
    audio_bytes = buffer.read()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    return audio_base64


def create_dummy_audio_base64(
    duration_seconds: float = 1.0,
    sample_rate: int = 16000,
    seed: Optional[int] = None
) -> str:
    """Create dummy audio and return as base64.
    
    Args:
        duration_seconds: Duration in seconds.
        sample_rate: Sample rate.
        seed: Random seed for reproducibility.
    
    Returns:
        Base64 encoded audio string.
    """
    rng = np.random.RandomState(seed)
    num_samples = int(sample_rate * duration_seconds)
    audio_array = rng.randn(num_samples).astype(np.float32)
    return create_audio_base64(audio_array, sample_rate)


def get_single_audio_response(
    client: OpenAI,
    model_name: str,
    text_input: str,
    base64_audio_input: Optional[str] = None,
    **params
):
    """Get response for single audio input.
    
    Args:
        client: OpenAI client.
        model_name: Model name.
        text_input: Text instruction.
        base64_audio_input: Base64 encoded audio (optional).
        **params: Additional generation parameters.
    
    Returns:
        Response object from chat.completions.create.
    """
    prompt_template = (
        "Instruction: {text_input} \n"
        "Follow the text instruction based on the following audio: <SpeechHere>"
    )
    
    if base64_audio_input:
        content = [
            {
                "type": "text",
                "text": prompt_template.format(text_input=text_input)
            },
            {
                "type": "audio_url",
                "audio_url": {
                    "url": f"data:audio/ogg;base64,{base64_audio_input}"
                },
            },
        ]
    else:
        content = text_input
    
    response_obj = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": content,
        }],
        **params
    )
    return response_obj


def get_multi_audio_response(
    client: OpenAI,
    model_name: str,
    text_input: str,
    base64_audio_inputs: List[str],
    **params
):
    """Get response for multiple audio inputs.
    
    Args:
        client: OpenAI client.
        model_name: Model name.
        text_input: Text instruction.
        base64_audio_inputs: List of base64 encoded audio strings.
        **params: Additional generation parameters.
    
    Returns:
        Response object from chat.completions.create.
    """
    if not base64_audio_inputs:
        raise ValueError("At least one audio input is required")
    
    # Build placeholder string for multiple audios
    placeholders = "<SpeechHere>" * len(base64_audio_inputs)
    prompt_template = (
        "Instruction: {text_input} \n"
        "Follow the text instruction based on the following audios: {placeholders}"
    )
    
    # Build content with text and multiple audio inputs
    content = [
        {
            "type": "text",
            "text": prompt_template.format(text_input=text_input, placeholders=placeholders)
        }
    ]
    
    # Add each audio input
    for audio_base64 in base64_audio_inputs:
        content.append({
            "type": "audio_url",
            "audio_url": {
                "url": f"data:audio/ogg;base64,{audio_base64}"
            },
        })
    
    response_obj = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": content,
        }],
        **params
    )
    return response_obj


def get_default_generation_params() -> dict:
    """Get default generation parameters for testing.
    
    Returns:
        Dictionary of default generation parameters.
    """
    return {
        "max_completion_tokens": 1024,
        "temperature": 0.0,
        "top_p": 0.9,
        "extra_body": {
            "repetition_penalty": 1.0,
            "top_k": 50,
            "length_penalty": 1.0,
        },
        "stream": False,
        "seed": 42
    }
