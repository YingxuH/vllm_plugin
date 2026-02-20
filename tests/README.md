# MERaLiON2 vLLM Plugin Tests

This directory contains comprehensive tests for the MERaLiON2 vLLM plugin, following the official vLLM repository testing standards.

## Test Structure

The test suite is organized into several test files:

- **`conftest.py`**: Shared fixtures for audio data generation and vLLM version detection
- **`test_model_registration.py`**: Tests for plugin registration and model registry integration
- **`test_multimodal_processing.py`**: Tests for audio processing functionality
- **`test_processing_correctness.py`**: Tests for processing correctness following vLLM patterns
- **`test_model_correctness.py`**: Tests for model correctness (requires model weights)
- **`test_asr_transcription_eval.py`**: Integration tests for served ASR transcription quality

## Running Tests

### Prerequisites

1. Install the plugin:
```bash
pip install -e .
```

2. Install test dependencies:
```bash
pip install pytest pytest-cov librosa numpy
```

### Basic Test Execution

Run all tests:
```bash
pytest tests/
```

Run specific test file:
```bash
pytest tests/test_model_registration.py
```

Run specific test:
```bash
pytest tests/test_model_registration.py::TestModelRegistration::test_plugin_registration
```

### Version-Specific Testing

The plugin supports multiple vLLM versions:
- **V0 Engine**: vLLM 0.6.5 ~ 0.7.3
- **V1 Engine**: vLLM 0.8.5 ~ 0.8.5.post1

To test with a specific vLLM version:

```bash
# Install specific vLLM version
pip install vllm==0.6.5  # or 0.7.3, 0.8.5, 0.8.5.post1

# Run tests
pytest tests/
```

### Test Markers

Tests are marked with pytest markers for selective execution:

- `@pytest.mark.slow`: Slow-running tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.requires_gpu`: Tests requiring GPU
- `@pytest.mark.requires_model`: Tests requiring model weights

Run tests excluding slow ones:
```bash
pytest tests/ -m "not slow"
```

Run only unit tests:
```bash
pytest tests/ -m unit
```

Run ASR transcription evaluation tests:
```bash
pytest tests/test_asr_transcription_eval.py -m integration
```

Useful environment variables for ASR evaluation:
```bash
export VLLM_OPENAI_BASE_URL="http://localhost:8000/v1"
export VLLM_OPENAI_API_KEY="EMPTY"
export VLLM_OPENAI_MODEL="<optional-model-id>"
export ASR_TEST_DATA_ROOT="/home/yingxu/private_data"
export ASR_TEST_DATASETS="idpc_short_ASR_v2,ste_test3,ytb_asr_batch1"
export ASR_TEST_MAX_SAMPLES=8
```

## Test Coverage

### Model Registration Tests
- Plugin registration
- Model registry integration
- Version compatibility
- Double registration handling

### Multimodal Processing Tests
- Audio input processing
- Multiple audio inputs
- Audio resampling
- Dummy data generation
- Audio chunking

### ASR Transcription Evaluation Tests
- Served AudioLLM transcription via OpenAI-compatible vLLM endpoint
- Local Hugging Face datasets loaded from `/home/yingxu/private_data` by default
- Dataset-level WER assertion (`wer < 1.0`) for each configured dataset
- Prompt fixed to `Please transcribe this speech.`

### Processing Correctness Tests
- Single audio processing
- Multiple audio processing
- Audio data format validation
- Chunking logic
- Cache functionality

### Model Correctness Tests
- Model initialization
- Forward pass
- Audio embedding generation
- Required methods presence
- Multimodal support

## Skipped Tests

Some tests are marked with `@pytest.mark.skip` because they require:
- Actual model weights from HuggingFace
- GPU availability
- Full model loading

To run these tests, you need:
1. A valid MERaLiON2 model path
2. GPU access
3. Sufficient memory

## Continuous Integration

For CI/CD pipelines, use:

```bash
# Run tests with coverage
pytest tests/ --cov=vllm_plugin_meralion2 --cov-report=html

# Run tests excluding slow/integration tests
pytest tests/ -m "not slow and not integration"
```

## Troubleshooting

### Import Errors
If you encounter import errors, ensure:
1. The plugin is installed: `pip install -e .`
2. vLLM is installed: `pip install vllm==<version>`
3. All dependencies are installed: `pip install -r requirements.txt`

### Version Mismatch
If tests fail due to version mismatch:
1. Check vLLM version: `python -c "import vllm; print(vllm.__version__)"`
2. Ensure version is in supported range (0.6.5-0.7.3 or 0.8.5-0.8.5.post1)
3. Install correct version if needed

### GPU Requirements
Some tests require GPU. To skip GPU-required tests:
```bash
pytest tests/ -m "not requires_gpu"
```

## Contributing

When adding new tests:
1. Follow the existing test structure
2. Use appropriate pytest markers
3. Add docstrings explaining what is being tested
4. Ensure tests work across supported vLLM versions
5. Update this README if adding new test categories
