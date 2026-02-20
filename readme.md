## MERaLiON2 vLLM Plugin

[![Security](https://github.com/YingxuH/vllm_plugin/actions/workflows/security.yml/badge.svg)](https://github.com/YingxuH/vllm_plugin/actions/workflows/security.yml)
[![Publish](https://github.com/YingxuH/vllm_plugin/actions/workflows/publish.yml/badge.svg)](https://github.com/YingxuH/vllm_plugin/actions/workflows/publish.yml)
[![CodeQL](https://github.com/YingxuH/vllm_plugin/actions/workflows/codeql.yml/badge.svg)](https://github.com/YingxuH/vllm_plugin/actions/workflows/codeql.yml)


### Licence

[MERaLiON-Public-Licence-v3](https://huggingface.co/datasets/MERaLiON/MERaLiON_Public_Licence/blob/main/MERaLiON-Public-Licence-v3.pdf)

### Set up Environment

This vLLM plugin supports vLLM version `0.6.5` ~ `0.7.3` (V0 engine), and `0.8.5` ~ `0.8.5.post1` (V1 engine). 

Install the MERaLiON2 vLLM plugin.

```bash
pip install vllm-plugin-meralion2
```

It's strongly recommended to install flash-attn for better memory and gpu utilization. 

```bash
pip install flash-attn --no-build-isolation
```

### Offline Inference

Refer to [offline_example.py](https://github.com/YingxuH/vllm_plugin/blob/main/example_scripts/offline_example.py) for offline inference example.

### OpenAI-compatible Serving

Refer to [openai_serve_example.sh](https://github.com/YingxuH/vllm_plugin/blob/main/example_scripts/openai_serve_example.sh) for openAI-compatible serving example.

To call the server, you can refer to [openai_client_example.py](https://github.com/YingxuH/vllm_plugin/blob/main/example_scripts/openai_client_example.py).

Alternatively, you can try calling the server with curl, refer to [openai_client_curl.sh](https://github.com/YingxuH/vllm_plugin/blob/main/example_scripts/openai_client_curl.sh).

### Changelog

#### 0.1.4

- Fixed multi-audio handling for a single request.
- Fixed server-side internal failure when multiple requests with different audio chunk counts are batched together.
- Added more docstrings for better code readability and maintenance.

Full history: see [CHANGELOG.md](https://github.com/YingxuH/vllm_plugin/blob/main/CHANGELOG.md).
