## MERaLiON2 vLLM Plugin

[![Security](https://github.com/YingxuH/vllm_plugin/actions/workflows/security.yml/badge.svg)](https://github.com/YingxuH/vllm_plugin/actions/workflows/security.yml)
[![Publish](https://github.com/YingxuH/vllm_plugin/actions/workflows/publish.yml/badge.svg)](https://github.com/YingxuH/vllm_plugin/actions/workflows/publish.yml)
[![CodeQL](https://github.com/YingxuH/vllm_plugin/actions/workflows/codeql.yml/badge.svg)](https://github.com/YingxuH/vllm_plugin/actions/workflows/codeql.yml)


### Licence

[MERaLiON-Public-Licence-v3](https://huggingface.co/datasets/MERaLiON/MERaLiON_Public_Licence/blob/main/MERaLiON-Public-Licence-v3.pdf)

### Set up Environment

This plugin family has two release lines:

- `v0.1.x`: compatibility lane for vLLM version `0.6.5` ~ `0.7.3` (V0 engine), and `0.8.5` ~ `0.8.5.post1` (V1 engine). 
- `v0.2.x`: compatibility lane for `vLLM >=0.8.5,<=0.10.0`.

Install by your vLLM version:

```bash
# For vLLM 0.6.5~0.7.3, 0.8.5.
pip install "vllm-plugin-meralion2<0.2"

# For vLLM 0.8.5 ~ 0.10.0
pip install "vllm-plugin-meralion2>=0.2,<0.3"
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

Full history: see [CHANGELOG.md](https://github.com/YingxuH/vllm_plugin/blob/main/CHANGELOG.md).
