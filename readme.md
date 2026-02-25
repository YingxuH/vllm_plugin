## MERaLiON2 vLLM Plugin

[![Security (Bandit)](https://github.com/YingxuH/vllm_plugin/actions/workflows/security.yml/badge.svg)](https://github.com/YingxuH/vllm_plugin/actions/workflows/security.yml)
[![Dependency Audit (pip-audit)](https://github.com/YingxuH/vllm_plugin/actions/workflows/dependency-audit.yml/badge.svg)](https://github.com/YingxuH/vllm_plugin/actions/workflows/dependency-audit.yml)
[![Dependency Review](https://github.com/YingxuH/vllm_plugin/actions/workflows/dependency-review.yml/badge.svg)](https://github.com/YingxuH/vllm_plugin/actions/workflows/dependency-review.yml)
[![CodeQL](https://github.com/YingxuH/vllm_plugin/actions/workflows/codeql.yml/badge.svg)](https://github.com/YingxuH/vllm_plugin/actions/workflows/codeql.yml)
[![Publish](https://github.com/YingxuH/vllm_plugin/actions/workflows/publish.yml/badge.svg)](https://github.com/YingxuH/vllm_plugin/actions/workflows/publish.yml)


### Licence

[MERaLiON-Public-Licence-v3](https://huggingface.co/datasets/MERaLiON/MERaLiON_Public_Licence/blob/main/MERaLiON-Public-Licence-v3.pdf)

### Set up Environment

This plugin family has two release lines:

- `v0.1.x`: compatibility lane for vLLM version `0.6.5` ~ `0.7.3` (V0 engine), and `0.8.5` ~ `0.8.5.post1` (V1 engine). 
- `v0.2.x`: compatibility lane for `vLLM >=0.8.5,<=0.10.0`. Refer to [matrix_summary.md](https://github.com/YingxuH/vllm_plugin/blob/main/matrix_summary.md) for detailed vLLM + transformers compatibility.

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

### Full release history 

See [CHANGELOG.md](https://github.com/YingxuH/vllm_plugin/blob/main/CHANGELOG.md).


### vLLM + transformers compatibility


See [matrix_summary.md](https://github.com/YingxuH/vllm_plugin/blob/main/matrix_summary.md)

### Security and dependency scanning

The repository uses separate workflows so each scan has a clear purpose:

- `Security (Bandit SAST)` (`.github/workflows/security.yml`): static security linting of project Python source (`bandit -r src`).
- `CodeQL` (`.github/workflows/codeql.yml`): semantic code scanning for Python + GitHub Actions security issues.
- `Dependency Audit (pip-audit)` (`.github/workflows/dependency-audit.yml`): installed dependency vulnerability scanning.
- `Dependency Review (PR)` (`.github/workflows/dependency-review.yml`): checks dependency changes in pull requests and fails on `moderate`+ severity vulnerabilities.
