## MERaLiON2 vLLM Plugin

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

Refer to [offline_example.py](https://huggingface.co/MERaLiON/MERaLiON-2-10B/blob/main/vllm_plugin_meralion2/offline_example.py) for offline inference example.

### OpenAI-compatible Serving

Refer to [openai_serve_example.sh](https://huggingface.co/MERaLiON/MERaLiON-2-10B/blob/main/vllm_plugin_meralion2/openai_serve_example.sh) for openAI-compatible serving example.

To call the server, you can refer to [openai_client_example.py](https://huggingface.co/MERaLiON/MERaLiON-2-10B/blob/main/vllm_plugin_meralion2/openai_client_example.py).

Alternatively, you can try calling the server with curl, refer to [openai_client_curl.sh](https://huggingface.co/MERaLiON/MERaLiON-2-10B/blob/main/vllm_plugin_meralion2/openai_client_curl.sh).

### Changelog

#### 0.1.4

- Fixed multi-audio handling for a single request.
- Fixed server-side internal failure when multiple requests with different audio chunk counts are batched together.
- Added more docstrings for better code readability and maintenance.

Full history: see [CHANGELOG.md](./CHANGELOG.md).
