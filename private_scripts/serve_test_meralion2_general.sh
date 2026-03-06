# export HF_TOKEN=<your_huggingface_token>
export CUDA_VISIBLE_DEVICES=1

# Use FlashInfer backend which supports Gemma2 attn logit softcapping natively.
# On H100 (SM 9.0) vLLM 0.12+ picks FlashAttention by default; FlashInfer
# must be requested explicitly to avoid the FA3 tanh-softcap incompatibility.
export VLLM_ATTENTION_BACKEND=FLASHINFER

model=/workspace/MERaLiON_local/MERaLiON-2-10B

vllm serve $model \
    --tokenizer $model \
    --trust-remote-code \
    --dtype bfloat16 \
    --port 8063 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 1200 \
    --max-num-seqs 16 \
    --limit-mm-per-prompt '{"audio": 2}' \
    --served-model-name MERaLiON-2-10B
