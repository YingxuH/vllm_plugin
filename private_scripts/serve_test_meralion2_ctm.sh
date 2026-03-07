# export HF_TOKEN=<your_huggingface_token>
export CUDA_VISIBLE_DEVICES=1

# Use FlashInfer backend which supports Gemma2 attn logit softcapping natively.
# On H100 (SM 9.0) vLLM defaults to FlashAttention; FlashInfer must be
# requested explicitly to avoid FA3's missing tanh-softcap support.
#
# vLLM 0.12.x reads VLLM_ATTENTION_BACKEND (env var only, no CLI flag).
# vLLM >= 0.13.0 uses --attention-backend CLI flag (env var silently ignored
# from 0.14.0 onwards).  Set both so all supported versions are covered.
export VLLM_ATTENTION_BACKEND=FLASHINFER
ATTN_FLAG=$(python3 -c "
import vllm
from packaging.version import Version
print('--attention-backend FLASHINFER' if Version(vllm.__version__) >= Version('0.13.0') else '')
" 2>/dev/null || echo "")

model=/workspace/MERaLiON_local/MERaLiON-CTM-1512

vllm serve $model \
    --tokenizer $model \
    --trust-remote-code \
    --dtype bfloat16 \
    --port 8063 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096 \
    --max-num-seqs 16 \
    --limit-mm-per-prompt '{"audio": 2}' \
    --served-model-name MERaLiON-2-10B \
    $ATTN_FLAG
