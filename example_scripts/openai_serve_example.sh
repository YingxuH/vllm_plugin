#!/usr/bin/env bash
set -euo pipefail

# MERaLiON-2 requires FlashInfer attention backend for Gemma2's attn logit
# softcapping (tanh softcap not supported by FlashAttention-3).
#
# vLLM 0.12.x reads VLLM_ATTENTION_BACKEND env var only (no CLI flag).
# vLLM >= 0.13.0 uses --attention-backend CLI flag; env var must NOT also be
# set (both together cause "mutually exclusive" ValueError in 0.13.0+).
ATTN_FLAG=$(python3 -c "
import vllm
from packaging.version import Version
print('--attention-backend FLASHINFER' if Version(vllm.__version__) >= Version('0.13.0') else '')
" 2>/dev/null || echo "")
if [ -z "$ATTN_FLAG" ]; then
    export VLLM_ATTENTION_BACKEND=FLASHINFER
else
    unset VLLM_ATTENTION_BACKEND
fi

# The NoRepeatNGram logits processor is auto-registered by the plugin's
# entry-point (no --logits-processor-pattern needed).

vllm serve MERaLiON/MERaLiON-2-10B \
    --trust-remote-code \
    --dtype bfloat16 \
    --port 8000 \
    $ATTN_FLAG