# export HF_TOKEN=<your_huggingface_token>
export CUDA_VISIBLE_DEVICES=1
# Use FlashInfer backend which supports Gemma2 attn logit softcapping natively.
# On H100 (SM 9.0) vLLM 0.12+ picks FlashAttention by default; FlashInfer
# must be requested explicitly to avoid the FA3 tanh-softcap incompatibility.
export VLLM_ATTENTION_BACKEND=FLASHINFER
# export VLLM_LOGITS_PROCESSOR_PLUGINS="vllm_plugin_meralion2.NoRepeatNGramLogitsProcessor"
# export VLLM_LOGGING_LEVEL=DEBUG
# # export VLLM_TRACE_FUNCTION=1
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG=TRACE
# export TRANSFORMERS_VERBOSITY=debug


remote_repo=/workspace/MERaLiON_local/MERaLiON-2-10B
# remote_repo=Qwen/Qwen2-Audio-7B-Instruct
local_model_path=/home/yingxu/workspace/MERaLiON_local/v1/MERaLiON-AudioLLM-Whisper-SEA-LION-wo-lora
local_lora_path=/home/yingxu/workspace/MERaLiON_local/v1/adapter
# local_full_model_path=/home/yingxu/workspace/MERaLiON_local/v2/MERaLiON-AudioLLM-v2-merged
# local_full_model_path=/home/yingxu/workspace/MERaLiON_local/v2/MERaLiON-AudioLLM-v2-1405
local_full_model_path=/home/yingxu/workspace/MERaLiON_local/v2/MERaLiON-AudioLLM-v2-1405-asr

vllm serve $remote_repo \
    --tokenizer $remote_repo \
    --trust-remote-code --dtype bfloat16 --port 8063 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 1200 \
    --max_num_seqs 16 \
    --limit-mm-per-prompt '{"audio": 2}' \
    --served-model-name MERaLiON-2-10B


# vllm serve $local_full_model_path \
#     --tokenizer $local_full_model_path \
#     --limit-mm-per-prompt audio=5 \
#     --max-num-seqs 16 --trust-remote-code --dtype bfloat16 --port 8002 \
#     --gpu-memory-utilization 0.95 \
#     --served-model-name MERaLiON-AudioLLM-v2-1405 \
#     --logits-processor-pattern vllm_plugin_meralion.NoRepeatNGramLogitsProcessor

    # --served-model-name MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION-it \
    # --enable-lora \
    # --lora-modules MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION=$local_lora_path