vllm serve MERaLiON/MERaLiON-2-10B \
    --trust-remote-code \
    --dtype bfloat16 \
    --logits-processor-pattern vllm_plugin_meralion2.NoRepeatNGramLogitsProcessor \
    --port 8000