#!/usr/bin/env bash
set -euo pipefail

# target url and port
TARGET_API=http://localhost:8000

# MERaLiON/MERaLiON-2-10B or MERaLiON/MERaLiON-2-10B-ARS or MERaLiON/MERaLiON-2-3B
MODEL_NAME=MERaLiON/MERaLiON-2-10B  

# Refer to https://huggingface.co/MERaLiON/MERaLiON-2-10B-ASR#audio-input and https://huggingface.co/MERaLiON/MERaLiON-2-10B#audio-input
PROMPT='Instruction: Please transcribe this speech. \nFollow the text instruction based on the following audio: <SpeechHere>'

# change to true if need stream output
STREAM=false

# change example.wav to your audio file.
base64 -w 0 -i example.wav > audio.b64


cat > payload.json <<EOF
{
  "model": "${MODEL_NAME}",
  "messages":[
    {"role":"user",
     "content":[
       {"type":"text","text":"${PROMPT}"},
       {"type":"audio_url","audio_url":{"url":"data:audio/ogg;base64,$(cat audio.b64)"}}
     ]
    }
  ],
  "max_completion_tokens":1024,
  "temperature":0.1,
  "top_p":0.9,
  "top_k":50,
  "repetition_penalty":1.0,
  "length_penalty":1.0,
  "logits_processors":[{"qualname":"vllm_plugin_meralion2.NoRepeatNGramLogitsProcessor","args":[6]}],
  "seed":42,
  "stream":${STREAM}
}
EOF

curl $TARGET_API/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data-binary @payload.json
