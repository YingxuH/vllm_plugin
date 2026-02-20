import os
import torch
import librosa

from vllm import LLM, SamplingParams


model_name = "MERaLiON/MERaLiON-2-10B"
# model_name = "MERaLiON/MERaLiON-2-10B-ASR"
# model_name = "MERaLiON/MERaLiON-2-3B"

llm = LLM(
    model=model_name,
    tokenizer=model_name,
    limit_mm_per_prompt={"audio": 1},
    trust_remote_code=True,
    dtype=torch.bfloat16
)

# change example.wav to your audio file.
audio_array, sample_rate = librosa.load("example.wav", sr=16000)

question= "Please trancribe this speech."
prompt = (
    "<start_of_turn>user\n"
    f"Instruction: {question} \nFollow the text instruction based on the following audio: <SpeechHere><end_of_turn>\n"
    "<start_of_turn>model\n")

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.0,
    seed=42,
    max_tokens=1024,
    stop_token_ids=None,
    # vLLM V1 does not support per-request logits_processors.
    # If you need NoRepeatNGramLogitsProcessor, run with V0 engine
    # (e.g. set VLLM_USE_V1=0 before launching).
    # logits_processors=[NoRepeatNGramLogitsProcessor(6)],
)

mm_data = {"audio": [(audio_array, sample_rate)]}
inputs = {"prompt": prompt, "multi_modal_data": mm_data}

# batch inference
inputs = [inputs] * 2

outputs = llm.generate(inputs, sampling_params=sampling_params)

for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)
