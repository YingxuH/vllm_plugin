import base64

from openai import OpenAI

prompt_template = "Instruction: {text_input} \nFollow the text instruction based on the following audio: <SpeechHere>"

def get_client(api_key="EMPTY", base_url="http://localhost:8000/v1"):
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    models = client.models.list()
    model_name = models.data[0].id
    return client, model_name


def get_response(text_input, base64_audio_input=None, **params):
    """Single audio input helper"""
    if base64_audio_input:
        content = [
            {
                "type": "text",
                "text": prompt_template.format(text_input=text_input)
            },
            {
                "type": "audio_url",
                "audio_url": {
                    "url": f"data:audio/ogg;base64,{base64_audio_input}"
                },
            },
        ]
    else:
        content = text_input
    
    response_obj = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": content,
        }],
        **params
    )
    return response_obj


def get_multi_audio_response(text_input, base64_audio_inputs, **params):
    """Multi-round audio input helper - supports multiple audio inputs in one message"""
    if not base64_audio_inputs:
        raise ValueError("At least one audio input is required")
    
    # Build content with text and multiple audio inputs
    content = [
        {
            "type": "text",
            "text": text_input
        }
    ]
    
    # Add each audio input
    for audio_base64 in base64_audio_inputs:
        content.append({
            "type": "audio_url",
            "audio_url": {
                "url": f"data:audio/ogg;base64,{audio_base64}"
            },
        })
    
    response_obj = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": content,
        }],
        **params
    )
    return response_obj


def get_conversation_response(messages, **params):
    """Multi-turn conversation helper"""
    response_obj = client.chat.completions.create(
        messages=messages,
        **params
    )
    return response_obj


# Load audio files
# Make sure these are 16khz, mono channel audio files
audio_file_1 = "/workspace/vllm/idpc_sample.mp3"
audio_file_2 = "/workspace/vllm/idpc_sample.mp3"  # Using same file for demo, replace with different audio

try:
    audio_bytes_1 = open(audio_file_1, "rb").read()
    audio_base64_1 = base64.b64encode(audio_bytes_1).decode('utf-8')
except FileNotFoundError:
    print(f"Warning: {audio_file_1} not found. Please update the path.")
    audio_base64_1 = None

try:
    audio_bytes_2 = open(audio_file_2, "rb").read()
    audio_base64_2 = base64.b64encode(audio_bytes_2).decode('utf-8')
except FileNotFoundError:
    print(f"Warning: {audio_file_2} not found. Using audio_file_1 as fallback.")
    audio_base64_2 = audio_base64_1

# Initialize client
client, model_name = get_client(base_url="http://localhost:8063/v1")

generation_parameters = dict(
    model=model_name,
    max_completion_tokens=1024,
    temperature=0.0,
    top_p=0.9,
    extra_body={
        "repetition_penalty": 1.0,
        "top_k": 50,
        "length_penalty": 1.0,
        # "logits_processors": [
        #     {"qualname": "vllm_plugin_meralion2.NoRepeatNGramLogitsProcessor", "args": [6]}
        # ]
    },
    stream=False,
    seed=42
)

print("=" * 80)
print("Test 1: Single message with multiple audio inputs")
print("=" * 80)
if audio_base64_1 and audio_base64_2:
    # Test multiple audio inputs in a single message
    multi_audio_prompt = (
        "Instruction: Please transcribe both audio clips. "
        "First transcribe the first audio, then the second audio.\n"
        "Follow the text instruction based on the following audios: <SpeechHere><SpeechHere>"
    )
    
    response_obj = get_multi_audio_response(
        multi_audio_prompt,
        [audio_base64_1, audio_base64_2],
        **generation_parameters
    )
    print("Response:")
    print(response_obj.choices[0].message.content)
    print()
else:
    print("Skipped: Audio files not found")

print("=" * 80)
print("Test 2: Multi-turn conversation with audio")
print("=" * 80)
if audio_base64_1 and audio_base64_2:
    # First turn: User sends first audio
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_template.format(text_input="Please transcribe this first audio clip.")
                },
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": f"data:audio/ogg;base64,{audio_base64_1}"
                    },
                },
            ]
        }
    ]
    
    response_obj = get_conversation_response(messages, **generation_parameters)
    assistant_response_1 = response_obj.choices[0].message.content
    print("Turn 1 - User: Please transcribe this first audio clip.")
    print("Turn 1 - Assistant:", assistant_response_1)
    print()
    
    # Add assistant response to conversation history
    messages.append({
        "role": "assistant",
        "content": assistant_response_1
    })
    
    # Second turn: User sends second audio
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt_template.format(text_input="Now please transcribe this second audio clip.")
            },
            {
                "type": "audio_url",
                "audio_url": {
                    "url": f"data:audio/ogg;base64,{audio_base64_2}"
                },
            },
        ]
    })
    
    response_obj = get_conversation_response(messages, **generation_parameters)
    assistant_response_2 = response_obj.choices[0].message.content
    print("Turn 2 - User: Now please transcribe this second audio clip.")
    print("Turn 2 - Assistant:", assistant_response_2)
    print()
else:
    print("Skipped: Audio files not found")

print("=" * 80)
print("Test 3: Single message with 3+ audio inputs (stress test)")
print("=" * 80)
if audio_base64_1:
    # Test with 3 audio inputs
    multi_audio_prompt_3 = (
        "Instruction: Please analyze these three audio clips and summarize each one.\n"
        "Follow the text instruction based on the following audios: <SpeechHere><SpeechHere><SpeechHere>"
    )
    
    response_obj = get_multi_audio_response(
        multi_audio_prompt_3,
        [audio_base64_1, audio_base64_1, audio_base64_1],  # Using same audio 3 times for demo
        **generation_parameters
    )
    print("Response:")
    print(response_obj.choices[0].message.content)
    print()
else:
    print("Skipped: Audio files not found")

print("=" * 80)
print("All tests completed!")
print("=" * 80)
