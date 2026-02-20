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


# change example.wav to your audio file. Make sure its 16khz, mono channel.
audio_bytes = open("example.wav", "rb").read()
audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

# use the port number of your vllm service.
client, model_name = get_client(base_url="http://localhost:8000/v1")

generation_parameters = dict(
    model=model_name,
    max_completion_tokens=1024,
    temperature=0.0,
    top_p=0.9,
    extra_body={
        "repetition_penalty": 1.0,
        "top_k": 50,
        "length_penalty": 1.0,
        "logits_processors": [
            {"qualname": "vllm_plugin_meralion2.NoRepeatNGramLogitsProcessor", "args": [6]}
        ]
    },
    seed=42
)

response_obj = get_response("Please transcribe this speech.", audio_base64, **generation_parameters)
print(response_obj.choices[0].message.content)