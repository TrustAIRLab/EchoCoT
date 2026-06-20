import os
from openai import OpenAI

api_key = os.getenv("DASHSCOPE_API_KEY")
assert api_key, "DASHSCOPE_API_KEY is not set"

client = OpenAI(
    api_key=api_key,
    base_url="http://ws-k46uszac7yrb36u6.eu-central-1.maas.aliyuncs.com/compatible-mode/v1",
    timeout=60,   # important: avoid hanging forever
)

MODEL = "qwen3.7-plus"   # or the exact deployed model id

resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": "Say hello in one short sentence."}
    ],
    temperature=0.0,
    max_tokens=128,
    extra_body={
        "enable_thinking": True,
        "thinking_budget": 1024,
    },
)

msg = resp.choices[0].message

print("CONTENT:")
print(msg.content)

print("\nRAW MESSAGE:")
print(msg)