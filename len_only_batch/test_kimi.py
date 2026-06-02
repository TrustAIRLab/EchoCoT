import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["MOONSHOT_API_KEY"],
    base_url="https://api.moonshot.ai/v1",
)

MODEL = "kimi-k2.6"  # replace with your actual Kimi model id

messages = [
    {
        "role": "user",
        "content": "Answer this OpenThought question: What is 17 * 23? Think carefully.",
    }
]

resp = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=1.0,
    extra_body={
        "thinking": {"type": "enabled"}
    },
)

msg = resp.choices[0].message

print("\n=== FINAL ANSWER ===")
print(msg.content)

print("\n=== EXPOSED REASONING / COT FIELD ===")

reasoning = getattr(msg, "reasoning_content", None)

if reasoning is None:
    reasoning = (getattr(msg, "model_extra", None) or {}).get("reasoning_content")

print(reasoning or "[No reasoning_content returned by API]")