# import os
# from openai import OpenAI

# api_key = os.getenv("DASHSCOPE_API_KEY")
# assert api_key, "DASHSCOPE_API_KEY is not set"

# client = OpenAI(
#     api_key=api_key,
#     base_url="http://ws-k46uszac7yrb36u6.eu-central-1.maas.aliyuncs.com/compatible-mode/v1",
#     timeout=60,   # important: avoid hanging forever
# )

# MODEL = "qwen3.7-plus"   # or the exact deployed model id

# resp = client.chat.completions.create(
#     model=MODEL,
#     messages=[
#         {"role": "user", "content": "Say hello in one short sentence."}
#     ],
#     temperature=0.0,
#     max_tokens=128,
#     extra_body={
#         "enable_thinking": True,
#         "thinking_budget": 1024,
#     },
# )

# msg = resp.choices[0].message

# print("CONTENT:")
# print(msg.content)

# print("\nRAW MESSAGE:")
# print(msg)

# from google import genai
# from google.genai import types

# client = genai.Client(
#     vertexai=True,
#     project="gen-lang-client-0688043640",
#     location="global",
# )

# response = client.models.generate_content(
#     model="gemini-2.5-flash",
#     contents="Say hello in one sentence.",
#     config=types.GenerateContentConfig(
#         thinking_config=types.ThinkingConfig(thinking_level="low")
#     ),
# )

# print(response)

import os
from openai import OpenAI

api_key = os.getenv("MINIMAX_API_KEY")
assert api_key, "MINIMAX_API_KEY is not set"

client = OpenAI(
    api_key=api_key,
    base_url="https://api.minimaxi.com/v1",
    timeout=60,
)

response = client.chat.completions.create(
    model="MiniMax-M3",
    messages=[
        {"role": "user", "content": "Solve: If x + 2 = 5, what is x?"}
    ],
    temperature=0.0,
    max_tokens=2048,
    extra_body={
        "thinking": {"type": "adaptive"},
        "reasoning_split": True,
    },
)

msg = response.choices[0].message

print("=== RAW MESSAGE ===")
print(msg)

print("\n=== Reasoning ===")
reasoning = getattr(msg, "reasoning_content", None)
if reasoning:
    print(reasoning)
elif getattr(msg, "reasoning_details", None):
    for item in msg.reasoning_details:
        if isinstance(item, dict):
            print(item.get("text", item))
        else:
            print(item)
else:
    print("(No separated reasoning returned)")

print("\n=== Answer ===")
print(msg.content)