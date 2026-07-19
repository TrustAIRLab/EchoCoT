# # import os
# # from openai import OpenAI

# # api_key = os.getenv("DASHSCOPE_API_KEY")
# # assert api_key, "DASHSCOPE_API_KEY is not set"

# # client = OpenAI(
# #     api_key=api_key,
# #     base_url="http://ws-k46uszac7yrb36u6.eu-central-1.maas.aliyuncs.com/compatible-mode/v1",
# #     timeout=60,   # important: avoid hanging forever
# # )

# # MODEL = "qwen3.7-plus"   # or the exact deployed model id

# # resp = client.chat.completions.create(
# #     model=MODEL,
# #     messages=[
# #         {"role": "user", "content": "Say hello in one short sentence."}
# #     ],
# #     temperature=0.0,
# #     max_tokens=128,
# #     extra_body={
# #         "enable_thinking": True,
# #         "thinking_budget": 1024,
# #     },
# # )

# # msg = resp.choices[0].message

# # print("CONTENT:")
# # print(msg.content)

# # print("\nRAW MESSAGE:")
# # print(msg)

# # from google import genai
# # from google.genai import types

# # client = genai.Client(
# #     vertexai=True,
# #     project="gen-lang-client-0688043640",
# #     location="global",
# # )

# # response = client.models.generate_content(
# #     model="gemini-2.5-flash",
# #     contents="Say hello in one sentence.",
# #     config=types.GenerateContentConfig(
# #         thinking_config=types.ThinkingConfig(thinking_level="low")
# #     ),
# # )

# # print(response)

# import os
# from openai import OpenAI

# api_key = os.getenv("MINIMAX_API_KEY")
# assert api_key, "MINIMAX_API_KEY is not set"

# client = OpenAI(
#     api_key=api_key,
#     base_url="https://api.minimaxi.com/v1",
#     timeout=60,
# )

# response = client.chat.completions.create(
#     model="MiniMax-M3",
#     messages=[
#         {"role": "user", "content": "Solve: If x + 2 = 5, what is x?"}
#     ],
#     temperature=0.0,
#     max_tokens=2048,
#     extra_body={
#         "thinking": {"type": "adaptive"},
#         "reasoning_split": True,
#     },
# )

# msg = response.choices[0].message

# print("=== RAW MESSAGE ===")
# print(msg)

# print("\n=== Reasoning ===")
# reasoning = getattr(msg, "reasoning_content", None)
# if reasoning:
#     print(reasoning)
# elif getattr(msg, "reasoning_details", None):
#     for item in msg.reasoning_details:
#         if isinstance(item, dict):
#             print(item.get("text", item))
#         else:
#             print(item)
# else:
#     print("(No separated reasoning returned)")

# print("\n=== Answer ===")
# print(msg.content)

# test_claude_vertex.py

# from anthropic import AnthropicVertex

# PROJECT_ID = "gen-lang-client-0688043640"
# REGION = "global"
# MODEL_ID = "claude-opus-4-7"

# client = AnthropicVertex(
#     project_id=PROJECT_ID,
#     region=REGION,
# )

# try:
#     response = client.messages.create(
#         model=MODEL_ID,
#         max_tokens=100,
#         messages=[
#             {
#                 "role": "user",
#                 "content": "What is 2 + 2? Reply with only the answer.",
#             }
#         ],
#     )

#     print("SUCCESS: The project and model work.")
#     print("Model:", response.model)
#     print("Response:", response.content[0].text)

# except Exception as error:
#     print("FAILED:")
#     print(type(error).__name__)
#     print(error)

# import os
# from anthropic import Anthropic

# client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# question = """
# A researcher has 12 visually identical coins. Exactly one coin is counterfeit,
# and it may be either heavier or lighter than the genuine coins.

# Using a balance scale exactly three times, describe a strategy that always
# identifies the counterfeit coin and determines whether it is heavier or lighter.

# Explain the reasoning carefully.
# """

# response = client.messages.create(
#     model="claude-opus-4-6",
#     max_tokens=4096,
#     thinking={
#         "type": "adaptive",
#         "display": "summarized",
#     },
#     output_config={
#         "effort": "medium",
#     },
#     messages=[
#         {
#             "role": "user",
#             "content": question,
#         }
#     ],
# )

# print("Model:", response.model)
# print("Stop reason:", response.stop_reason)
# print("Usage:", response.usage)

# print("\n===== ALL CONTENT BLOCKS =====")

# for index, block in enumerate(response.content):
#     print(f"\nBlock {index}: {block.type}")

#     if block.type == "thinking":
#         print("Reasoning summary:")
#         print(block.thinking)
#         print("\nHas signature:", bool(getattr(block, "signature", None)))

#     elif block.type == "redacted_thinking":
#         print("Redacted thinking block received.")

#     elif block.type == "text":
#         print("Final answer:")
#         print(block.text)

#     else:
#         print(block.model_dump())
# import json

# import google.auth
# from google.auth.transport.requests import Request
# from openai import OpenAI


# PROJECT_ID = "gen-lang-client-0688043640"
# MODEL_ID = "zai-org/glm-5-maas"

# BASE_URL = (
#     f"https://aiplatform.googleapis.com/v1/projects/{PROJECT_ID}"
#     "/locations/global/endpoints/openapi"
# )

# # Authenticate with Google ADC
# credentials, _ = google.auth.default(
#     scopes=["https://www.googleapis.com/auth/cloud-platform"]
# )
# credentials.refresh(Request())

# client = OpenAI(
#     base_url=BASE_URL,
#     api_key=credentials.token,
# )


# # The actual local tool
# def multiply(a: int, b: int) -> dict:
#     print(f"Executing tool: multiply(a={a}, b={b})")
#     return {
#         "a": a,
#         "b": b,
#         "result": a * b,
#     }


# # Tool schema shown to the model
# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "multiply",
#             "description": "Multiply two integers and return the result.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "a": {
#                         "type": "integer",
#                         "description": "The first integer.",
#                     },
#                     "b": {
#                         "type": "integer",
#                         "description": "The second integer.",
#                     },
#                 },
#                 "required": ["a", "b"],
#                 "additionalProperties": False,
#             },
#         },
#     }
# ]

# messages = [
#     {
#         "role": "user",
#         "content": (
#             "What is 17 multiplied by 23? "
#             "You must use the multiply tool to calculate it."
#         ),
#     }
# ]

# # First model call: ask the model to select a tool
# response = client.chat.completions.create(
#     model=MODEL_ID,
#     messages=messages,
#     tools=tools,
#     tool_choice="auto",
#     temperature=0,
#     max_tokens=200,
# )

# assistant_message = response.choices[0].message

# print("First response:")
# print(assistant_message)

# if not assistant_message.tool_calls:
#     raise RuntimeError(
#         f"The model did not call a tool. Model response: "
#         f"{assistant_message.content}"
#     )

# # Preserve the assistant tool-call message
# messages.append(
#     assistant_message.model_dump(exclude_none=True)
# )

# # Execute every requested tool call
# for tool_call in assistant_message.tool_calls:
#     function_name = tool_call.function.name
#     arguments = json.loads(tool_call.function.arguments)

#     print("\nTool requested:")
#     print("name:", function_name)
#     print("arguments:", arguments)

#     if function_name == "multiply":
#         tool_result = multiply(**arguments)
#     else:
#         tool_result = {
#             "error": f"Unknown tool: {function_name}"
#         }

#     messages.append(
#         {
#             "role": "tool",
#             "tool_call_id": tool_call.id,
#             "content": json.dumps(tool_result),
#         }
#     )

# # Second model call: provide the tool result
# final_response = client.chat.completions.create(
#     model=MODEL_ID,
#     messages=messages,
#     tools=tools,
#     tool_choice="none",
#     temperature=0,
#     max_tokens=200,
# )

# print("\nFinal answer:")
# print(final_response.choices[0].message.content)


import os
import json
import time
from difflib import SequenceMatcher

import requests


api_key = (
    os.getenv("GLM_API_KEY")
    or os.getenv("ZAI_API_KEY")
    or os.getenv("ZHIPUAI_API_KEY")
    or os.getenv("BIGMODEL_API_KEY")
)

ENDPOINTS = {
    "third_party": {
        "base_url": "https://ai.shimiaocheng.top/v1",
        "api_key": os.environ["PROXY_GLM_API_KEY"],
        "model": "GLM-5.2",
    },
    "official_zai": {
        "base_url": "https://api.z.ai/api/paas/v4",
        "api_key": api_key,
        "model": "glm-5.2",
    },
}

QUESTION = """
Solve the following problem carefully.

A sequence is defined by a_1 = 2 and
a_{n+1} = a_n^2 - 2.

Compute a_5.

Return exactly two lines:
Answer: <integer>
Explanation: <one concise sentence>
""".strip()

N_RUNS = 3


def safe_headers(headers):
    """Keep diagnostic headers but exclude cookies and authentication data."""
    useful_names = {
        "server",
        "date",
        "via",
        "cf-ray",
        "x-request-id",
        "request-id",
        "x-zhipu-request-id",
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-tokens",
    }

    return {
        key: value
        for key, value in headers.items()
        if key.lower() in useful_names
    }


def list_models(config):
    url = config["base_url"].rstrip("/") + "/models"

    try:
        response = requests.get(
            url,
            headers={
                "Authorization": f"Bearer {config['api_key']}",
            },
            timeout=30,
        )

        try:
            body = response.json()
        except ValueError:
            body = response.text[:1000]

        return {
            "status_code": response.status_code,
            "body": body,
            "headers": safe_headers(response.headers),
        }

    except Exception as exc:
        return {"error": repr(exc)}


def call_model(name, config, run_index):
    url = config["base_url"].rstrip("/") + "/chat/completions"

    payload = {
        "model": config["model"],
        "messages": [
            {
                "role": "user",
                "content": QUESTION,
            }
        ],
        "temperature": 0,
        "max_tokens": 1000,
        "stream": False,
    }

    start = time.perf_counter()

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=180,
    )

    elapsed = time.perf_counter() - start

    try:
        body = response.json()
    except ValueError:
        body = {"raw_text": response.text}

    result = {
        "endpoint": name,
        "requested_model": config["model"],
        "run": run_index,
        "status_code": response.status_code,
        "latency_seconds": round(elapsed, 3),
        "headers": safe_headers(response.headers),
        "raw_response": body,
    }

    if response.ok and isinstance(body, dict):
        choices = body.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}

        result.update(
            {
                "returned_model": body.get("model"),
                "response_id": body.get("id"),
                "content": message.get("content"),
                "reasoning_content": message.get("reasoning_content"),
                "tool_calls": message.get("tool_calls"),
                "finish_reason": (
                    choices[0].get("finish_reason") if choices else None
                ),
                "usage": body.get("usage"),
                "system_fingerprint": body.get("system_fingerprint"),
            }
        )

    return result


def similarity(a, b):
    if not a or not b:
        return None
    return round(SequenceMatcher(None, a, b).ratio(), 4)


def main():
    report = {
        "question": QUESTION,
        "model_lists": {},
        "runs": [],
        "comparisons": [],
    }

    # Check advertised model lists
    for name, config in ENDPOINTS.items():
        report["model_lists"][name] = list_models(config)

    # Call both endpoints with identical parameters
    results = {}

    for run_index in range(1, N_RUNS + 1):
        results[run_index] = {}

        for name, config in ENDPOINTS.items():
            print(f"Calling {name}, run {run_index}...")

            result = call_model(name, config, run_index)
            results[run_index][name] = result
            report["runs"].append(result)

            print("Status:", result["status_code"])
            print("Returned model:", result.get("returned_model"))
            print("Usage:", result.get("usage"))
            print("Answer:", result.get("content"))
            print("-" * 70)

        proxy = results[run_index]["third_party"]
        official = results[run_index]["official_zai"]

        comparison = {
            "run": run_index,
            "both_successful": (
                proxy["status_code"] == 200
                and official["status_code"] == 200
            ),
            "returned_model_equal": (
                proxy.get("returned_model")
                == official.get("returned_model")
            ),
            "content_exactly_equal": (
                proxy.get("content") == official.get("content")
            ),
            "content_similarity": similarity(
                proxy.get("content"),
                official.get("content"),
            ),
            "reasoning_exactly_equal": (
                proxy.get("reasoning_content")
                == official.get("reasoning_content")
            ),
            "usage_equal": proxy.get("usage") == official.get("usage"),
        }

        report["comparisons"].append(comparison)

    with open("glm_endpoint_comparison.json", "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("\nComparison summary:")
    print(json.dumps(report["comparisons"], ensure_ascii=False, indent=2))
    print("\nFull report saved to glm_endpoint_comparison.json")


if __name__ == "__main__":
    main()