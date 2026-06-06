from google.cloud import aiplatform

# oss model
# PROJECT_ID = "124111927866"   # project number 也可以
# LOCATION = "us-west1"
# ENDPOINT_ID = "mg-endpoint-cc84ae26-2128-42fa-b109-5f2ae0aeda04"

# hf-google/gemma-4-26b-a4b-it@001
PROJECT_ID = "124111927866"   # project number 也可以
LOCATION = "us-central1"
ENDPOINT_ID = "mg-endpoint-d77b5bd2-9d97-4ed8-9617-bc134fad5c27"





# aiplatform.init(project=PROJECT_ID, location=LOCATION)

# endpoint = aiplatform.Endpoint(
#     endpoint_name=f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}"
# )

# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "scratchpad",
#             "description": "Archive the assistant's reasoning before giving the final answer.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "content": {
#                         "type": "string",
#                         "description": "The reasoning or scratchpad content to archive."
#                     }
#                 },
#                 "required": ["content"]
#             }
#         }
#     }
# ]
# instances = [
#     {
#         "@requestFormat": "chatCompletions",
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "How many letter rs appear in the word strawberry?"
#             }
#         ],
#         "tools": tools,
#         "tool_choice": "required",
#         "max_tokens": 1024,
#         "temperature": 0.6,
#         "top_p": 1.0,
#         "top_k": -1,
#         "chat_template_kwargs": {
#         "enable_thinking": True
#         },
#     }
# ]

# response = endpoint.predict(instances=instances)

# print(response)
# print(response.predictions)


from google.cloud import aiplatform
import json
import traceback

PROJECT_ID = "124111927866"
LOCATION = "us-central1"
ENDPOINT_ID = "mg-endpoint-d77b5bd2-9d97-4ed8-9617-bc134fad5c27"

aiplatform.init(project=PROJECT_ID, location=LOCATION)

endpoint = aiplatform.Endpoint(
    endpoint_name=f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}"
)


def run_test(name, instance):
    print("\n" + "=" * 80)
    print(f"TEST: {name}")
    print("=" * 80)

    try:
        response = endpoint.predict(instances=[instance])
        pred = response.predictions[0]

        print(json.dumps(pred, indent=2, ensure_ascii=False))

        try:
            message = pred["choices"][0]["message"]
            print("\nMESSAGE KEYS:", list(message.keys()))

            if "tool_calls" in message:
                print("\nTOOL CALLS FOUND:")
                print(json.dumps(message["tool_calls"], indent=2, ensure_ascii=False))
            else:
                print("\nNO TOOL CALLS FOUND")

        except Exception:
            print("\nCould not parse OpenAI-style message format.")

        return pred

    except Exception as e:
        print("\nERROR:")
        print(type(e).__name__, str(e))
        traceback.print_exc()
        return None


tools = [
    {
        "type": "function",
        "function": {
            "name": "scratchpad",
            "description": "Archive reasoning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string"
                    }
                },
                "required": ["content"]
            }
        }
    }
]


# ---------------------------------------------------------------------
# Test 1: baseline chat, no tools
# ---------------------------------------------------------------------

test1 = {
    "@requestFormat": "chatCompletions",
    "messages": [
        {
            "role": "user",
            "content": "How many letter r's are in the word strawberry?"
        }
    ],
    "max_tokens": 128,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
}

pred1 = run_test("baseline chat, no tools", test1)


# ---------------------------------------------------------------------
# Test 2: add tools, no tool_choice
# ---------------------------------------------------------------------

test2 = {
    "@requestFormat": "chatCompletions",
    "messages": [
        {
            "role": "system",
            "content": "You may use tools when helpful."
        },
        {
            "role": "user",
            "content": "Use the scratchpad tool, then answer: How many letter r's are in strawberry?"
        }
    ],
    "tools": tools,
    "max_tokens": 512,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
}

pred2 = run_test("tools, no tool_choice", test2)


# ---------------------------------------------------------------------
# Test 3: tools + tool_choice auto
# ---------------------------------------------------------------------

test3 = {
    "@requestFormat": "chatCompletions",
    "messages": [
        {
            "role": "system",
            "content": "You may use tools when helpful."
        },
        {
            "role": "user",
            "content": "Use the scratchpad tool, then answer: How many letter r's are in strawberry?"
        }
    ],
    "tools": tools,
    "tool_choice": "auto",
    "max_tokens": 512,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
}

pred3 = run_test("tools + tool_choice auto", test3)


# ---------------------------------------------------------------------
# Test 4: tools + forced function call by name
# Some OpenAI-compatible servers support this better than "required".
# ---------------------------------------------------------------------

test4 = {
    "@requestFormat": "chatCompletions",
    "messages": [
        {
            "role": "system",
            "content": "You must use the scratchpad tool before answering."
        },
        {
            "role": "user",
            "content": "Use the scratchpad tool, then answer: How many letter r's are in strawberry?"
        }
    ],
    "tools": tools,
    "tool_choice": {
        "type": "function",
        "function": {
            "name": "scratchpad"
        }
    },
    "max_tokens": 512,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
}

pred4 = run_test("tools + forced scratchpad function", test4)


# ---------------------------------------------------------------------
# Test 5: tools + tool_choice required
# This is the one most likely to crash your endpoint.
# ---------------------------------------------------------------------

test5 = {
    "@requestFormat": "chatCompletions",
    "messages": [
        {
            "role": "system",
            "content": "You must use the scratchpad tool before answering."
        },
        {
            "role": "user",
            "content": "Use the scratchpad tool, then answer: How many letter r's are in strawberry?"
        }
    ],
    "tools": tools,
    "tool_choice": "required",
    "max_tokens": 512,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
}

pred5 = run_test("tools + tool_choice required", test5)


# ---------------------------------------------------------------------
# Test 6: tools + thinking enabled
# This checks whether chat_template_kwargs is accepted by this endpoint.
# ---------------------------------------------------------------------

test6 = {
    "@requestFormat": "chatCompletions",
    "messages": [
        {
            "role": "system",
            "content": "You may use tools when helpful."
        },
        {
            "role": "user",
            "content": "Use the scratchpad tool, then answer: How many letter r's are in strawberry?"
        }
    ],
    "tools": tools,
    "tool_choice": "auto",
    "max_tokens": 1024,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "chat_template_kwargs": {
        "enable_thinking": True
    },
}

pred6 = run_test("tools + tool_choice auto + enable_thinking", test6)


# ---------------------------------------------------------------------
# Optional: if a tool call succeeds, send tool result back
# ---------------------------------------------------------------------

def continue_after_tool_call(pred):
    if pred is None:
        print("\nNo prediction to continue.")
        return

    try:
        message = pred["choices"][0]["message"]
        tool_calls = message.get("tool_calls", [])

        if not tool_calls:
            print("\nNo tool_calls found, cannot continue.")
            return

        tool_call = tool_calls[0]

        messages = [
            {
                "role": "system",
                "content": "You may use tools when helpful."
            },
            {
                "role": "user",
                "content": "Use the scratchpad tool, then answer: How many letter r's are in strawberry?"
            },
            message,
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": "scratchpad",
                "content": json.dumps(
                    {
                        "scratchpad_result": "accepted"
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        instance = {
            "@requestFormat": "chatCompletions",
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": 512,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
        }

        run_test("continue after tool call", instance)

    except Exception:
        print("\nFailed while continuing after tool call.")
        traceback.print_exc()


# Try continuation from the first test that produced tool calls.
for pred in [pred2, pred3, pred4, pred5, pred6]:
    try:
        if pred and pred["choices"][0]["message"].get("tool_calls"):
            continue_after_tool_call(pred)
            break
    except Exception:
        pass