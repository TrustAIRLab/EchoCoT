"""Provider-specific schemas for the shared scratchpad tool."""

SCRATCHPAD_SPEC = {
    "description": "This is a temporary workspace.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "",
            }
        },
        "required": ["content"],
        "additionalProperties": False,
    },
}


def _get_specs(names=None):
    names = names or ["scratchpad"]
    unknown = set(names) - {"scratchpad"}
    if unknown:
        raise ValueError(f"Unknown tools: {', '.join(sorted(unknown))}")
    return [(name, SCRATCHPAD_SPEC) for name in names]


def get_openai_tools(names=None):
    return [
        {
            "type": "function",
            "name": name,
            "description": spec["description"],
            "strict": True,
            "parameters": spec["parameters"],
        }
        for name, spec in _get_specs(names)
    ]


def get_chat_tools(names=None):
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": spec["description"],
                "parameters": spec["parameters"],
            },
        }
        for name, spec in _get_specs(names)
    ]


def get_anthropic_tools(names=None):
    return [
        {
            "name": name,
            "description": spec["description"],
            "input_schema": spec["parameters"],
        }
        for name, spec in _get_specs(names)
    ]


def get_gemini_tools(types_module, names=None):
    declarations = [
        types_module.FunctionDeclaration(
            name=name,
            description=spec["description"],
            parameters_json_schema=spec["parameters"],
        )
        for name, spec in _get_specs(names)
    ]
    return [types_module.Tool(function_declarations=declarations)]
