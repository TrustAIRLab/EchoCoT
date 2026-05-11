from __future__ import annotations

from pathlib import Path

import requests


def get_location():
    try:
        response = requests.get("https://ipinfo.io/json", timeout=10)
        response.raise_for_status()
        data = response.json()
        latitude = None
        longitude = None
        loc = data.get("loc")
        if loc and "," in loc:
            lat_text, lon_text = loc.split(",", 1)
            latitude = float(lat_text)
            longitude = float(lon_text)
        return {
            "city": data.get("city"),
            "region": data.get("region"),
            "country": data.get("country"),
            "latitude": latitude,
            "longitude": longitude,
        }
    except Exception as exc:
        return {"error": f"Failed to get location: {exc}"}


def get_weather(latitude, longitude):
    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,weather_code",
                "timezone": "auto",
                "forecast_days": 1,
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        current = data.get("current", {})
        daily = data.get("daily", {})
        return {
            "timezone": data.get("timezone"),
            "current_temperature_2m": current.get("temperature_2m"),
            "current_weather_code": current.get("weather_code"),
            "today_temperature_2m_max": (
                daily.get("temperature_2m_max", [None])[0]
                if daily.get("temperature_2m_max")
                else None
            ),
            "today_temperature_2m_min": (
                daily.get("temperature_2m_min", [None])[0]
                if daily.get("temperature_2m_min")
                else None
            ),
            "today_weather_code": (
                daily.get("weather_code", [None])[0]
                if daily.get("weather_code")
                else None
            ),
        }
    except Exception as exc:
        return {"error": f"Failed to get weather: {exc}"}


def scratchpad(content: str):
    return content


def verify_proof(proof: str):
    issues = []
    if len(proof) < 20:
        issues.append("Proof too short to be valid.")
    if "therefore" not in proof.lower():
        issues.append("Missing logical conclusion indicators.")
    if "because" not in proof.lower() and "since" not in proof.lower():
        issues.append("Lack of justification steps.")
    if issues:
        return {"valid": False, "issues": issues}
    return {
        "valid": True,
        "message": "Proof appears logically consistent (heuristic check).",
    }


NOTES_MEMORY = []


def notes(content: str):
    NOTES_MEMORY.append(content)
    return {
        "status": "saved",
        "notes": NOTES_MEMORY,
    }


def format_answer(answer: str):
    return {"formatted_answer": f"\\boxed{{{answer}}}"}


TOOL_SPECS = {
    "scratchpad": {
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
    },
    "verify_proof": {
        "description": "Check whether a given mathematical or geometric proof is logically valid.",
        "parameters": {
            "type": "object",
            "properties": {
                "proof": {
                    "type": "string",
                    "description": "The full proof to verify.",
                }
            },
            "required": ["proof"],
            "additionalProperties": False,
        },
    },
    "format_answer": {
        "description": "Format the final answer according to required output conventions (e.g., wrap in \\boxed{}).",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Final answer to be formatted.",
                }
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
    "notes": {
        "description": "Store important intermediate facts, equations, or observations during problem solving.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Key facts or observations to remember.",
                }
            },
            "required": ["content"],
            "additionalProperties": False,
        },
    },
    "get_location": {
        "description": "Get the user's current approximate location, including city, region, country, latitude, and longitude.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    "get_weather": {
        "description": "Get today's weather for a given latitude and longitude.",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Latitude of the target location",
                },
                "longitude": {
                    "type": "number",
                    "description": "Longitude of the target location",
                },
            },
            "required": ["latitude", "longitude"],
            "additionalProperties": False,
        },
    },
}


def call_tool(name, args):
    if name == "scratchpad":
        return scratchpad(**args)
    if name == "verify_proof":
        return verify_proof(**args)
    if name == "notes":
        return notes(**args)
    if name == "format_answer":
        return format_answer(**args)
    if name == "get_location":
        return get_location()
    if name == "get_weather":
        return get_weather(**args)
    return {"error": f"Unknown tool: {name}"}


def get_openai_tools(names=None):
    tool_names = names or list(TOOL_SPECS)
    tools = []
    for name in tool_names:
        spec = TOOL_SPECS[name]
        tools.append(
            {
                "type": "function",
                "name": name,
                "description": spec["description"],
                "strict": True,
                "parameters": spec["parameters"],
            }
        )
    return tools


def get_chat_tools(names=None):
    tool_names = names or list(TOOL_SPECS)
    tools = []
    for name in tool_names:
        spec = TOOL_SPECS[name]
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": spec["description"],
                    "parameters": spec["parameters"],
                },
            }
        )
    return tools


def get_anthropic_tools(names=None):
    tool_names = names or list(TOOL_SPECS)
    tools = []
    for name in tool_names:
        spec = TOOL_SPECS[name]
        tools.append(
            {
                "name": name,
                "description": spec["description"],
                "input_schema": spec["parameters"],
            }
        )
    return tools


def get_harmony_tools(tool_description_cls, names=None):
    tool_names = names or list(TOOL_SPECS)
    tools = []
    for name in tool_names:
        spec = TOOL_SPECS[name]
        tools.append(
            tool_description_cls.new(
                name,
                spec["description"],
                parameters=spec["parameters"],
            )
        )
    return tools


def get_gemini_tools(types_module, names=None):
    declarations = []
    for tool in get_chat_tools(names):
        function_spec = tool["function"]
        declarations.append(
            types_module.FunctionDeclaration(
                name=function_spec["name"],
                description=function_spec.get("description", ""),
                parameters_json_schema=function_spec.get("parameters", {}),
            )
        )
    return [types_module.Tool(function_declarations=declarations)]


def load_questions(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        questions = [line.rstrip("\n") for line in handle if line.strip()]
    if not questions:
        raise ValueError(f"No questions found in {path}")
    return questions


def load_instruction(path_or_pattern, turn):
    if not path_or_pattern:
        return ""
    if "{turn}" in path_or_pattern:
        path = path_or_pattern.format(turn=turn)
    else:
        path = path_or_pattern
    return Path(path).read_text(encoding="utf-8")
