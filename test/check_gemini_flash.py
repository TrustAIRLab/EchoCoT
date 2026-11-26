"""Run a live Vertex AI check with: python3 -m test.check_gemini_flash."""

import json

from config import OPTIMIZER_MODEL_OPTIONS
from optimize.__main__ import create_optimizer_client


def main():
    entry = OPTIMIZER_MODEL_OPTIONS["gemini-2.5-flash"]
    with create_optimizer_client(entry) as client:
        response = client.models.generate_content(
            model=entry["model"], contents="Reply with OK.",
        )

    output = response.text
    print(json.dumps({
        "provider": entry["provider"],
        "requested_model": entry["model"],
        "returned_model": response.model_version,
        "output": output,
        "usage": response.usage_metadata.model_dump() if response.usage_metadata else None,
    }, ensure_ascii=False, indent=2))
    if response.model_version != entry["model"] or not output:
        raise SystemExit("Check failed: unexpected model or empty output.")


if __name__ == "__main__":
    main()
