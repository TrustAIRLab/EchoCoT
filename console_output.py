"""Small, consistent terminal messages; saved experiment records are unchanged."""

import os


def print_detail(label, content):
    """Show long diagnostic text only when explicitly requested."""
    if os.getenv("ECHOCOT_VERBOSE") == "1":
        print(f"\n[{label}]\n{content}", flush=True)


def print_turn(model, sample_id, turn_idx, tokens, reference_tokens, length_error):
    print(
        f"[{model}] sample={sample_id or '-'} turn={turn_idx} "
        f"tokens={tokens}/{reference_tokens} length_error={length_error:.3f}",
        flush=True,
    )
