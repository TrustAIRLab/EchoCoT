# Manual extraction evaluation

Evaluate manually supplied JSON injection trajectories on the same scratchpad
extraction sessions used by `len_only_batch` for DeepSeek, OSS, and Gemini.

## Injection trajectory input

Pass one JSON or JSONL file with `turn_idx` and `injection` records:

```json
[
  {"turn_idx": 1, "injection": "First injection text."},
  {"turn_idx": 2, "injection": "Second injection text."}
]
```

JSONL is also accepted:

```jsonl
{"turn_idx": 1, "injection": "First injection text."}
{"turn_idx": 2, "injection": "Second injection text."}
```

## Dataset and outputs

By default, the runner samples 10 examples per source from OpenThoughts, shuffles
that subset with `--seed`, and then applies `--max-samples` if provided.

Each run writes:

* `run_config.json` with the minimal run settings and injections.
* `samples.jsonl` with final answer, all scratchpads, per-turn metrics, and the
  scratchpad whose `token_count_ratio` is closest to 1.0.
* `summary.json` with aggregate statistics over those selected scratchpads.

## Examples

```bash
python evaluation/evaluate_manual_trajectory.py \
  --target-model deepseek \
  --trajectory-json trajectory.json
```

```bash
python evaluation/evaluate_manual_trajectory.py \
  --target-model all \
  --trajectory-json trajectory.json \
  --max-samples 30
```
