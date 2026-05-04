# APTEW Workflow (Quick Map)

## 1) Configure run
- Select target model (`--target`) and optimizer model (`--optimizer`) from `config.py` registries.
- Provide API keys for optimizer provider (`OPENAI_API_KEY` or `DEEPSEEK_API_KEY`).
- Core success criteria are:
  - `token_count_ratio` within `[RATIO_LOWER_BOUND, RATIO_UPPER_BOUND]`
  - `token_overlap_pct > OVERLAP_THRESHOLD`

## 2) Launch training loop (`main.py`)
- `train()` loads the target runner, initializes optimizer client/prompt mode, and writes `run_config.json`.
- Dataset is loaded from `../data/openthoughts_sample.json`, batched, then iterated.
- For each sample, APTEW calls target-specific `run_extraction(...)` and logs:
  - per-sample scores into `scores.jsonl`
  - full trajectory artifacts into `artifacts.jsonl`

## 3) Inner extraction loop (`extract_*.py` + `apte.py`)
- Target-specific extractor runs stepwise injections up to `MAX_INJECTION_TURNS`.
- On each turn it computes metrics (ratio + overlap + jaccard), then asks `TrajectoryOptimizer.inject(...)` for the next perturbation.
- Optimizer is metric-driven in APTEW mode; ground-truth reasoning text is not shown to it.

## 4) Batch reflection and strategy update
- After each batch, optimizer performs `reflect_batch(...)` over all trajectories.
- `distill(...)` writes updated strategy/experience into `experience.md`.
- Training tracks success rate and can early-stop on stagnation (`EARLY_STOP_PATIENCE`, `MIN_SCORE_DELTA`).

## 5) Output layout
- Results live under `APTEW/results/<target>/`:
  - `run_config.json`
  - `scores.jsonl`
  - `artifacts.jsonl`
  - `batch_reflect.jsonl`
  - `experience.md`

## Typical command
```bash
cd APTEW
python main.py --target oss --optimizer o4-mini --prompt-mode plain
```
