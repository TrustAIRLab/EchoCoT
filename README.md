# EchoCoT

This is the implementation repo of [EchoCoT: Extracting Hidden Chain-of-Thought from Large Reasoning Models](https://arxiv.org/abs/2608.20055), where we use the scratchpad tool to extract target model's intenral CoT near-verbatim.

## Environment configuration

Run commands from the repository root with Python 3.12.

```bash
python3 -m pip install -r requirements.txt
```

Set the credentials needed for your selected target and optimizer:

```bash
export OPENROUTER_API_KEY="your-key"  # DeepSeek target through OpenRouter
export GLM_API_KEY="your-key"         # GLM target
export DASHSCOPE_API_KEY="your-key"   # Qwen target
export QWEN_BASE_URL="your-openai-compatible-endpoint"  # Qwen endpoint
export GOOGLE_CLOUD_PROJECT="your-project-id"  # Vertex AI project
export ANTHROPIC_API_KEY="your-key"   # Claude targets
export OPENAI_API_KEY="your-key"     # CoT summaries / OpenAI models
```

All Gemini targets and optimizers use Vertex AI with Google Cloud Application Default Credentials. To configure them:

```bash
gcloud auth application-default login
```

Model options are in [config.py](config.py).

The training example below requires OpenRouter and OpenAI API keys, as well as Vertex AI credentials.
Note that since DeepSeek-V4-Flash has been updated and the 20260423 version is no longer accessible through the original DeepSeek endpoint, we use OpenRouter to access this version.

## Interactive demo

```bash
bash scripts/evaluate_ui.sh
```

Open <http://127.0.0.1:8765>. Choose a model and dataset, send the question,
then click to send each injection. The page shows the live responses and
per-scratchpad metrics. **Next sample** moves to the next example.

```mermaid
flowchart LR
    B("Question sent") --> C("Sent Turn-1") --> D("Sent Turn-2") --> E("Sent Turn-3")

    classDef button fill:#2563EB,color:#FFFFFF,stroke:none
    class A,B,C,D,E button
```

![Interactive demo screenshot 1](evaluation_ui/assets/demo-1.png)

![Interactive demo screenshot 2](evaluation_ui/assets/demo-2.png)

## Evaluation

Place the original test split at `data/<dataset>/test.json` (not included).
Dataset files are JSON arrays with `sample_id` and `question` fields.

```bash
python3 -m evaluate \
  --target-model deepseek-v4-flash \
  --dataset OpenThoughts \
  --experience_md injections/deepseek-v4-flash_expereince.md \
  --mode length_lexical \
  --max-samples 10 \
  --output-dir outputs/evaluation
```

Saved injections run in order, followed by the fixed final instruction.
Results are saved under `outputs/evaluation/<target>/<mode>/<dataset>/`.
Use `--mode length_only` for the supplied Claude Opus 4.6 injections.

## Training

Place the original training split at `data/<dataset>/train.json` (not included).
This example uses Gemini as the optimizer and OpenAI for CoT summaries.

```bash
python3 -m optimize \
  --target-model deepseek-v4-flash \
  --optimizer-model gemini-2.5-flash \
  --dataset OpenThoughts \
  --mode length_lexical
```

The default optimizes two injections; the third turn uses the fixed final
instruction. The selected model and mode determine the prompt files in
`config.py`. Learned injections and training records are saved under
`outputs/train/<target>/<mode>/<dataset>/`.

For detailed optimizer diagnostics, prefix the training command with
`ECHOCOT_VERBOSE=1`.
