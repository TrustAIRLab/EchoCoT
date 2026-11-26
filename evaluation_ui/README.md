# Interactive injection demo

From the repository root, in the Python environment used for the model APIs:

```bash
bash scripts/evaluate_ui.sh
```

Open **http://127.0.0.1:8765**. For another port:

```bash
python3 -B -m evaluation_ui --port 8766
```

Choose a **target model** and **dataset**. Questions follow the order in
`evaluation_ui/examples/<target>/<dataset>/test.json`, without shuffling. **Next sample** loads the next
question; there is no sample-ID selector or custom-question input. Only the
question is taken from the dataset.

The bundled examples are selected separately for each target from CoTRecon:
JEEBench, LiveCodeBench, MATH500, and OpenThoughts. Each combination has 10
successful examples, except Gemini-3.5/MATH500, which has only 4 qualifying
records. There are 314 examples across 8 targets.

DeepSeek, Qwen, and GLM use `results/evaluation/<target>/length_lexical/<dataset>/samples.jsonl`.
The best scratchpad must pass ASR@90: length error ≤ 0.1 and Token EM ≥ 0.9,
using the unrounded token ratio for the length boundary and the initial CoT as
the Token EM reference. Gemini, Opus, and Sonnet use the provider-tokenized
records in `results/analysis/provider_tokenizers/<target>_<dataset>.json`,
requiring best-scratchpad length error ≤ 0.1. Those records use `length_only`
for Opus and `length_lexical` for the other closed-source targets. CoTRecon's
`gemini-3.1` corresponds to this repository's `gemini-3.1-flash-lite`.

The first qualifying unique samples are copied in source order, keeping the
latest record for any repeated sample ID. Question text and IDs are unchanged.
No saved reasoning, answers, or metrics are copied; all responses are generated
live, so a new run may have a different outcome. Full training/evaluation
datasets under `data/` are not used by the demo.

1. **Send question** starts a live model session with the existing archive suffix.
2. The **green ground-truth CoT panel** shows the first response's reference
   reasoning. The initial scratchpad is displayed beside it.
3. Edit a turn's injection and click **Send Turn …**. Its exact sent text and the
   resulting scratchpad appear in order. Each scratchpad includes its token count,
   ground-truth token count, length error, and Token EM.
4. Send the final instruction to finish the conversation. It defaults to the
   repository's existing acceptance instruction and is editable for this demo.
5. Click **Next sample**, then **Send question**, to start the next conversation.
   Next sample is disabled at the end of the dataset.

The demo excludes `o4-mini` and `deepseek-v4-flash-20261017`; other configured
targets remain selectable. Saved injections are loaded from
`injections/<target>_expereince.md`. Models without a saved file have empty editable
fields; no injection text is invented. Sonnet's saved file contains one injection.
Edits stay in the browser tab's session storage, including across page refreshes
and sample changes. **Restore saved text** resets drafts. Sent turns are read-only;
restart the conversation to try different text for an already-sent turn.

Models and datasets can change between requests. Changing either starts from
the first question for the selected model and dataset. **Restart
conversation** repeats the current question. During a request the controls are
temporarily disabled to prevent duplicate calls or switching the target
underneath an active session.

The backend calls `load_target`, `make_session`, `start`, `send`, and `final_send`
directly. It keeps the same session object and its reasoning/tool history across
clicks. It does not import the evaluation runner, read evaluation result files,
write evaluation records, or calculate cross-sample statistics. The original
provider routes, tokenizers, prompts, and tool handling are unchanged.

Gemini/Claude do not expose full ground-truth CoT: the green panel explicitly
labels the available reasoning summary instead. Token EM is **N/A** for these
targets; their scratchpad token counts still come from the existing provider
counting code. No summary is presented as a full CoT.

Server-sent events update request status and each completed response. Individual
model tokens are not streamed. Refreshing reconnects to the active conversation
while the server is running. Browser tabs share one active conversation; server
restarts clear it. API keys remain on the backend. No extra web dependencies are
needed beyond the repository's existing Python environment.

Verification (no model API calls):

```bash
python3 -B -m unittest -b test.test_evaluation_ui
```
