import unittest
import os
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

try:
    import tiktoken  # noqa: F401
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("tiktoken is not installed") from exc

from extractors import FINAL_ANSWER_INJECTION
from config import OPTIMIZATION_MODES, OPTIMIZER_MODEL_OPTIONS, TARGET_MODEL_OPTIONS, get_prompt_module
from metrics import InjectionTurn, TrajectoryResult
from optimize import __main__ as optimize_main


class FakeSession:
    def __init__(self, _question, _extractor=None):
        self.done = False
        self.trajectory = TrajectoryResult(ground_truth_tokens=10)
        self.received = []
        self.calls = []

    def start(self):
        self.trajectory.turns.append(self._turn(0))

    def send(self, injection):
        self.calls.append(("send", injection))
        self.received.append(injection)
        self.trajectory.turns[-1].injection_out = injection
        self.trajectory.turns.append(self._turn(len(self.trajectory.turns)))

    def final_send(self, injection):
        self.calls.append(("final_send", injection))
        self.received.append(injection)
        self.trajectory.turns[-1].injection_out = injection
        self.done = True

    @staticmethod
    def _turn(turn_idx):
        return InjectionTurn(
            turn_idx=turn_idx,
            cot=f"reaction-{turn_idx}",
            scratchpad_in="archive",
            scratchpad_in_tokens=2,
            token_count_ratio=0.2,
            length_error=0.8,
        )


class FakeOptimizer:
    def __init__(self):
        self.turns = []
        self.previews = []

    def inject_batch(self, turn_idx, *_args, **_kwargs):
        self.turns.append(turn_idx)
        self.previews.append(_kwargs["cot_preview"])
        return f"injection-{turn_idx}"


class OptimizeFlowTests(unittest.TestCase):
    def test_gemini_optimizer_uses_vertex_without_an_api_key(self):
        for model in ("gemini-2.5-flash",):
            with self.subTest(model=model), patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "test-project"}, clear=True), \
                 patch.object(optimize_main.genai, "Client") as vertex_client:
                options = runpy.run_path(str(Path(__file__).resolve().parents[1] / "config.py"))["OPTIMIZER_MODEL_OPTIONS"]
                client = optimize_main.create_optimizer_client(options[model])
                vertex_client.assert_called_once_with(
                    vertexai=True, project="test-project", location="global",
                )
                self.assertIs(client, vertex_client.return_value)

    def test_vertex_preserves_message_text_and_history(self):
        messages = [
            {"role": "system", "content": "system text\n{unchanged}"},
            {"role": "user", "content": "first input"},
            {"role": "assistant", "content": "first response"},
            {"role": "user", "content": "next input\n摘要"},
        ]
        client = MagicMock()
        client.models.generate_content.return_value = SimpleNamespace(text=" response ")
        with patch.object(optimize_main.opt_module, "client", client), \
             patch.object(optimize_main.opt_module, "api_type", "vertex"), \
             patch.object(optimize_main.opt_module, "model_id", "gemini-2.5-flash"):
            self.assertEqual(optimize_main.opt_module._call_api(messages), "response")
        request = client.models.generate_content.call_args.kwargs
        self.assertEqual(request["model"], "gemini-2.5-flash")
        self.assertEqual(request["config"].system_instruction, messages[0]["content"])
        self.assertEqual([item.role for item in request["contents"]], ["user", "model", "user"])
        self.assertEqual([item.parts[0].text for item in request["contents"]], [item["content"] for item in messages[1:]])

    @patch("optimize.__main__.summarize_cot", return_value="summary only")
    @patch("optimize.__main__.make_session", side_effect=FakeSession)
    def test_only_first_two_turns_are_optimized(self, _make_session, _summarize):
        optimizer = FakeOptimizer()
        samples = [SimpleNamespace(question="question", sample_id="sample")]

        trajectories, realized, _, _, cot_preview = optimize_main.run_batch(
            optimizer,
            samples,
            extractor="fake",
            max_turns=3,
            mode="length_only",
        )

        self.assertEqual(optimizer.turns, [1, 2])
        self.assertEqual(
            realized,
            ["injection-1", "injection-2"],
        )
        self.assertEqual(trajectories[0].turns[-1].injection_out, FINAL_ANSWER_INJECTION)
        self.assertTrue(all("summary only" in preview for preview in optimizer.previews))
        self.assertTrue(all("reaction-" not in preview for preview in optimizer.previews))
        self.assertIn("summary only", cot_preview)
        self.assertNotIn("reaction-", cot_preview)

    @patch("optimize.__main__.summarize_cot", side_effect=lambda cot: "summary-" + cot.rsplit("-", 1)[-1])
    def test_only_two_optimizer_requests_for_every_target_and_mode(self, _summarize):
        for model in TARGET_MODEL_OPTIONS:
            for mode in OPTIMIZATION_MODES:
                session = FakeSession("question")
                with self.subTest(model=model, mode=mode), \
                     patch.object(optimize_main, "make_session", return_value=session), \
                     patch.object(optimize_main.opt_module, "_call_api", side_effect=["injection-1", "injection-2"]) as call_api:
                    optimize_main.opt_module.init(None, "test", "chat", 3, get_prompt_module(model, mode), mode)
                    optimizer = optimize_main.opt_module.TrajectoryOptimizer()
                    samples = [SimpleNamespace(question="question", sample_id="sample")]
                    trajectories, realized, scores, _, cot_preview = optimize_main.run_batch(
                        optimizer, samples, extractor="fake", max_turns=3, mode=mode,
                    )

                    self.assertEqual(call_api.call_count, 2)
                    self.assertEqual(session.calls, [
                        ("send", "injection-1"), ("send", "injection-2"),
                        ("final_send", FINAL_ANSWER_INJECTION),
                    ])
                    self.assertEqual(realized, ["injection-1", "injection-2"])
                    optimizer.record_batch_trajectory(realized, (0.8, 0.0, 0.0), 0.0)
                    self.assertEqual(optimizer.current_trajectory, realized)
                    self.assertEqual(optimizer.best_trajectory, realized)
                    self.assertEqual(len(scores), 3)
                    self.assertEqual(len(trajectories[0].turns), 3)
                    self.assertEqual(trajectories[0].turns[-1].injection_out, FINAL_ANSWER_INJECTION)
                    self.assertIn("summary-2", cot_preview)

    @patch("optimize.__main__.summarize_cot")
    def test_provider_summary_is_reused(self, summarize):
        session = FakeSession("question")
        session.start()
        session.trajectory.turns[0].cot_summary = "provider summary"
        optimize_main._sync_cot_summaries([session])
        summarize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
