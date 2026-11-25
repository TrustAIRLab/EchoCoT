import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import tiktoken  # noqa: F401
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("tiktoken is not installed") from exc

from extractors import FINAL_ANSWER_INJECTION
from metrics import InjectionTurn, TrajectoryResult
from optimize import __main__ as optimize_main


class FakeSession:
    def __init__(self, _question, _extractor=None):
        self.done = False
        self.trajectory = TrajectoryResult(ground_truth_tokens=10)
        self.received = []

    def start(self):
        self.trajectory.turns.append(self._turn(0))

    def send(self, injection):
        self.received.append(injection)
        self.trajectory.turns[-1].injection_out = injection
        self.trajectory.turns.append(self._turn(len(self.trajectory.turns)))

    def final_send(self, injection):
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
            ["injection-1", "injection-2", FINAL_ANSWER_INJECTION],
        )
        self.assertEqual(trajectories[0].turns[-1].injection_out, FINAL_ANSWER_INJECTION)
        self.assertTrue(all("summary only" in preview for preview in optimizer.previews))
        self.assertTrue(all("reaction-" not in preview for preview in optimizer.previews))
        self.assertIn("summary only", cot_preview)
        self.assertNotIn("reaction-", cot_preview)

    @patch("optimize.__main__.summarize_cot")
    def test_provider_summary_is_reused(self, summarize):
        session = FakeSession("question")
        session.start()
        session.trajectory.turns[0].cot_summary = "provider summary"
        optimize_main._sync_cot_summaries([session])
        summarize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
