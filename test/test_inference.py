import unittest
from types import SimpleNamespace
from unittest.mock import patch

from test import __main__ as inference


def make_turn(turn_idx, scratchpad):
    return SimpleNamespace(
        turn_idx=turn_idx,
        cot=f"reaction-{turn_idx}",
        scratchpad_in=scratchpad,
        injection_out="",
        scratchpad_in_tokens=len(scratchpad),
    )


class FakeSession:
    def __init__(self):
        self.done = False
        self.trajectory = SimpleNamespace(turns=[], final_output="", ground_truth_tokens=10)
        self.received = []

    def start(self):
        self.trajectory.turns.append(make_turn(0, "initial"))

    def send(self, injection):
        self.trajectory.turns[-1].injection_out = injection
        self.received.append(injection)
        self.trajectory.turns.append(
            make_turn(len(self.trajectory.turns), f"archive-{len(self.received)}")
        )

    def final_send(self, injection):
        self.received.append(injection)
        self.trajectory.final_output = "final"
        self.done = True


class InferenceTests(unittest.TestCase):
    @patch("test.__main__.make_session")
    @patch("test.__main__.load_target", return_value="fake-extractor")
    def test_custom_trajectory_uses_shared_session(self, load_target, make_session):
        session = FakeSession()
        make_session.return_value = session
        injections = [
            {"turn_idx": 1, "injection": "first"},
            {"turn_idx": 2, "injection": "second"},
        ]

        result = inference.run_inference("deepseek-v4-flash", "question", injections)

        load_target.assert_called_once_with("deepseek-v4-flash")
        self.assertTrue(make_session.call_args.args[0].endswith(inference.ARCHIVE_SUFFIX))
        self.assertEqual(
            session.received,
            ["first", "second", inference.FINAL_ANSWER_INJECTION],
        )
        self.assertEqual(result["final_answer"], "final")
        self.assertEqual([turn["turn_idx"] for turn in result["turns"]], [0, 1, 2])
        self.assertTrue(all(turn["ground_truth_cot_tokens"] == 10 for turn in result["turns"]))
        self.assertEqual(session.mode, "length_only")
        self.assertTrue(session.skip_cot_summary)

    @patch("test.__main__.make_session")
    @patch("test.__main__.load_target", return_value="fake-extractor")
    def test_fixed_acceptance_is_not_sent_twice(self, _load_target, make_session):
        session = FakeSession()
        make_session.return_value = session
        injections = [
            {"turn_idx": 1, "injection": "first"},
            {"turn_idx": 2, "injection": "second"},
            {"turn_idx": 3, "injection": inference.FINAL_ANSWER_INJECTION},
        ]

        inference.run_inference("deepseek-v4-flash", "question", injections)

        self.assertEqual(
            session.received,
            ["first", "second", inference.FINAL_ANSWER_INJECTION],
        )


if __name__ == "__main__":
    unittest.main()
