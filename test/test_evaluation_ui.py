"""Single-conversation demo checks, using simulated sessions only."""

import contextlib
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from config import TARGET_MODEL_OPTIONS
from evaluation_ui import server
from extractors import ARCHIVE_SUFFIX, FINAL_ANSWER_INJECTION
from metrics import InjectionTurn, TrajectoryResult, compute_length_metrics, count_tokens


class FakeSession:
    def __init__(self, question, model):
        self.question, self.model = question, model
        self.calls = []
        self.done = False
        self.reference = "First inspect the values, then compute the result."
        self.trajectory = TrajectoryResult()
        self.gate = None
        self.early = False
        self.fail = False

    def _record(self, content):
        tokens = count_tokens(content)
        self.trajectory.turns.append(InjectionTurn(
            len(self.trajectory.turns), "Returned reasoning", content, tokens,
            **compute_length_metrics(tokens, self.trajectory.ground_truth_tokens),
        ))

    def start(self):
        self.calls.append("start")
        if self.gate:
            self.gate.wait(timeout=5)
        closed = TARGET_MODEL_OPTIONS[self.model]["extractor"] in (
            "extractors.extract_gemini", "extractors.extract_claude")
        self.trajectory.ground_truth_cot = "" if closed else self.reference
        self.trajectory.ground_truth_cot_summary = "Returned reasoning summary." if closed else ""
        self.trajectory.ground_truth_tokens = count_tokens(self.reference)
        if self.early:
            self.done = True
            self.trajectory.final_output = "An early answer."
        else:
            self._record("First inspect the values.")

    def send(self, injection):
        self.calls.append(("send", injection))
        if self.fail:
            raise RuntimeError("Provider request failed")
        self.trajectory.turns[-1].injection_out = injection
        self._record(self.reference)

    def final_send(self, injection):
        if self.done:
            return
        self.calls.append(("final_send", injection))
        self.trajectory.turns[-1].injection_out = injection
        self.trajectory.final_output = "The final answer is 42."
        self.done = True


class DemoUITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name)
        for model in TARGET_MODEL_OPTIONS:
            if model in ("o4-mini", "deepseek-v4-flash-20261017"):
                continue
            for name in ("MATH500", "OpenThoughts"):
                path = self.data / model / name / "test.json"
                path.parent.mkdir(parents=True)
                count = 1 if model == "gemini-3.5" and name == "MATH500" else 2
                path.write_text(json.dumps([
                    {"sample_id": f"{name}-{i}", "question": f"{model} {name} question {i}",
                     "answer": "Dataset answer must not be sent", "ground_truth_cot": "Stored trace must not be used"}
                    for i in range(count)
                ]))
        self.sessions = []
        for name, kwargs in (("load_target", {"side_effect": lambda key: key}),
                             ("make_session", {"side_effect": self.make_session})):
            patcher = patch.object(server, name, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.app = server.DemoUI(self.data)

    def make_session(self, question, model):
        session = FakeSession(question, model)
        self.sessions.append(session)
        return session

    def configure(self, model="deepseek-v4-flash", dataset="MATH500", sample_idx=0):
        return self.app.configure({"model": model, "dataset": dataset, "sample_idx": sample_idx})

    def step(self, action, injection=None):
        state = self.app.snapshot()
        self.assertEqual(state["next_action"], action)
        self.app.act(action, state["revision"], injection)
        with self.app.condition:
            self.assertTrue(self.app.condition.wait_for(lambda: not self.app.busy, timeout=5))
        return self.app.snapshot()

    def test_selectors_use_file_index_and_only_load_a_question(self):
        excluded = {"o4-mini", "deepseek-v4-flash-20261017"}
        self.assertEqual({m["key"] for m in self.app.catalog["models"]}, set(TARGET_MODEL_OPTIONS) - excluded)
        for model in excluded:
            with self.assertRaises(ValueError):
                self.configure(model)
        state = self.configure(sample_idx=1)
        self.assertEqual(state["question"], "deepseek-v4-flash MATH500 question 1")
        self.assertEqual(state["sample_id"], "MATH500-1")
        self.assertIsNone(state["response"])
        self.assertEqual(self.sessions, [])
        self.assertNotIn("summary", state)
        self.assertNotIn("seed", state["config"])
        self.step("start")
        previous = self.app.session
        state = self.configure("gemini-2.5", "OpenThoughts", 0)  # Allowed between requests.
        self.assertEqual(state["question"], "gemini-2.5 OpenThoughts question 0")
        self.assertEqual(state["events"], [])
        self.assertEqual(previous.calls, ["start"])
        self.assertEqual(state["next_action"], "start")
        with self.assertRaises(ValueError):
            self.configure(sample_idx=2)

    def test_live_session_and_exact_edited_injections_for_every_target(self):
        for model in (m["key"] for m in self.app.catalog["models"]):
            with self.subTest(model=model):
                state = self.configure(model)
                before = state["config"]["injections"]
                state = self.step("start")
                active = self.app.session
                self.assertEqual(active.question, f"{model} MATH500 question 0" + ARCHIVE_SUFFIX)
                self.assertTrue(active.retain_reasoning_context)
                self.assertTrue(active.skip_cot_summary)
                self.assertEqual(active.calls, ["start"])
                self.assertEqual(state["response"]["ground_truth_cot"], active.trajectory.ground_truth_cot)
                self.assertNotIn("Stored trace", json.dumps(state["response"]))
                for i, _ in enumerate(before):
                    edited = f"  User-edited injection {i + 1}\n<literal markup>\n"
                    state = self.step("inject", edited)
                    self.assertIs(self.app.session, active)
                    self.assertEqual(active.calls[-1], ("send", edited))
                    self.assertEqual(state["events"][-1]["input"], edited)
                edited_final = "User-edited final instruction.\n"
                state = self.step("final", edited_final)
                self.assertEqual(active.calls[-1], ("final_send", edited_final))
                self.assertEqual(state["phase"], "complete")
                self.assertIsNone(state["next_action"])
                self.assertEqual(state["config"]["injections"], before)
                closed = state["config"]["length_only"]
                if closed:
                    self.assertNotIn("token_em_threshold", state["config"])
                    self.assertEqual(state["response"]["ground_truth_cot_summary"], "Returned reasoning summary.")
                for scratchpad in state["response"]["scratchpads"]:
                    self.assertEqual(scratchpad["ground_truth_cot_tokens"], active.trajectory.ground_truth_tokens)
                    self.assertEqual(scratchpad["token_em"] is None, closed)

    def test_model_specific_questions_and_sample_counts(self):
        first = self.configure(sample_idx=1)
        self.assertEqual(first["config"]["sample_count"], 2)
        second = self.configure("gemini-3.5")
        self.assertEqual(second["config"]["sample_count"], 1)
        self.assertEqual(second["question"], "gemini-3.5 MATH500 question 0")
        with self.assertRaises(ValueError):
            self.configure("gemini-3.5", sample_idx=1)

    def test_bundled_examples_cover_each_target_and_dataset(self):
        app = server.DemoUI()
        self.assertEqual(set(app.datasets), {m["key"] for m in app.catalog["models"]})
        for model, datasets in app.datasets.items():
            self.assertEqual(set(datasets), {"JEEBench", "LiveCodeBench", "MATH500", "OpenThoughts"})
            for name, samples in datasets.items():
                expected = 4 if model == "gemini-3.5" and name == "MATH500" else 10
                self.assertEqual(len(samples), expected)
                self.assertEqual(len({s["sample_id"] for s in samples}), expected)
                self.assertTrue(all(set(s) == {"sample_id", "question"} and s["question"] for s in samples))

    def test_default_final_and_provider_length_counts(self):
        state = self.configure("sonnet-4.6")
        self.assertEqual(len(state["config"]["injections"]), 1)
        self.step("start")
        self.step("inject")
        state = self.step("final")
        self.assertEqual(self.app.session.calls[-1], ("final_send", FINAL_ANSWER_INJECTION))
        trajectory = self.app.session.trajectory
        trajectory.ground_truth_tokens = 900
        trajectory.turns[0].scratchpad_in_tokens = 891
        trajectory.turns[0].length_error = 0.01
        response = server.response_view(trajectory, True)
        self.assertEqual(response["scratchpads"][0]["scratchpad_tokens"], 891)
        self.assertEqual(response["scratchpads"][0]["ground_truth_cot_tokens"], 900)
        self.assertEqual(response["scratchpads"][0]["length_error"], 0.01)
        self.assertIsNone(response["scratchpads"][0]["token_em"])

    def test_busy_duplicate_and_out_of_order_actions_do_not_send_requests(self):
        initial = self.configure()
        gate = threading.Event()
        def gated(question, model):
            session = self.make_session(question, model)
            session.gate = gate
            return session
        try:
            with patch.object(server, "make_session", side_effect=gated):
                self.app.act("start", initial["revision"])
                for action in ("start", "inject", "final"):
                    with self.assertRaises(ValueError):
                        self.app.act(action, initial["revision"])
                with self.assertRaises(ValueError):
                    self.configure("gemini-2.5")
                gate.set()
                with self.app.condition:
                    self.assertTrue(self.app.condition.wait_for(lambda: not self.app.busy, timeout=5))
            current = self.app.snapshot()
            for action, revision, text in (("final", current["revision"], "final"),
                                           ("inject", initial["revision"], "stale"),
                                           ("inject", current["revision"], "  ")):
                with self.assertRaises(ValueError):
                    self.app.act(action, revision, text)
            self.assertEqual(self.sessions[0].calls, ["start"])
        finally:
            gate.set()

    def test_early_completion_and_request_failure(self):
        self.configure()
        def early(question, model):
            session = self.make_session(question, model)
            session.early = True
            return session
        with patch.object(server, "make_session", side_effect=early):
            state = self.step("start")
        self.assertEqual(state["response"]["scratchpads"], [])
        self.assertEqual(state["phase"], "complete")
        self.assertIsNone(state["next_action"])
        self.configure(sample_idx=1)
        self.step("start")
        self.app.session.fail = True
        with contextlib.redirect_stderr(io.StringIO()):
            state = self.step("inject")
        self.assertEqual(state["phase"], "error")
        self.assertIn("Provider request failed", state["error"])
        self.assertFalse(state["busy"])
        self.assertEqual(list(self.data.rglob('samples.jsonl')), [])

    def test_http_assets_live_events_and_local_post_guard(self):
        http = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        http.app = self.app
        threading.Thread(target=http.serve_forever, daemon=True).start()
        self.addCleanup(http.server_close)
        self.addCleanup(http.shutdown)
        base = f"http://127.0.0.1:{http.server_port}"
        for path in ("/", "/style.css", "/app.js", "/api/catalog", "/api/state"):
            with urlopen(base + path, timeout=2) as response:
                self.assertEqual(response.status, 200)
                self.assertGreater(len(response.read()), 0)
        data = json.dumps({"model": "sonnet-4.6", "dataset": "MATH500", "sample_idx": 1}).encode()
        request = Request(base + "/api/configure", data, {"Content-Type": "application/json"})
        with urlopen(request, timeout=2) as response:
            self.assertEqual(json.load(response)["question"], "sonnet-4.6 MATH500 question 1")
        with urlopen(base + "/api/events", timeout=2) as response:
            self.assertEqual(response.readline().strip(), b"event: state")
            state = json.loads(response.readline().removeprefix(b"data: "))
            self.assertEqual(state["config"]["sample_idx"], 1)
        request.add_header("Origin", "https://unrelated.example")
        with self.assertRaises(HTTPError) as error:
            urlopen(request, timeout=2)
        self.assertEqual(error.exception.code, 403)
        self.assertEqual(server.json_bytes({"missing": float("nan")}), b'{"missing": null}')


if __name__ == "__main__":
    unittest.main()
