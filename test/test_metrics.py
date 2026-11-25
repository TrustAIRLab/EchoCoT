import unittest

try:
    import tiktoken  # noqa: F401
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("tiktoken is not installed") from exc

from metrics import compute_eval_metrics, compute_token_em, is_attack_successful


class FormalMetricTests(unittest.TestCase):
    def test_identical_trace_passes_all_asr_thresholds(self):
        trace = "First inspect the values, then compute the result."
        metrics = compute_eval_metrics(trace, trace)

        self.assertEqual(metrics["token_em"], 1.0)
        self.assertTrue(metrics["asr_90"])
        self.assertTrue(metrics["asr_95"])
        self.assertTrue(metrics["asr_99"])
        self.assertTrue(is_attack_successful(trace, trace, 99))

    def test_token_em_is_reference_coverage(self):
        reference = "alpha beta gamma delta"
        score = compute_token_em("alpha beta", reference)

        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)


if __name__ == "__main__":
    unittest.main()
