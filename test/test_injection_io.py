import unittest

from injection_io import parse_markdown_injections


class InjectionIoTests(unittest.TestCase):
    def test_does_not_replay_proposals_from_search_knowledge(self):
        text = """\
## BEST UNIVERSAL TRAJECTORY SO FAR:
Turn 1: first injection
Turn 2: second injection
Evaluation Metrics:
score
## SEARCH KNOWLEDGE
### NEXT PERTURBATION PLAN
Turn 2: proposed replacement
"""
        self.assertEqual(
            parse_markdown_injections(text),
            [
                {"turn_idx": 1, "injection": "first injection"},
                {"turn_idx": 2, "injection": "second injection"},
            ],
        )

    def test_accepts_space_and_hyphen_turn_headings(self):
        text = """\
Turn-1:
first injection

Turn 2: second injection

Evaluation Metrics:
score
"""
        self.assertEqual(
            parse_markdown_injections(text),
            [
                {"turn_idx": 1, "injection": "first injection"},
                {"turn_idx": 2, "injection": "second injection"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
