import importlib
import importlib.util
import unittest

from config import MAX_INJECTION_TURNS, OPTIMIZATION_MODES, TARGET_MODEL_OPTIONS, get_prompt_module
from extractors import get_target_config
from injection_io import parse_markdown_injections


class PromptRoutingTests(unittest.TestCase):
    def test_supported_modes(self):
        self.assertEqual(OPTIMIZATION_MODES, ("length_only", "length_lexical"))
        self.assertEqual(MAX_INJECTION_TURNS, 3)

    def test_standard_targets(self):
        self.assertEqual(get_target_config("deepseek-v4-flash")["extractor"], "extractors.extract_deepseek")
        for model in ("deepseek-v4-flash", "qwen3.5-plus", "glm-5.2"):
            self.assertEqual(
                get_prompt_module(model, "length_only"),
                "prompts.standard_length_only",
            )
            self.assertEqual(
                get_prompt_module(model, "length_lexical"),
                "prompts.standard_length_lexical",
            )

    def test_frontier_targets(self):
        models = (
            "gemini-2.5",
            "gemini-3.1-flash-lite",
            "gemini-3.5",
            "opus-4.6",
            "sonnet-4.6",
            "o4-mini",
        )
        for model in models:
            self.assertEqual(
                get_prompt_module(model, "length_only"),
                "prompts.frontier_length_only",
            )
            self.assertEqual(
                get_prompt_module(model, "length_lexical"),
                "prompts.frontier_length_lexical",
            )

    def test_official_deepseek_target(self):
        entry = get_target_config("deepseek-v4-flash-20261017")
        self.assertEqual(entry["model"], "deepseek-flash")
        self.assertEqual(entry["extractor"], "extractors.extract_deepseek_official")
        self.assertEqual(
            get_prompt_module("deepseek-v4-flash-20261017", "length_only"),
            "prompts.new_deepseek_length_only",
        )
        self.assertEqual(
            get_prompt_module("deepseek-v4-flash-20261017", "length_lexical"),
            "prompts.new_deepseek_length_lexical",
        )

    def test_every_target_has_an_extractor(self):
        for model, entry in TARGET_MODEL_OPTIONS.items():
            self.assertEqual(get_target_config(model), entry)
            self.assertIsNotNone(importlib.util.find_spec(entry["extractor"]))

    def test_every_prompt_module_is_importable(self):
        for model in TARGET_MODEL_OPTIONS:
            for mode in OPTIMIZATION_MODES:
                module = importlib.import_module(get_prompt_module(model, mode))
                self.assertTrue(module.OPTIMIZER_SYSTEM_PROMPT)
                self.assertTrue(module.INITIAL_EXPERIENCE_MD)
                injections = parse_markdown_injections(module.INITIAL_EXPERIENCE_MD)
                self.assertEqual([item["turn_idx"] for item in injections], [1, 2, 3])

    def test_gemini_31_model_id(self):
        self.assertEqual(
            TARGET_MODEL_OPTIONS["gemini-3.1-flash-lite"]["model"],
            "gemini-3.1-flash-lite",
        )


if __name__ == "__main__":
    unittest.main()
