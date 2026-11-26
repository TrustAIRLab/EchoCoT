import contextlib
import copy
import importlib
import io
import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

from anthropic.types import RedactedThinkingBlock, TextBlock, ThinkingBlock, ToolUseBlock
from google.genai import types
from openai.types.responses import ResponseReasoningItem

from config import TARGET_MODEL_OPTIONS
from extractors import FINAL_ANSWER_INJECTION


TRACES = [f"  reasoning {index}\n分析  " for index in range(3)]


def tool_calls(turn, count):
    return [NS(index=i, id=f"call-{turn}-{i}", type="function",
               function=NS(name="scratchpad", arguments='{ "content": "archive" }'))
            for i in range(count)]


def run_session(module, model, client):
    with patch.object(module, "_client", client), patch.object(module, "_model_id", model), \
         contextlib.redirect_stdout(io.StringIO()):
        session = module.new_session("question")
        session.mode = "length_only"
        session.skip_cot_summary = True
        session.start()
        session.send("feedback-1")
        session.send("feedback-2")
        session.final_send(FINAL_ANSWER_INJECTION)
        return session


class ReasoningReplayTests(unittest.TestCase):
    def test_chat_targets_preserve_reasoning_through_final_answer(self):
        models = ("deepseek-v4-flash", "deepseek-v4-flash-20261017", "qwen3.5-plus", "glm-5.2")
        for model in models:
            for count in (1, 2):
                with self.subTest(model=model, tool_calls=count):
                    entry = TARGET_MODEL_OPTIONS[model]
                    module = importlib.import_module(entry["extractor"])
                    requests, details = [], []

                    def create(**kwargs):
                        requests.append(copy.deepcopy(kwargs))
                        turn = len(requests) - 1
                        reasoning = TRACES[turn] if turn < 3 else "final reasoning"
                        calls = tool_calls(turn, count) if turn < 3 else []
                        detail = [{"type": "reasoning.text", "text": reasoning},
                                  {"type": "reasoning.encrypted", "data": f"opaque-{turn}"}]
                        details.append(detail)
                        message = NS(content="" if calls else "answer", reasoning_content=reasoning,
                                     reasoning_details=detail, tool_calls=calls)
                        if kwargs.get("stream"):
                            return iter([NS(choices=[NS(delta=message)])])
                        return NS(choices=[NS(message=message)])

                    client = MagicMock()
                    client.chat.completions.create.side_effect = create
                    session = run_session(module, entry["model"], client)
                    self.assertEqual(len(requests), 4)
                    self.assertTrue(session.done)
                    for turn, request in enumerate(requests[1:], 1):
                        history = request["messages"]
                        assistants = [message for message in history if message["role"] == "assistant"]
                        if model == "deepseek-v4-flash":
                            self.assertEqual([message["reasoning_details"] for message in assistants], details[:turn])
                        else:
                            self.assertEqual([message["reasoning_content"] for message in assistants], TRACES[:turn])
                        expected_ids = [f"call-{t}-{i}" for t in range(turn) for i in range(count)]
                        self.assertEqual([message["tool_call_id"] for message in history if message["role"] == "tool"], expected_ids)
                        if model == "qwen3.5-plus":
                            self.assertTrue(all(call["function"]["arguments"] == '{ "content": "archive" }'
                                                for message in assistants for call in message["tool_calls"]))
                    if model == "glm-5.2":
                        self.assertEqual(requests[-1]["extra_body"]["thinking"], {"type": "enabled", "clear_thinking": False})

    def test_openrouter_preserves_opaque_reasoning_without_visible_text(self):
        module = importlib.import_module("extractors.extract_deepseek")
        details = [{"type": "reasoning.encrypted", "data": "opaque state", "index": 0}]
        message = NS(content="", tool_calls=[], reasoning_details=details)
        session = NS(messages=[])
        module.SampleSession._append_assistant(session, message, "")
        self.assertEqual(session.messages[-1]["reasoning_details"], details)
        message.reasoning_details = []
        module.SampleSession._append_assistant(session, message, "plain reasoning")
        self.assertEqual(session.messages[-1]["reasoning_content"], "plain reasoning")
        session.retain_reasoning_context = False
        message.reasoning_details = details
        module.SampleSession._append_assistant(session, message, "")
        self.assertNotIn("reasoning_details", session.messages[-1])

    def test_gemini_target_uses_vertex_without_an_api_key(self):
        module = importlib.import_module("extractors.extract_gemini")
        with patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "test-project"}, clear=True), \
             patch.object(module.genai, "Client") as vertex_client, \
             patch.object(module, "_client"), patch.object(module, "_model_id"):
            module.load_model("gemini-2.5-flash")
            vertex_client.assert_called_once_with(
                vertexai=True, project="test-project", location="global",
            )

    def test_gemini_preserves_complete_parts_and_signatures(self):
        module = importlib.import_module("extractors.extract_gemini")
        for model in ("gemini-2.5-flash", "gemini-3.1-flash-lite", "gemini-3.5-flash"):
            with self.subTest(model=model):
                requests, contents = [], []

                def create(**kwargs):
                    requests.append(copy.deepcopy(kwargs))
                    turn = len(requests) - 1
                    parts = [types.Part(text="answer")]
                    if turn < 3:
                        parts = [types.Part(text=TRACES[turn], thought=True)] + [
                            types.Part(function_call=types.FunctionCall(name="scratchpad", args={"content": "archive"}),
                                       thought_signature=f"signature-{turn}-{i}".encode()) for i in range(2)]
                    content = types.Content(role="model", parts=parts)
                    contents.append(content)
                    return NS(candidates=[NS(content=content)], usage_metadata=NS(thoughts_token_count=10))

                client = MagicMock()
                client.models.generate_content.side_effect = create
                client.models.count_tokens.return_value = NS(total_tokens=10)
                session = run_session(module, model, client)
                self.assertTrue(session.done)
                self.assertEqual(len(requests), 4)
                for turn, request in enumerate(requests[1:], 1):
                    replayed = [content for content in request["contents"] if content.role == "model"]
                    self.assertEqual(replayed, contents[:turn])
                    self.assertEqual(len(request["contents"][-1].parts), 2)
                    self.assertTrue(request["config"].thinking_config.include_thoughts)

    def test_claude_preserves_all_thinking_blocks_and_tool_results(self):
        module = importlib.import_module("extractors.extract_claude")
        for model in ("claude-opus-4-6", "claude-sonnet-4-6"):
            with self.subTest(model=model):
                requests, contents = [], []

                def stream(**kwargs):
                    requests.append(copy.deepcopy(kwargs))
                    turn = len(requests) - 1
                    content = [TextBlock(type="text", text="answer")]
                    if turn < 3:
                        content = [ThinkingBlock(type="thinking", thinking=TRACES[turn], signature=f"signature-{turn}"),
                                   RedactedThinkingBlock(type="redacted_thinking", data=f"opaque-{turn}")] + [
                            ToolUseBlock(type="tool_use", id=f"call-{turn}-{i}", name="scratchpad", input={"content": "archive"})
                            for i in range(2)]
                    contents.append([block.model_dump() for block in content])
                    manager = MagicMock()
                    manager.__enter__.return_value.get_final_message.return_value = NS(
                        content=content, usage=NS(output_tokens_details={"thinking_tokens": 10}))
                    return manager

                client = MagicMock()
                client.messages.stream.side_effect = stream
                client.messages.count_tokens.return_value = NS(input_tokens=10)
                session = run_session(module, model, client)
                self.assertTrue(session.done)
                self.assertEqual(len(requests), 4)
                for turn, request in enumerate(requests[1:], 1):
                    replayed = [message["content"] for message in request["messages"] if message["role"] == "assistant"]
                    self.assertEqual(replayed, contents[:turn])
                    self.assertEqual([block["tool_use_id"] for block in request["messages"][-1]["content"]],
                                     [f"call-{turn - 1}-{i}" for i in range(2)])

    def test_openai_continues_reasoning_response_chain_with_all_tool_results(self):
        module = importlib.import_module("extractors.extract_gpt")
        requests = []

        def create(**kwargs):
            requests.append(copy.deepcopy(kwargs))
            turn = len(requests) - 1
            output = []
            if turn < 3:
                output = [ResponseReasoningItem(type="reasoning", id=f"reasoning-{turn}", summary=[],
                                                encrypted_content=f"opaque-{turn}")] + [
                    NS(type="function_call", call_id=f"call-{turn}-{i}", name="scratchpad", arguments='{"content":"archive"}')
                    for i in range(2)]
            return NS(id=f"response-{turn}", output=output, output_text="answer" if turn == 3 else "",
                      usage=NS(output_tokens_details=NS(reasoning_tokens=10)))

        client = MagicMock()
        client.responses.create.side_effect = create
        session = run_session(module, "o4-mini", client)
        self.assertTrue(session.done)
        self.assertEqual(session.trajectory.turns[0].scratchpad_in, "archive")
        self.assertEqual(len(requests), 4)
        self.assertIsNone(requests[0]["previous_response_id"])
        for turn, request in enumerate(requests[1:], 1):
            self.assertEqual(request["previous_response_id"], f"response-{turn - 1}")
            self.assertEqual([item["call_id"] for item in request["input"]], [f"call-{turn - 1}-{i}" for i in range(2)])
            self.assertEqual(request["reasoning"]["effort"], "medium")


if __name__ == "__main__":
    unittest.main()
