import io
import os
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import anthropic
import httpx2

import agent


class FakeStream:
    def __init__(self, chunks, message, error=None):
        self.text_stream = self._iter(chunks, error)
        self._message = message

    @staticmethod
    def _iter(chunks, error):
        yield from chunks
        if error:
            raise error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._message


class FakeClient:
    """模拟 client.beta.messages.stream(...)，记录每次请求的参数。"""

    def __init__(self, stream):
        self.calls = []

        def make_stream(**kwargs):
            self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
            return stream

        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=make_stream))


def final_message(stop_reason="end_turn"):
    return SimpleNamespace(stop_reason=stop_reason, content=["<content blocks>"])


class ChatTurnTest(unittest.TestCase):
    def test_normal_turn_streams_text_and_appends_both_messages(self):
        client = FakeClient(FakeStream(["你", "好"], final_message()))
        history, out = [], []

        message = agent.chat_turn(client, history, "hi", write=out.append)

        self.assertIsNotNone(message)
        self.assertEqual(out, ["你", "好"])
        self.assertEqual(
            history,
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": ["<content blocks>"]},
            ],
        )
        call = client.calls[0]
        self.assertEqual(call["model"], agent.MODEL)
        self.assertEqual(call["max_tokens"], 64000)
        self.assertEqual(call["system"], agent.SYSTEM)
        self.assertEqual(call["output_config"], {"effort": "medium"})
        self.assertEqual(call["betas"], ["server-side-fallback-2026-07-01"])
        self.assertEqual(call["fallbacks"], "default")
        self.assertEqual(call["messages"], [{"role": "user", "content": "hi"}])

    def test_history_is_sent_on_next_turn(self):
        history = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": ["x"]},
        ]
        client = FakeClient(FakeStream([], final_message()))

        agent.chat_turn(client, history, "b", write=lambda _: None)

        self.assertEqual(len(client.calls[0]["messages"]), 3)
        self.assertEqual(len(history), 4)

    def test_refusal_returns_none_and_leaves_history_unchanged(self):
        client = FakeClient(FakeStream([], final_message("refusal")))
        history = []

        self.assertIsNone(agent.chat_turn(client, history, "hi", write=lambda _: None))
        self.assertEqual(history, [])

    def test_error_mid_stream_rolls_back_user_message(self):
        client = FakeClient(FakeStream(["半"], final_message(), error=RuntimeError("boom")))
        history = []

        with self.assertRaises(RuntimeError):
            agent.chat_turn(client, history, "hi", write=lambda _: None)
        self.assertEqual(history, [])

    def test_keyboard_interrupt_rolls_back_user_message(self):
        client = FakeClient(
            FakeStream(["半"], final_message(), error=KeyboardInterrupt())
        )
        history = []

        with self.assertRaises(KeyboardInterrupt):
            agent.chat_turn(client, history, "hi", write=lambda _: None)
        self.assertEqual(history, [])

    def test_empty_content_leaves_history_unchanged(self):
        empty = SimpleNamespace(stop_reason="max_tokens", content=[])
        client = FakeClient(FakeStream([], empty))
        history = []

        message = agent.chat_turn(client, history, "hi", write=lambda _: None)

        self.assertIs(message, empty)
        self.assertEqual(history, [])

    def test_max_tokens_with_content_is_kept(self):
        client = FakeClient(FakeStream(["x"], final_message("max_tokens")))
        history = []

        agent.chat_turn(client, history, "hi", write=lambda _: None)

        self.assertEqual(len(history), 2)


REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


class MainTest(unittest.TestCase):
    def run_main(self, inputs, turns=()):
        """用给定的输入序列和 chat_turn 结果跑 main()，返回 (返回值, 输出, chat_turn 的 mock)。

        inputs 必须以 "exit" 或 EOFError 结尾，否则 mock 抛出的 StopIteration 会让测试崩溃。
        """
        out = io.StringIO()
        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}),
            patch("agent.anthropic.Anthropic"),
            patch("builtins.input", side_effect=inputs),
            patch("agent.chat_turn", side_effect=list(turns)) as chat_turn,
            redirect_stdout(out),
        ):
            code = agent.main()
        return code, out.getvalue(), chat_turn

    def test_missing_api_key_returns_1_without_creating_client(self):
        out = io.StringIO()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("agent.anthropic.Anthropic") as client_cls,
            redirect_stdout(out),
        ):
            code = agent.main()

        self.assertEqual(code, 1)
        self.assertIn("ANTHROPIC_API_KEY", out.getvalue())
        client_cls.assert_not_called()

    def test_exit_returns_0(self):
        code, _, chat_turn = self.run_main(["exit"])

        self.assertEqual(code, 0)
        chat_turn.assert_not_called()

    def test_eof_returns_0(self):
        code, _, _ = self.run_main([EOFError()])

        self.assertEqual(code, 0)

    def test_blank_input_is_skipped(self):
        code, _, chat_turn = self.run_main(["   ", "exit"])

        self.assertEqual(code, 0)
        chat_turn.assert_not_called()

    def test_refusal_and_empty_reply_print_hint_and_continue(self):
        empty = SimpleNamespace(stop_reason="max_tokens", content=[])
        code, out, chat_turn = self.run_main(["a", "b", "exit"], [None, empty])

        self.assertEqual(code, 0)
        self.assertEqual(chat_turn.call_count, 2)
        self.assertIn("模型拒绝回答", out)
        self.assertIn("模型没有返回内容", out)

    def test_authentication_error_returns_1(self):
        error = anthropic.AuthenticationError(
            "invalid x-api-key",
            response=httpx2.Response(401, request=REQUEST),
            body=None,
        )
        code, out, chat_turn = self.run_main(["a", "exit"], [error])

        self.assertEqual(code, 1)
        self.assertEqual(chat_turn.call_count, 1)
        self.assertIn("API key 无效", out)

    def test_connection_error_and_interrupt_continue(self):
        turns = [
            anthropic.APIConnectionError(request=REQUEST),
            KeyboardInterrupt(),
            final_message(),
        ]
        code, out, chat_turn = self.run_main(["a", "b", "c", "exit"], turns)

        self.assertEqual(code, 0)
        self.assertEqual(chat_turn.call_count, 3)
        self.assertIn("网络连接失败", out)
        self.assertIn("已中断本次回复", out)


if __name__ == "__main__":
    unittest.main()
