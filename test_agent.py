import unittest
from types import SimpleNamespace

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


if __name__ == "__main__":
    unittest.main()
