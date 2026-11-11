"""最简单的命令行 AI 智能体：多轮对话，流式输出。

用法：
    export ANTHROPIC_API_KEY=你的key
    python agent.py
"""

import os
import sys

import anthropic

MODEL = "claude-opus-5-5"
SYSTEM = "你是一个友好、简洁的中文助手。"


def chat_turn(client, history, user_input, write=sys.stdout.write):
    """发送一轮对话并流式输出回复，成功后把这一轮追加进 history。

    返回最终消息；被模型拒答时返回 None。出错或拒答时 history 保持不变。
    """
    history.append({"role": "user", "content": user_input})
    try:
        with client.beta.messages.stream(
            model=MODEL,
            max_tokens=64000,
            system=SYSTEM,
            output_config={"effort": "medium"},
            # 模型拒答时由服务端自动换模型重试
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=history,
        ) as stream:
            for text in stream.text_stream:
                write(text)
            message = stream.get_final_message()
    except BaseException:
        history.pop()
        raise

    if message.stop_reason == "refusal":
        history.pop()
        return None
    # 原样回传 content（含 thinking 块），保证下一轮请求合法
    history.append({"role": "assistant", "content": message.content})
    return message


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("请先设置环境变量 ANTHROPIC_API_KEY。")
        return 1

    client = anthropic.Anthropic()
    history = []
    print("开始对话吧（输入 exit 退出）。")

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            return 0

        print("AI: ", end="", flush=True)
        try:
            if chat_turn(client, history, user_input) is None:
                print("\n（这个问题模型拒绝回答，换个问法试试。）")
            else:
                print()
        except anthropic.AuthenticationError:
            print("\nAPI key 无效，请检查 ANTHROPIC_API_KEY。")
            return 1
        except anthropic.RateLimitError:
            print("\n请求太频繁，稍后再试。")
        except anthropic.APIStatusError as e:
            print(f"\n请求失败（{e.status_code}）：{e.message}")
        except anthropic.APIConnectionError:
            print("\n网络连接失败，请检查网络。")
        except KeyboardInterrupt:
            print("\n（已中断本次回复）")


if __name__ == "__main__":
    sys.exit(main())
