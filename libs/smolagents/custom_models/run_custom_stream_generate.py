# run_custom_stream_generate.py

from collections.abc import Generator

from jet.libs.smolagents.custom_models import (  # noqa: E402
    ChatMessage,
    ChatMessageStreamDelta,
    MessageRole,
    OpenAIModel,
)
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from smolagents.tools import Tool, tool

console = Console(highlight=False)


@tool
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers. Use for precise floating-point math.

    Args:
        a: First number (can be float)
        b: Second number (can be float)

    Returns:
        Product of a and b
    """
    return a * b


@tool
def shout(text: str) -> str:
    """Returns the input text in uppercase with exclamation marks.

    Args:
        text: The text to shout

    Returns:
        The shouted version of the text
    """
    return f"{text.upper()} !!!"


def print_delta_live(delta: ChatMessageStreamDelta, buffer: Text):
    """Append delta to live buffer + highlight tool progress"""
    if delta.content:
        buffer.append(delta.content, style="white")
    if delta.tool_calls:
        for tc in delta.tool_calls:
            if tc.function and tc.function.name:
                buffer.append(
                    f"\n→ Tool call started: [bold magenta]{tc.function.name}[/]",
                    style="bold magenta",
                )
            if tc.function and tc.function.arguments:
                buffer.append(tc.function.arguments, style="dim cyan")


def main():
    console.rule("Streaming generation example **with tools**", style="bold green")

    model = OpenAIModel(
        temperature=0.7,
        max_tokens=800,
        verbose=False,
    )

    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "You are a cheerful calculator assistant. "
                "Use tools for math and shouting when emphasis is needed."
            ),
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=(
                "Shout 'We did it!' and then compute 12.5 × 8.2. "
                "Use the appropriate tools please."
            ),
        ),
    ]

    available_tools: list[Tool] = [multiply, shout]

    console.print("[dim]Sending messages with tools...[/]")

    stream: Generator[ChatMessageStreamDelta, None, None] = model.generate_stream(
        messages=messages,
        stop_sequences=["<|eot_id|>", "</s>"],
        tools_to_call_from=available_tools,
        tool_choice="auto",
    )

    accumulated = Text()
    final_usage = None

    with Live(
        Panel(
            accumulated, title="Streaming response + tool calls", border_style="green"
        ),
        refresh_per_second=12,
        console=console,
    ) as live:
        for delta in stream:
            print_delta_live(delta, accumulated)
            live.update(
                Panel(
                    accumulated,
                    title="Streaming response + tool calls",
                    border_style="green",
                )
            )

    console.print("\n" + "─" * 80)

    # Final summary
    console.print(
        Panel(
            Markdown(accumulated.plain.strip() or "(no content)"),
            title=f"Final streamed answer — {model.model_id}",
            subtitle="streaming complete",
            border_style="bright_green",
        )
    )

    # If you collected final ChatMessage somehow (not directly from stream generator),
    # you could also show .tool_calls here — but for pure streaming demo we focus on deltas.


if __name__ == "__main__":
    main()
