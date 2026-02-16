"""
examples/openai_model_usage_streaming.py

Demonstrates usage of smolagents.models.OpenAIModel with streaming.

Features shown:
- Streaming completion with rich live display
- Tool calling (function calling) with streaming deltas using class-based Tool

Requirements:
    pip install smolagents[openai] rich

Set your OpenAI API key:
    export OPENAI_API_KEY=sk-...
"""

import json
from typing import Any

from jet.libs.smolagents.custom_models import OpenAIModel
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from smolagents.models import (
    ChatMessage,
    ChatMessageStreamDelta,
    MessageRole,
    Tool,
)

console = Console()


def print_chat_message(msg: ChatMessage, title: str = "Message") -> None:
    """Pretty-print a ChatMessage using rich."""
    table = Table(show_header=False, expand=True, border_style="dim")
    table.add_row("[bold]Role[/bold]", f"[cyan]{msg.role}[/cyan]")

    if msg.content:
        table.add_row("[bold]Content[/bold]", msg.content)

    if msg.tool_calls:
        for i, tc in enumerate(msg.tool_calls, 1):
            args = (
                json.dumps(tc.function.arguments, indent=2)
                if isinstance(tc.function.arguments, dict)
                else tc.function.arguments
            )
            table.add_row(
                f"[bold]Tool Call {i}[/bold]",
                f"[yellow]{tc.function.name}[/yellow]\nArguments:\n[dim]{args}[/dim]",
            )

    if msg.token_usage:
        table.add_row(
            "[bold]Usage[/bold]",
            f"input: {msg.token_usage.input_tokens} • output: {msg.token_usage.output_tokens}",
        )

    console.print(Panel(table, title=title, border_style="blue"))


def print_streaming_delta(
    delta: ChatMessageStreamDelta, live: Live, current_text: list[str]
) -> None:
    """Update live display during streaming."""
    if delta.content:
        current_text[0] += delta.content
        live.update(Text(current_text[0], style="green"))

    if delta.tool_calls:
        tc_lines = []
        for tc_delta in delta.tool_calls:
            if tc_delta.function and tc_delta.function.name:
                tc_lines.append(f"→ Calling: [yellow]{tc_delta.function.name}[/yellow]")
            if tc_delta.function and tc_delta.function.arguments:
                tc_lines.append(f"  args: [dim]{tc_delta.function.arguments}[/dim]")
        if tc_lines:
            live.update(
                Text(current_text[0] + "\n\n" + "\n".join(tc_lines), style="green")
            )

    if delta.token_usage:
        usage_str = f"\n[dim]→ usage so far: {delta.token_usage}[/dim]"
        live.update(Text(current_text[0] + usage_str, style="green"))


# ──────────────────────────────────────────────────────────────────────────────
#   Configuration
# ──────────────────────────────────────────────────────────────────────────────

model = OpenAIModel(
    temperature=0.7,
    max_tokens=2048,
)


# ──────────────────────────────────────────────────────────────────────────────
#   Modern class-based Tool (recommended style in recent smolagents)
# ──────────────────────────────────────────────────────────────────────────────


class GetCurrentWeather(Tool):
    name = "get_current_weather"
    description = "Get the current weather in a given location"
    inputs = {
        "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
        },
        "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "Temperature unit",
            "nullable": True,
        },
    }
    output_type = "string"  # string literal — common & safe choice for text output

    def forward(self, location: str, unit: str = "celsius") -> str:
        # Placeholder — in real usage this would call weather API
        return f"Weather in {location}: 22° {unit.capitalize()[:1]}, sunny"


# ──────────────────────────────────────────────────────────────────────────────
#   Example 1: Streaming chat completion
# ──────────────────────────────────────────────────────────────────────────────

console.rule("Example 1 — Streaming completion")

messages = [
    ChatMessage(
        role=MessageRole.USER,
        content="Explain in one short paragraph why list comprehensions are considered more 'pythonic' than map() + lambda.",
    )
]

console.print("\n[bold cyan]Streaming:[/bold cyan]")

current_text = [""]

with Live("", refresh_per_second=15, console=console) as live:
    for delta in model.generate_stream(messages):
        print_streaming_delta(delta, live, current_text)

    live.update(Text(current_text[0] + "\n\n[Streaming finished]", style="bold green"))

# Append assistant's final message
final_content = current_text[0].strip()
messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=final_content))


# ──────────────────────────────────────────────────────────────────────────────
#   Example 2: Tool calling with streaming
# ──────────────────────────────────────────────────────────────────────────────

console.rule("Example 2 — Tool calling + streaming")

messages.append(
    ChatMessage(
        role=MessageRole.USER,
        content="What's the weather like right now in Kraków, Poland?",
    )
)

console.print("\n[bold cyan]Streaming with tool calling:[/bold cyan]")

current_text = [""]
tool_deltas: dict[int, dict[str, Any]] = {}

with Live("", refresh_per_second=12, console=console) as live:
    for delta in model.generate_stream(
        messages,
        tools_to_call_from=[GetCurrentWeather()],
        tool_choice="required",  # force tool use (optional)
    ):
        print_streaming_delta(delta, live, current_text)

        if delta.tool_calls:
            for tc_delta in delta.tool_calls or []:
                idx = tc_delta.index if tc_delta.index is not None else 0
                if idx not in tool_deltas:
                    tool_deltas[idx] = {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }

                d = tool_deltas[idx]
                if tc_delta.id:
                    d["id"] = tc_delta.id
                if tc_delta.type:
                    d["type"] = tc_delta.type
                if tc_delta.function:
                    if tc_delta.function.name:
                        d["function"]["name"] += tc_delta.function.name
                    if tc_delta.function.arguments:
                        d["function"]["arguments"] += tc_delta.function.arguments

    # Final update with parsed tool calls
    extra = ""
    if tool_deltas:
        extra = "\n\n[yellow]Tool calls detected:[/yellow]"
        for idx, tc in tool_deltas.items():
            args_str = tc["function"]["arguments"].strip()
            try:
                args_parsed = json.loads(args_str)
                args_str = json.dumps(args_parsed, indent=2)
            except json.JSONDecodeError:
                pass
            extra += f"\n→ [yellow]{tc['function']['name']}[/yellow]  id={tc['id']}\n  args:\n[dim]{args_str}[/dim]"

    live.update(
        Text(current_text[0] + extra + "\n\n[Streaming finished]", style="bold green")
    )

console.print(
    "\n[dim]Tip: in a real agent loop you would now execute the tool and append TOOL_RESPONSE message[/dim]\n"
)
