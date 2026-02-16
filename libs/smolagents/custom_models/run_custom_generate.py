# run_custom_generate.py

from jet.libs.smolagents.custom_models import (
    ChatMessage,
    MessageRole,
    OpenAIModel,
)
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from smolagents.tools import Tool, tool

console = Console()


@tool
def add_numbers(a: int, b: int) -> int:
    """Adds two integers and returns the result.

    Useful when you need precise arithmetic without hallucinating numbers.

    Args:
        a: First integer
        b: Second integer

    Returns:
        The sum of a and b
    """
    return a + b


@tool
def greet_person(name: str) -> str:
    """Returns a friendly greeting for the given name.

    Args:
        name: The person's name to greet

    Returns:
        A friendly greeting message
    """
    return f"Hello {name}! How can I help you today? 😄"


def main():
    console.rule("Non-streaming generation example **with tools**", style="bold blue")

    model = OpenAIModel(
        temperature=0.6,
        max_tokens=600,
        verbose=True,
    )

    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "You are a helpful assistant. Use tools when precise calculation "
                "or formatted greeting is needed. Always think step-by-step."
            ),
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=(
                "Please greet Anna nicely and then tell me what is 47 + 89 + 156. "
                "Use tools for both parts if appropriate."
            ),
        ),
    ]

    available_tools: list[Tool] = [add_numbers, greet_person]

    with console.status("[bold cyan]Generating...", spinner="dots") as status:
        response = model.generate(
            messages=messages,
            stop_sequences=["</s>", "<|im_end|>"],
            tools_to_call_from=available_tools,
            tool_choice="auto",  # or "required" if you want to force tool use
        )

    # ────────────────────────────────────────────────
    console.print(
        Panel(
            "[dim]Model response area[/dim]"
            if not response.content and not response.tool_calls
            else "",
            title=f"[bold]Response from {model.model_id}[/bold]",
            subtitle=f"tokens: {response.token_usage or '?'}",
            border_style="bright_blue",
            expand=False,
            padding=(1, 2),
        ),
    )

    if response.tool_calls:
        console.print("[yellow bold]→ Tool calls detected[/]", end=" ")
        for tc in response.tool_calls:
            name = tc.function.name
            args = tc.function.arguments
            console.print(f"[cyan]{name}[/]({args}) → ", end="")
            # Optionally: simulate/show what would happen if executed
            if name == "add_numbers":
                # Example parsing for demonstration (should robustify for prod)
                try:
                    import ast

                    parsed = ast.literal_eval(args) if isinstance(args, str) else args
                    if isinstance(parsed, dict):
                        a = int(parsed.get("a", 0))
                        b = int(parsed.get("b", 0))
                        console.print(f"[green]{a + b}[/green]", end="")
                    else:
                        console.print("[red]parse error[/red]", end="")
                except Exception:
                    console.print("[red]parse error[/red]", end="")
            elif name == "greet_person":
                console.print("[green]greeting generated[/green]", end="")
            console.print()
    else:
        console.print("[dim]No tool calls[/dim]")

    if response.content:
        console.print(Markdown(response.content.strip()))
    else:
        console.print("[dim italic](empty content — probably tool call only)[/]")


if __name__ == "__main__":
    main()
