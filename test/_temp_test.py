import json
import os
import time

from headroom import compress  # pip install "headroom-ai[all]"
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

# Initialize rich console
console = Console()

# Local model config — matches your llama-server setup
LOCAL_MODEL = "Qwen3.5-0.8B-Q4_K_M"  # model string (llama-server ignores it, but required by client)
LOCAL_BASE_URL = os.getenv("LLAMA_CPP_LLM_URL", "http://localhost:8080/v1")

# Your llama-server has: -c 8192 context, -b 1024 batch, -ub 512 micro-batch
# Leave headroom for the system prompt, user message, and LLM output (~1500 tokens)
TOKEN_BUDGET = 5500  # safe headroom within 8192 ctx

# Example: Large tool output (e.g., search results or DB query)
# Reduced from 500 to 50 items — 500 items would already overflow 8192 tokens
large_tool_output = {
    "results": [
        {
            "id": i,
            "title": f"Item {i}",
            "description": f"Long description with details {i} "
            * 10,  # trimmed from *50
            "score": 100 - i,
        }
        for i in range(50)  # reduced from 500; compress() will handle the rest
    ],
    "metadata": {"total": 50, "query": "example search"},
}

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Analyze tool outputs carefully.",
    },
    {"role": "user", "content": "Summarize the top results from this search."},
    {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": json.dumps(large_tool_output),
    },
]

# === Compression Step ===
console.print(
    Panel.fit("[bold cyan]📦 Compression Phase[/bold cyan]", border_style="cyan")
)

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    console=console,
    transient=True,
) as progress:
    task = progress.add_task("[cyan]Compressing messages...", total=None)
    # Pass token_budget to hard-cap tokens for your 8192-ctx local model.
    # model= here is used by headroom to pick compression strategy (not sent to llama-server).
    result = compress(
        messages,
        model="gpt-4o",  # headroom uses this for strategy selection only
        token_budget=TOKEN_BUDGET,  # enforce fit within llama-server context
        ccr_enabled=True,  # reversible compression (default)
    )
    progress.update(task, completed=True)

# Display compression stats in a table
stats_table = Table(title="Compression Statistics", box=box.ROUNDED, style="cyan")
stats_table.add_column("Metric", style="bold cyan")
stats_table.add_column("Value", style="green")
stats_table.add_row("Tokens before", f"{result.tokens_before:,}")
stats_table.add_row("Tokens after", f"{result.tokens_after:,}")
stats_table.add_row(
    "Tokens saved", f"{result.tokens_saved:,} ({result.compression_ratio:.1%})"
)
stats_table.add_row("Transforms applied", str(result.transforms_applied))
console.print(stats_table)

# Show compressed messages preview (fixed for non-JSON serializable object)
if result.tokens_after > 0:
    # Access the compressed messages directly
    compressed_messages = result.messages if hasattr(result, "messages") else result
    compressed_preview = json.dumps(compressed_messages, indent=2, ensure_ascii=False)[
        :500
    ]
    console.print(
        Panel(
            Syntax(
                compressed_preview
                + ("..." if len(json.dumps(compressed_messages)) > 500 else ""),
                "json",
                theme="monokai",
            ),
            title="[bold]Compressed Messages Preview[/bold]",
            border_style="blue",
        )
    )

# Abort if still too large (safety check)
if result.tokens_after > TOKEN_BUDGET:
    console.print(
        "[bold red]❌ Error:[/bold red] Compressed messages still exceed token budget!"
    )
    raise RuntimeError(
        f"Compressed messages still too large: {result.tokens_after} tokens "
        f"(budget: {TOKEN_BUDGET}). Reduce input size or lower token_budget."
    )

# === Send to local llama-server ===
client = OpenAI(
    base_url=LOCAL_BASE_URL,
    api_key="not-needed",  # llama-server doesn't require an API key
)

console.print(
    Panel.fit("[bold green]🤖 LLM Generation Phase[/bold green]", border_style="green")
)

# Show messages being sent
messages_preview = json.dumps(messages, indent=2, ensure_ascii=False)[:800]
console.print(
    Panel(
        Syntax(
            messages_preview + ("..." if len(json.dumps(messages)) > 800 else ""),
            "json",
            theme="monokai",
        ),
        title="[bold]Messages to LLM[/bold]",
        border_style="yellow",
    )
)

stream: Stream[ChatCompletionChunk] = client.chat.completions.create(
    model="Qwen/Qwen3.5-2B",
    messages=messages,
    max_tokens=500,
    temperature=1.0,
    top_p=1.0,
    presence_penalty=2.0,
    extra_body={
        "top_k": 20,
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
    },
    stream=True,
)

console.print("\n[bold cyan]📝 LLM Response:[/bold cyan]")
console.print("[dim]" + "─" * 50 + "[/dim]")

# Use standard print with flush for immediate token display
char_count = 0
start_time = time.time()

for part in stream:
    if part.choices and part.choices[0].delta:
        delta = part.choices[0].delta
        if delta.content:
            chunk = delta.content
            char_count += len(chunk)
            # Print with cyan color using ANSI codes, flush immediately
            print(f"\033[36m{chunk}\033[0m", end="", flush=True)

elapsed_time = time.time() - start_time

print()  # New line after streaming
console.print("[dim]" + "─" * 50 + "[/dim]")
console.print(
    f"[dim]📊 Response Stats: {char_count:,} chars in {elapsed_time:.2f}s ({char_count / elapsed_time:.1f} chars/s)[/dim]"
)
console.print("[bold green]✅ Complete![/bold green]")
