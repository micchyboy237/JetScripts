import json
import logging
import os
import re
import time
from contextlib import contextmanager

# NEW: Import from llm_utils instead of raw OpenAI
from jet.adapters.llama_cpp.llm_utils import (
    LLMContext,
    chat,
)
from jet.db.postgres.cleanup import drop_table_if_exists, drop_type_if_exists
from jet.db.postgres.config import (
    DEFAULT_DB,
    DEFAULT_HOST,
    DEFAULT_PASSWORD,
    DEFAULT_PORT,
    DEFAULT_USER,
)
from mem0 import Memory
from mem0.configs.base import MemoryConfig
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

console = Console()
logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

console.print("\n[bold blue]🔧 Initializing Memory System...[/bold blue]")

LLM_BASE_URL = os.getenv("LLAMA_CPP_LLM_URL", "http://localhost:8080/v1")
LLM_MODEL = os.getenv("LLAMA_CPP_LLM_MODEL", "llama-3-8b")
EMBED_MODEL = os.getenv("LLAMA_CPP_EMBED_MODEL", "nomic-embed-text")
EMBED_BASE_URL = os.getenv("LLAMA_CPP_EMBED_URL", "http://localhost:8080/v1")
EMBED_DIMS = int(os.getenv("LLAMA_CPP_EMBED_DIMS", "768"))
USE_INFERENCE = True
USE_STREAMING = True


config_dict = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": LLM_MODEL,
            "api_key": "not-needed",
            "openai_base_url": LLM_BASE_URL,
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": EMBED_MODEL,
            "api_key": "not-needed",
            "openai_base_url": EMBED_BASE_URL,
            "embedding_dims": EMBED_DIMS,
        },
    },
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "collection_name": "my_memories",
            "embedding_model_dims": EMBED_DIMS,
            "dbname": DEFAULT_DB,
            "user": DEFAULT_USER,
            "password": DEFAULT_PASSWORD,
            "host": DEFAULT_HOST,
            "port": DEFAULT_PORT,
        },
    },
}

collection_name = config_dict["vector_store"]["config"]["collection_name"]
drop_table_if_exists(f"public.{collection_name}_entities")
drop_type_if_exists(f"public.{collection_name}_entities")

with console.status("[bold cyan]Connecting to LLM & Embedding services..."):
    config = MemoryConfig(**config_dict)
    base_memory = Memory(config)

# NEW: Create LLMContext for streaming memory via llm_utils
streaming_ctx = LLMContext.from_params(
    model=LLM_MODEL,
    base_url=LLM_BASE_URL,
    verbose=False,  # We handle display ourselves via StreamingDisplay
)
console.print("[dim]LLMContext initialized for streaming[/dim]")


class ProgressTracker:
    def __init__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            expand=False,
        )

    def start(self, description: str, total: int = 100) -> int:
        return self.progress.add_task(f"[cyan]{description}", total=total, start=True)

    def update(self, task_id: int, advance: int = 1, description: str = None):
        kwargs = {"advance": advance}
        if description:
            kwargs["description"] = f"[cyan]{description}"
        self.progress.update(task_id, **kwargs)

    def complete(self, task_id: int, message: str = "Done"):
        self.progress.update(
            task_id,
            description=f"[green]✓ {message}",
            completed=self.progress.tasks[task_id].total,
            refresh=True,
        )


@contextmanager
def track_operation(description: str, steps: list[str] = None):
    """Track operation with optional sub-steps progress bar."""
    console.print(f"\n[bold yellow]▶ {description}[/bold yellow]")
    start_time = time.perf_counter()
    if steps:
        tracker = ProgressTracker()
        task_id = tracker.start(description, total=len(steps))
        yield tracker, task_id
        tracker.complete(task_id, description)
    else:
        with console.status(f"[bold cyan]{description}..."):
            yield None, None
    elapsed = time.perf_counter() - start_time
    console.print(f"[dim]⏱  Completed in {elapsed:.2f}s[/dim]")


def print_section(title: str):
    console.print()
    console.rule(f"[bold blue]{title}")


def display_results(results: dict, title: str = "Results", max_items: int = 10):
    if not results.get("results"):
        console.print("[dim]  (no results)[/dim]")
        return
    table = Table(
        title=f"[bold green]{title} ({len(results['results'])} items)",
        show_header=True,
        header_style="bold magenta",
        border_style="blue",
    )
    table.add_column("Event", style="cyan", width=6)
    table.add_column("ID", style="dim", width=12)
    table.add_column("Memory", style="white")
    table.add_column("Score", style="yellow", width=8)
    for item in results["results"][:max_items]:
        event_map = {
            "ADD": "[green]ADD[/green]",
            "UPDATE": "[yellow]UPDATE[/yellow]",
            "DELETE": "[red]DELETE[/red]",
        }
        event = event_map.get(item.get("event", ""), item.get("event", "N/A"))
        mem_id = item.get("id", "N/A")[:8] + "..."
        memory = item.get("memory", "N/A")
        if len(memory) > 100:
            memory = memory[:97] + "..."
        score = f"{item['score']:.3f}" if item.get("score") is not None else "-"
        table.add_row(event, mem_id, memory, score)
    console.print(table)


class StreamingDisplay:
    """Handles real-time streaming output display with Rich."""

    def __init__(self, title: str = "Streaming Output"):
        self.title = title
        self.chunks = []
        self._live = None
        self._panel = None

    def __enter__(self):
        self._panel = Panel(
            "",
            title=f"[bold cyan]🔴 {self.title}",
            border_style="cyan",
            padding=(1, 2),
        )
        self._live = Live(
            self._panel, console=console, refresh_per_second=10, transient=False
        )
        self._live.start()
        return self

    def __exit__(self, *args):
        if self._live:
            text = "".join(self.chunks)
            display_text = text[-800:] if len(text) > 800 else text
            if len(text) > 800:
                display_text = f"...(earlier output)...\n{display_text}"
            self._panel.renderable = Text(display_text, style="white")
            self._panel.title = f"[bold green]✅ {self.title} (complete)"
            self._panel.border_style = "green"
            self._live.update(self._panel)
            self._live.refresh()
            time.sleep(0.3)
            self._live.stop()

    def add_chunk(self, chunk: str):
        """Add a chunk to the streaming display."""
        self.chunks.append(chunk)
        text = "".join(self.chunks)
        display_text = text[-500:] if len(text) > 500 else text
        if len(text) > 500:
            display_text = f"...(earlier output)...\n{display_text}"
        self._panel.renderable = Text(display_text, style="white")
        self._live.update(self._panel)

    def get_full_text(self) -> str:
        return "".join(self.chunks)


class StreamingMemory:
    """Wrapper around Memory that adds streaming visibility using llm_utils."""

    def __init__(self, memory: Memory, ctx: LLMContext):
        self.memory = memory
        self.ctx = ctx
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            "StreamingMemory initialized | model=%s | base_url=%s",
            ctx.model,
            ctx.sync_client.base_url,
        )

    def _parse_llm_output(self, raw_text: str) -> list[dict]:
        """
        Robust parser for various LLM output formats.
        Handles:
        - {"memory": [{"text": "..."}]}
        - [{"text": "..."}]
        - ```json ... ```
        - Plain JSON array
        """
        text = raw_text.strip()
        if "```" in text:
            match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()
        strategies = [
            lambda t: json.loads(t),
            lambda t: json.loads(re.sub(r",\s*}", "}", re.sub(r",\s*]", "]", t))),
        ]
        parsed = None
        for strategy in strategies:
            try:
                parsed = strategy(text)
                break
            except json.JSONDecodeError:
                continue
        if parsed is None:
            console.print("[red]⚠ Failed to parse LLM output[/red]")
            console.print(f"[dim]Raw: {text[:300]}...[/dim]")
            return []
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            if "memory" in parsed:
                return parsed["memory"]
            if "facts" in parsed:
                return parsed["facts"]
            if "text" in parsed:
                return [parsed]
        console.print(f"[yellow]⚠ Unexpected JSON structure: {type(parsed)}[/yellow]")
        return []

    def add_with_streaming(
        self,
        messages,
        user_id: str = None,
        agent_id: str = None,
        run_id: str = None,
        infer: bool = True,
    ) -> dict:
        """
        Add memories with real-time streaming of LLM output via llm_utils.chat.
        """
        if not infer:
            return self.memory.add(
                messages,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                infer=False,
            )

        console.print(
            "\n[bold magenta]🤖 Starting LLM Memory Extraction...[/bold magenta]"
        )
        console.print("[dim]Streaming chunks below:[/dim]\n")

        with StreamingDisplay("LLM Memory Extraction") as display:
            extraction_result = self._stream_extraction(messages, display)

        console.print("\n[bold cyan]🔍 Parsing extracted memories...[/bold cyan]")
        extracted_memories = self._parse_llm_output(extraction_result)

        if not extracted_memories:
            console.print(
                "[yellow]⚠ No memories extracted. Storing raw input instead.[/yellow]"
            )
            return self.memory.add(
                messages,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                infer=False,
            )

        valid_memories = []
        for mem in extracted_memories:
            if isinstance(mem, dict) and "text" in mem and mem["text"].strip():
                valid_memories.append(mem)
            elif isinstance(mem, str) and mem.strip():
                valid_memories.append({"text": mem.strip()})

        console.print(f"[green]✓ Extracted {len(valid_memories)} memories[/green]")
        results = []
        for i, mem in enumerate(valid_memories):
            text = mem["text"]
            with console.status(
                f"[cyan]Embedding {i + 1}/{len(valid_memories)}: {text[:60]}..."
            ):
                result = self.memory.add(
                    text,
                    user_id=user_id,
                    agent_id=agent_id,
                    run_id=run_id,
                    infer=False,
                )
            if result.get("results"):
                results.extend(result["results"])
                mem_id = result["results"][0]["id"][:8]
                console.print(
                    f"  [green]✓ [{mem_id}...][/green] [white]{text[:80]}[/white]"
                )
        return {"results": results}

    def _stream_extraction(self, messages, display: StreamingDisplay) -> str:
        """
        Stream LLM extraction response chunk-by-chunk using llm_utils.chat.
        """
        system_prompt = """You are a memory extraction system. Extract key facts from conversations.
Output ONLY a JSON object with this exact structure:
{"memory": [{"text": "fact 1"}, {"text": "fact 2"}]}
Rules:
- Extract specific, standalone facts
- Each fact must be a complete sentence
- Output ONLY the JSON, no markdown, no explanations
- Do NOT wrap in ```json code blocks"""

        if isinstance(messages, list):
            conversation = "\n".join(
                f"{msg['role'].upper()}: {msg['content']}" for msg in messages
            )
        else:
            conversation = str(messages)

        user_content = f"Extract key facts from:\n\n{conversation}"

        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        full_response_chunks: list[str] = []
        chunk_count = 0

        self.logger.info("Starting streaming extraction via llm_utils.chat")
        try:
            # Use llm_utils.chat with stream=True
            stream = chat(
                messages=chat_messages,
                temperature=0.1,
                max_tokens=500,
                stream=True,
                enable_thinking=False,
                ctx=self.ctx,
            )
            for chunk in stream:
                full_response_chunks.append(chunk)
                display.add_chunk(chunk)
                chunk_count += 1

            console.print(f"\n[dim]Streamed {chunk_count} chunks total[/dim]")
            self.logger.info("Streaming extraction complete | chunks=%d", chunk_count)
        except Exception as e:
            self.logger.error("Streaming extraction failed | error=%s", e)
            console.print(f"\n[red]✗ Streaming error: {e}[/red]")
            raise

        return "".join(full_response_chunks)

    def get(self, memory_id):
        return self.memory.get(memory_id)

    def get_all(self, **kwargs):
        return self.memory.get_all(**kwargs)

    def search(self, query, **kwargs):
        return self.memory.search(query, **kwargs)

    def update(self, memory_id, **kwargs):
        return self.memory.update(memory_id, **kwargs)

    def delete(self, memory_id):
        return self.memory.delete(memory_id)

    def history(self, memory_id):
        return self.memory.history(memory_id)

    def reset(self):
        self.memory.reset()

    def close(self):
        self.memory.close()


# UPDATED: Pass LLMContext instead of raw URL/model strings
streaming_mem = StreamingMemory(base_memory, streaming_ctx)
console.print("[green]✓ StreamingMemory wrapper ready[/green]")

if USE_INFERENCE and USE_STREAMING:
    mode_text = (
        "[bold green]STREAMING MODE[/bold green] - LLM output shown in real-time"
    )
elif USE_INFERENCE:
    mode_text = (
        "[bold yellow]INFERENCE MODE[/bold yellow] - LLM extraction (no streaming)"
    )
else:
    mode_text = "[bold cyan]FAST MODE[/bold cyan] - No LLM, instant storage"
console.print(f"\n{mode_text}")
console.print(f"[dim]LLM: {LLM_MODEL} | Embedding: {EMBED_MODEL}[/dim]")

print_section("0. WARMUP")
with track_operation("Warming up services"):
    try:
        _ = base_memory.embedding_model.embed("warmup", "add")
        console.print("[green]✓ Embedding service ready[/green]")
    except Exception as e:
        console.print(f"[red]✗ Embedding error: {e}[/red]")

USER_ID = "demo_user"

print_section("1. ADDING MEMORIES WITH STREAMING")
console.print("[bold]📝 Input conversation:[/bold]")
console.print(
    Panel(
        "[dim]User: I love visiting Japan. The food and culture are amazing.[/dim]\n"
        "[dim]Assistant: Japan is wonderful! What's your favorite dish?[/dim]\n"
        "[dim]User: Definitely sushi and ramen. I also enjoy hiking near Mt. Fuji.[/dim]",
        title="Conversation",
        border_style="blue",
    )
)

conversation = [
    {
        "role": "user",
        "content": "I love visiting Japan. The food and culture are amazing.",
    },
    {"role": "assistant", "content": "Japan is wonderful! What's your favorite dish?"},
    {
        "role": "user",
        "content": "Definitely sushi and ramen. I also enjoy hiking near Mt. Fuji.",
    },
]

add_result = streaming_mem.add_with_streaming(
    conversation,
    user_id=USER_ID,
    infer=USE_INFERENCE,
)
display_results(add_result, "Added Memories")

print_section("2. ADDING WORK INFO (FAST)")
with track_operation("Storing work information"):
    streaming_mem.memory.add(
        "I'm a software engineer working remotely, using Python daily.",
        user_id=USER_ID,
        infer=False,
    )
console.print("[green]✓ Work info stored[/green]")

print_section("3. GET ALL MEMORIES")
with track_operation("Fetching all memories"):
    all_memories = streaming_mem.get_all(filters={"user_id": USER_ID})
display_results(all_memories, f"All Memories for {USER_ID}")

print_section("4. SEARCHING MEMORIES")
search_queries = [
    ("Japan food travel", "Travel-related"),
    ("programming Python engineer", "Work-related"),
]
for query, label in search_queries:
    with track_operation(f'Searching: "{query}"'):
        search_result = streaming_mem.search(
            query, filters={"user_id": USER_ID}, top_k=5
        )
    display_results(search_result, label)

print_section("5. GET SINGLE MEMORY")
if all_memories["results"]:
    first_id = all_memories["results"][0]["id"]
    with track_operation(f"Retrieving {first_id[:8]}..."):
        single = streaming_mem.get(first_id)
    if single:
        console.print(
            Panel(
                f"[white]{single['memory']}[/white]\n\n"
                f"[dim]Created: {single['created_at']}[/dim]",
                title=f"[bold cyan]Memory: {first_id[:12]}...",
                border_style="blue",
            )
        )

print_section("6. UPDATING A MEMORY")
if all_memories["results"]:
    target = all_memories["results"][0]
    console.print(f"  [yellow]Before:[/yellow] {target['memory'][:80]}")
    with track_operation("Updating memory"):
        streaming_mem.update(
            target["id"],
            data=f"{target['memory']} I'm also planning a trip to Kyoto.",
        )
    updated = streaming_mem.get(target["id"])
    if updated:
        console.print(f"  [green]After:[/green]  {updated['memory'][:80]}")

print_section("7. MEMORY HISTORY")
if all_memories["results"]:
    with track_operation("Fetching history"):
        history = streaming_mem.history(all_memories["results"][0]["id"])
    history_tree = Tree(
        f"[bold cyan]History: {all_memories['results'][0]['id'][:8]}..."
    )
    for entry in history:
        event_style = {"ADD": "[green]", "UPDATE": "[yellow]", "DELETE": "[red]"}.get(
            entry["event"], "[white]"
        )
        node = history_tree.add(
            f"{event_style}[{entry['event']}] {entry.get('new_memory', 'N/A')[:80]}"
        )
        if entry.get("created_at"):
            node.add(f"[dim]{entry['created_at']}[/dim]")
    console.print(history_tree)

print_section("8. DELETE A MEMORY")
if len(all_memories["results"]) > 1:
    to_delete = all_memories["results"][1]["id"]
    with track_operation(f"Deleting {to_delete[:8]}..."):
        streaming_mem.delete(to_delete)
    remaining = streaming_mem.get_all(filters={"user_id": USER_ID})
    console.print(f"  [green]✓ Remaining: {len(remaining['results'])}[/green]")

print_section("SUMMARY")
final_memories = streaming_mem.get_all(filters={"user_id": USER_ID})
stats_table = Table(title="[bold]Final Stats", show_header=False)
stats_table.add_column("Metric", style="cyan")
stats_table.add_column("Value", style="white")
stats_table.add_row("Total Memories", str(len(final_memories["results"])))
stats_table.add_row("User ID", USER_ID)
mode = (
    "Streaming LLM"
    if (USE_INFERENCE and USE_STREAMING)
    else ("LLM" if USE_INFERENCE else "Fast")
)
stats_table.add_row("Mode", mode)
console.print(stats_table)

print_section("CLEANUP")
with track_operation("Resetting memory store"):
    streaming_mem.reset()
console.print("[green]✓ All memories cleared[/green]")
streaming_mem.close()
console.print()
console.rule("[bold green]✨ DEMO COMPLETE")
console.print()
