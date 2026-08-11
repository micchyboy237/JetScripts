from jet.adapters.llama_cpp.tasks.evaluate_multiple_contexts_relevance import (
    evaluate_multiple_contexts_relevance,
)
from rich.console import Console
from rich.table import Table

console = Console()

query = "What is the capital of France?"
contexts = [
    "The capital of France is Paris.",
    "Paris is a popular tourist destination.",
    "Einstein developed the theory of relativity.",
]

results = evaluate_multiple_contexts_relevance(query, contexts, verbose=True)

console.print("\n[bold green]Multiple Contexts Relevance Results[/bold green]")
table = Table(show_header=True, header_style="bold magenta", show_lines=True)
table.add_column("#", justify="center", style="dim", width=3)
table.add_column("Context", style="white", no_wrap=False, max_width=55)
table.add_column("Score", justify="center", width=8)
table.add_column("Confidence", justify="right", width=10)
table.add_column("P(0)", justify="right", style="dim red", width=7)
table.add_column("P(1)", justify="right", style="yellow", width=7)
table.add_column("P(2)", justify="right", style="bold green", width=7)
table.add_column("Priority", justify="center", width=9)

score_styles = {0: "dim red", 1: "yellow", 2: "bold green"}
priority_styles = {"low": "dim red", "medium": "yellow", "high": "bold green"}

for rank, r in enumerate(results, start=1):
    s_style = score_styles.get(r["relevance_score"], "dim")
    p_style = priority_styles.get(r["priority"], "dim")
    probs = r["probabilities"]

    table.add_row(
        str(rank),
        r["context"][:80] + ("..." if len(r["context"]) > 80 else ""),
        f"[{s_style}]{r['relevance_score']}[/{s_style}]",
        f"{r['score']:.4f}",
        f"{probs[0]:.3f}" if len(probs) > 0 else "-",
        f"{probs[1]:.3f}" if len(probs) > 1 else "-",
        f"{probs[2]:.3f}" if len(probs) > 2 else "-",
        f"[{p_style}]{r['priority']}[/{p_style}]",
    )

console.print(table)
