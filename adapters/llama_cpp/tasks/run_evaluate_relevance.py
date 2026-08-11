from jet.adapters.llama_cpp.tasks.evaluate_relevance import evaluate_relevance
from rich.console import Console
from rich.table import Table

console = Console()

task = "Given a web search query, retrieve relevant passages that answer the query"
test_queries = [
    "What is the capital of China?",
    "Explain gravity",
]
test_documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. "
    "It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

result_scores = evaluate_relevance(
    test_queries,
    test_documents,
    task,
    show_progress=True,
)

# Display results sorted by score (descending) per query
console.print("\n[bold green]Relevance Evaluation Results[/bold green]")
table = Table(show_header=True, header_style="bold magenta", show_lines=True)
table.add_column("Rank", justify="center", style="dim", width=4)
table.add_column("Query", style="cyan", no_wrap=False, max_width=40)
table.add_column("Document", style="white", no_wrap=False, max_width=60)
table.add_column("Score", justify="right", style="yellow")

for qi, query in enumerate(test_queries):
    # Pair documents with scores and sort descending
    scored_docs = list(zip(result_scores[qi], test_documents))
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    for rank, (score, doc) in enumerate(scored_docs, start=1):
        # Color-code scores for quick visual assessment
        if score >= 0.7:
            score_style = "bold green"
        elif score >= 0.4:
            score_style = "yellow"
        else:
            score_style = "dim red"

        table.add_row(
            str(rank),
            query if rank == 1 else "",  # Only show query on first row
            doc[:100] + ("..." if len(doc) > 100 else ""),
            f"[{score_style}]{score:.4f}[/{score_style}]",
        )

    # Add a separator between query groups for readability
    if qi < len(test_queries) - 1:
        table.add_row("", "", "", "")

console.print(table)
