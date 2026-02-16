import json
from pathlib import Path
from typing import Any, TypedDict

from jet.search.heuristics.generic_search import (
    GenericSearchEngine,
    SearchResult,
)
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table

# -----------------------------------------
# Output directory
# -----------------------------------------
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

console = Console()


# @dataclass(slots=True)
# class Article:
#     id: int
#     title: str
#     content: str
class Article(TypedDict):
    id: int
    title: str
    content: str


def serialize_results(
    results: list[SearchResult[Article]],
) -> list[dict[str, Any]]:
    return [
        {
            # "item": asdict(result.item),
            "item": result.item,
            "score": result.score,
            "matched_fields": result.matched_fields,
            "matched_terms": result.matched_terms,
            "highlights": result.highlights,
        }
        for result in results
    ]


def save_results(filename: str, data: list[dict[str, Any]]) -> None:
    path = OUTPUT_DIR / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    console.print(f"[green]Saved:[/green] {path}")


def render_results(
    title: str,
    results: list[SearchResult[Article]],
) -> None:
    console.rule(f"[bold cyan]{title}[/bold cyan]")

    if not results:
        console.print("[bold red]No results found.[/bold red]")
        return

    table = Table(
        title="Search Results",
        box=box.ROUNDED,
        show_lines=True,
    )

    table.add_column("Rank", justify="right")
    table.add_column("ID", justify="right")
    table.add_column("Title", style="bold magenta")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Matched Fields", style="yellow")

    for idx, result in enumerate(results, start=1):
        table.add_row(
            str(idx),
            str(result.item["id"]),
            result.item["title"],
            f"{result.score:.4f}",
            ", ".join(result.matched_fields),
        )

    console.print(table)

    console.rule("[bold cyan]Highlights[/bold cyan]")

    for idx, result in enumerate(results, start=1):
        highlight_text = "\n\n".join(
            f"[bold]{field}[/bold]\n{snippet}"
            for field, snippet in result.highlights.items()
        )

        console.print(
            Panel(
                highlight_text,
                title=f"Result #{idx} — ID {result.item['id']}",
                border_style="cyan",
            )
        )

        console.print(
            Panel(
                Pretty(result.matched_terms),
                title="Matched Terms",
                border_style="yellow",
            )
        )


def main() -> None:
    items: list[Article] = [
        Article(
            id=1,
            title="Python search engine",
            content="Building a generic search engine using BM25 scoring and ranking algorithms.",
        ),
        Article(
            id=2,
            title="FastAPI tutorial",
            content="Learn how to build APIs using FastAPI and Python with production-ready patterns.",
        ),
        Article(
            id=3,
            title="Search algorithms overview",
            content="An introduction to search algorithms, heuristics, and ranking strategies.",
        ),
        Article(
            id=4,
            title="Distributed search systems",
            content="Scaling search engines using distributed indexing and sharding techniques.",
        ),
        Article(
            id=5,
            title="Introduction to databases",
            content="Understanding relational databases, indexing, and query optimization.",
        ),
        Article(
            id=6,
            title="Machine learning ranking",
            content="Using machine learning models to improve search relevance and scoring.",
        ),
        Article(
            id=7,
            title="Rust systems programming",
            content="Memory safety, performance, and concurrency in Rust applications.",
        ),
        Article(
            id=8,
            title="Full-text search with PostgreSQL",
            content="Implementing full-text search, indexing, and ranking inside PostgreSQL.",
        ),
    ]

    def extract_text(article: Article) -> dict[str, str]:
        return {
            "title": article["title"],
            "content": article["content"],
        }

    engine = GenericSearchEngine[Article](
        items=items,
        # text_extractor=extract_text,
        field_weights={"title": 2.0, "content": 1.0},
    )

    console.rule("[bold green]Generic Search Demo[/bold green]")

    # -----------------------------
    # AND Example
    # -----------------------------
    and_query = "search engine"
    console.print(f"\n[bold]AND Query:[/bold] '{and_query}'\n")

    and_results = engine.search(
        query=and_query,
        logic="AND",
        limit=10,
    )

    render_results("AND Logic Results", and_results)

    save_results(
        "and_results.json",
        serialize_results(and_results),
    )

    # -----------------------------
    # OR Example
    # -----------------------------
    or_query = "search database"
    console.print(f"\n[bold]OR Query:[/bold] '{or_query}'\n")

    or_results = engine.search(
        query=or_query,
        logic="OR",
        limit=10,
    )

    render_results("OR Logic Results", or_results)

    save_results(
        "or_results.json",
        serialize_results(or_results),
    )


if __name__ == "__main__":
    main()
