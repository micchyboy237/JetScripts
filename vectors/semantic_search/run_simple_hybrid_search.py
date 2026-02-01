import os
import shutil

from jet.adapters.llama_cpp.hybrid_search import HybridSearch
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS
from jet.file.utils import save_file
from jet.libs.bertopic.examples.mock import load_sample_data_with_info
from rich.console import Console
from rich.table import Table

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "generated",
    os.path.splitext(os.path.basename(__file__))[0],
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

console = Console()

if __name__ == "__main__":
    model: LLAMACPP_EMBED_KEYS = "nomic-embed-text"

    query = "Top isekai anime 2025"
    chunks_with_info = load_sample_data_with_info(model=model, includes=["p"])
    save_file(chunks_with_info, f"{OUTPUT_DIR}/chunks_with_info.json")

    documents = [chunk["content"] for chunk in chunks_with_info]
    token_counts = [chunk["num_tokens"] for chunk in chunks_with_info]
    console.print(f"Total tokens: {sum(token_counts)}")

    save_file(
        {
            "count": len(documents),
            "tokens": {
                "max": max(token_counts),
                "min": min(token_counts),
                "total": sum(token_counts),
            },
            "results": [
                {"doc_index": idx, "tokens": tokens, "text": text}
                for idx, (tokens, text) in enumerate(zip(token_counts, documents))
            ],
        },
        f"{OUTPUT_DIR}/documents.json",
    )

    hybrid = HybridSearch.from_documents(
        documents=documents,
        model=model,
    )

    results = hybrid.search(query, top_k=10)

    table = Table(title=f"Hybrid Results for: {query!r}")
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Hybrid", justify="right")
    table.add_column("Dense", justify="right")
    table.add_column("Sparse", justify="right")
    table.add_column("Category", style="bold")
    table.add_column("Level", justify="right", style="dim cyan")
    table.add_column("ID")
    table.add_column("Preview", style="dim")

    for res in results:
        preview = res["text"][:80] + "..." if len(res["text"]) > 80 else res["text"]
        table.add_row(
            str(res["rank"]),
            f"{res['hybrid_score']:.4f}",
            f"{res['dense_score']:.3f}",
            f"{res['sparse_score']:.3f}",
            res["category"],
            str(res["category_level"]),
            res["id"] or "-",
            preview,
        )

    console.print(table)
