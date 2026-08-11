import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.tasks.evaluate_relevance import evaluate_relevance
from rich.console import Console
from rich.table import Table

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_5/top_isekai_anime_2026/contexts.json"

with open(docs_file, "r", encoding="utf-8") as f:
    docs_json = json.load(f)

query = docs_json["query"]
documents = [f"{doc['header']}\n{doc['content']}" for doc in docs_json["results"]]

task_description = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

# ─── Save Config ───
config = {
    "timestamp": datetime.now().isoformat(),
    "source_file": str(docs_file),
    "embed_model": EMBED_MODEL,
    "task_description": task_description,
    "query": query,
    "num_documents": len(documents),
    "document_previews": [
        d[:100] + "..." if len(d) > 100 else d for d in documents[:5]
    ],
}
config_path = OUTPUT_DIR / "config.json"
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
console.print(f"[dim]Config saved to {config_path}[/dim]")

# ─── Save Inputs ───
inputs_path = OUTPUT_DIR / "inputs.json"
with open(inputs_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "query": query,
            "task_description": task_description,
            "documents": [
                {"index": i, "document": d, "char_length": len(d)}
                for i, d in enumerate(documents)
            ],
        },
        f,
        indent=2,
        ensure_ascii=False,
    )
console.print(f"[dim]Inputs saved to {inputs_path}[/dim]")

# ─── Run Evaluation ───
scores = evaluate_relevance(
    query,
    documents,
    task_description,
    show_progress=True,
)

# Flatten: scores[0] is the list of scores for the single query
score_list = scores[0] if scores else []

# Build results list
results = []
for i, (doc, score) in enumerate(zip(documents, score_list)):
    if score >= 0.7:
        relevance = "high"
    elif score >= 0.4:
        relevance = "medium"
    else:
        relevance = "low"
    results.append(
        {
            "rank": 0,  # filled after sorting
            "document": doc,
            "score": round(score, 6),
            "relevance": relevance,
            "index": i,
        }
    )

# Sort by score descending and assign ranks
results.sort(key=lambda x: x["score"], reverse=True)
for rank, r in enumerate(results, start=1):
    r["rank"] = rank

# ─── Save Results JSON ───
json_path = OUTPUT_DIR / "results.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "query": query,
            "task_description": task_description,
            "documents_evaluated": len(documents),
            "results": results,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )
console.print(f"[dim]Results saved to {json_path}[/dim]")

# ─── Save Results CSV ───
csv_path = OUTPUT_DIR / "results.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["rank", "document", "score", "relevance", "original_index"])
    for r in results:
        writer.writerow(
            [
                r["rank"],
                r["document"][:120],
                f"{r['score']:.6f}",
                r["relevance"],
                r["index"],
            ]
        )
console.print(f"[dim]CSV saved to {csv_path}[/dim]")

# ─── Save Summary ───
scores_only = [r["score"] for r in results]
summary = {
    "total_documents": len(results),
    "score_range": {
        "min": round(min(scores_only), 6) if scores_only else 0,
        "max": round(max(scores_only), 6) if scores_only else 0,
        "avg": round(sum(scores_only) / len(scores_only), 6) if scores_only else 0,
    },
    "relevance_distribution": {
        "high": sum(1 for r in results if r["relevance"] == "high"),
        "medium": sum(1 for r in results if r["relevance"] == "medium"),
        "low": sum(1 for r in results if r["relevance"] == "low"),
    },
    "top_document": results[0]["document"][:200] if results else "",
    "top_score": results[0]["score"] if results else 0,
    "top_relevance": results[0]["relevance"] if results else "",
}
summary_path = OUTPUT_DIR / "summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
console.print(f"[dim]Summary saved to {summary_path}[/dim]")

# ─── Console Table ───
console.print("\n[bold green]Relevance Evaluation Results[/bold green]")
table = Table(show_header=True, header_style="bold magenta", show_lines=True)
table.add_column("Rank", justify="center", style="dim", width=4)
table.add_column("Document", style="white", no_wrap=False, max_width=60)
table.add_column("Score", justify="right", width=10)
table.add_column("Relevance", justify="center", width=10)

relevance_styles = {
    "high": "bold green",
    "medium": "yellow",
    "low": "dim red",
}

for r in results[:10]:
    r_style = relevance_styles.get(r["relevance"], "dim")
    table.add_row(
        str(r["rank"]),
        r["document"][:100] + ("..." if len(r["document"]) > 100 else ""),
        f"[{r_style}]{r['score']:.4f}[/{r_style}]",
        f"[{r_style}]{r['relevance']}[/{r_style}]",
    )

console.print(table)

# ─── Resource Links ───
console.print("\n[bold cyan]Saved Files:[/bold cyan]")
console.print(f"  ⚙️  [link=file://{config_path}]{config_path.name}[/link]")
console.print(f"  📥 [link=file://{inputs_path}]{inputs_path.name}[/link]")
console.print(f"  📄 [link=file://{json_path}]{json_path.name}[/link]")
console.print(f"  📊 [link=file://{csv_path}]{csv_path.name}[/link]")
console.print(f"  📈 [link=file://{summary_path}]{summary_path.name}[/link]")
