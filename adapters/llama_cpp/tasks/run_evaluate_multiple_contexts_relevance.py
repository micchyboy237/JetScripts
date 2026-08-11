# JetScripts/adapters/llama_cpp/tasks/run_evaluate_multiple_contexts_relevance.py
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.tasks.evaluate_multiple_contexts_relevance import (
    evaluate_multiple_contexts_relevance,
)
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

# --- Save config ---
config = {
    "timestamp": datetime.now().isoformat(),
    "source_file": str(docs_file),
    "llm_model": LLM_MODEL,
    "evaluation_method": "grammar_constrained_llm",
    "evaluation_type": "answer_containment",
    "scoring_scale": {"true": "contains answer", "false": "no answer"},
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

# --- Save inputs ---
inputs_path = OUTPUT_DIR / "inputs.json"
with open(inputs_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "query": query,
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

# --- Run evaluation ---
console.print(
    "\n[bold cyan]Running multi-context answer containment evaluation...[/bold cyan]"
)
eval_results = evaluate_multiple_contexts_relevance(query, documents)

# --- Transform & rank results ---
results = []
for r in eval_results:
    has_answer = r["has_answer"]
    label = "contains answer" if has_answer else "no answer"
    results.append(
        {
            "rank": 0,
            "document": documents[r["context_index"]],
            "has_answer": has_answer,
            "label": label,
            "is_valid": r["is_valid"],
            "error": r["error"],
            "original_index": r["context_index"],
        }
    )

# Sort: valid first, then has_answer=True first, then original index for stability
results.sort(key=lambda x: (-x["is_valid"], -x["has_answer"], x["original_index"]))
for rank, r in enumerate(results, start=1):
    r["rank"] = rank

# --- Save results JSON ---
json_path = OUTPUT_DIR / "results.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "query": query,
            "evaluation_method": "grammar_constrained_llm",
            "evaluation_type": "answer_containment",
            "documents_evaluated": len(documents),
            "results": results,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )
console.print(f"[dim]Results saved to {json_path}[/dim]")

# --- Save results CSV ---
csv_path = OUTPUT_DIR / "results.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["rank", "document", "has_answer", "label", "is_valid", "original_index"]
    )
    for r in results:
        writer.writerow(
            [
                r["rank"],
                r["document"][:120],
                r["has_answer"],
                r["label"],
                r["is_valid"],
                r["original_index"],
            ]
        )
console.print(f"[dim]CSV saved to {csv_path}[/dim]")

# --- Save summary ---
valid_results = [r for r in results if r["is_valid"]]
summary = {
    "total_documents": len(results),
    "valid_evaluations": len(valid_results),
    "containment_distribution": {
        "contains_answer": sum(1 for r in valid_results if r["has_answer"]),
        "no_answer": sum(1 for r in valid_results if not r["has_answer"]),
    },
    "containment_rate": round(
        sum(1 for r in valid_results if r["has_answer"]) / len(valid_results), 4
    )
    if valid_results
    else 0,
    "top_document": results[0]["document"][:200] if results else "",
    "top_has_answer": results[0]["has_answer"] if results else False,
    "top_label": results[0]["label"] if results else "",
}
summary_path = OUTPUT_DIR / "summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
console.print(f"[dim]Summary saved to {summary_path}[/dim]")

# --- Display table ---
console.print("\n[bold green]Multi-Context Answer Containment Results[/bold green]")
table = Table(show_header=True, header_style="bold magenta", show_lines=True)
table.add_column("Rank", justify="center", style="dim", width=4)
table.add_column("Document", style="white", no_wrap=False, max_width=60)
table.add_column("Has Answer", justify="center", width=12)
table.add_column("Valid", justify="center", width=6)

for r in results[:15]:
    if r["has_answer"]:
        ans_str = "[bold green]✓ Yes[/bold green]"
    else:
        ans_str = "[dim red]✗ No[/dim red]"
    v_str = "[bold green]✓[/bold green]" if r["is_valid"] else "[bold red]✗[/bold red]"
    doc_preview = r["document"][:100] + ("..." if len(r["document"]) > 100 else "")
    table.add_row(
        str(r["rank"]),
        doc_preview,
        ans_str,
        v_str,
    )

console.print(table)

console.print("\n[bold cyan]Saved Files:[/bold cyan]")
console.print(f"  ⚙️  [link=file://{config_path}]{config_path.name}[/link]")
console.print(f"  📥 [link=file://{inputs_path}]{inputs_path.name}[/link]")
console.print(f"  📄 [link=file://{json_path}]{json_path.name}[/link]")
console.print(f"  📊 [link=file://{csv_path}]{csv_path.name}[/link]")
console.print(f"  📈 [link=file://{summary_path}]{summary_path.name}[/link]")
