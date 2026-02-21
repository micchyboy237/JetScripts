# File: embed_client_demo.py
from __future__ import annotations

import time

from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from rich.console import Console

console = Console()
MAX_WORKERS = 6
embedder = LlamacppEmbedding(max_workers=MAX_WORKERS)


def timed_batch_embed(
    texts: list[str],
    batch_size: int,
) -> tuple[list[list[float]], float]:
    """Run batch embed and return results + wall-clock time in seconds."""
    start = time.perf_counter()
    embeddings = embedder.embed_parallel(texts, batch_size=batch_size)
    elapsed = time.perf_counter() - start
    return embeddings, elapsed


# === Real-world example data ===
# Simulate 16 support articles / product FAQs (short-medium length)
ORIGINAL_DOCS = [
    "Our return policy allows returns within 30 days of purchase with original packaging.",
    "To reset your password, click 'Forgot Password' on the login page and follow the email instructions.",
    "The Premium plan includes unlimited storage, priority support, and advanced analytics.",
    "We support payments via credit card, PayPal, and bank transfer in most countries.",
    "If your order is delayed, please check tracking or contact support with your order ID.",
    "All devices come with a 1-year warranty covering manufacturing defects.",
    "To cancel subscription, go to Account Settings > Billing > Cancel Subscription.",
    "Our app is available on iOS 15+ and Android 10+ devices.",
    "For bulk orders over 100 units, please contact sales@company.com for custom pricing.",
    "Data is encrypted in transit (TLS 1.3) and at rest (AES-256).",
    "You can export your data anytime from Settings > Privacy > Export Data.",
    "Troubleshooting steps for login issues: clear cache, try incognito, check credentials.",
    "We offer free shipping on orders over $50 in the continental US.",
    "Product X is not compatible with older OS versions prior to 2022.",
    "To request a refund, submit a ticket with proof of purchase and reason.",
    "Our team responds to support tickets within 24 hours on business days.",
]

# Create larger dataset by repeating the original documents
SAMPLE_DOCS = ORIGINAL_DOCS * 20  # 16 × 20 = 320 documents


if __name__ == "__main__":
    console.rule("llama.cpp Embedding Server Performance Demo")
    console.print(
        "\n[bold cyan]Scenario:[/bold cyan] Indexing customer support articles "
        "for semantic search in a helpdesk RAG system.\n"
        "Goal: Measure embedding throughput at different dataset sizes.\n"
    )

    console.print(f"[green]Dataset size:[/green] {len(SAMPLE_DOCS)} documents")
    console.print(
        f"[green]Using concurrent version[/green] (max_workers={MAX_WORKERS}) – "
        f"leverages --parallel {MAX_WORKERS} + continuous batching\n"
    )

    embeddings, time_taken = timed_batch_embed(SAMPLE_DOCS, batch_size=2)
    throughput = len(SAMPLE_DOCS) / time_taken if time_taken > 0 else 0

    console.print(
        f"→ Processed {len(SAMPLE_DOCS)} docs in {time_taken:.2f} seconds "
        f"| ~{throughput:.1f} docs/sec\n"
    )

    # Optional: uncomment to also test smaller / larger batches
    # console.print("Testing with batch_size=8...")
    # _, time_b8 = timed_batch_embed(SAMPLE_DOCS, batch_size=8)
    # console.print(f"  → batch_size=8: {time_b8:.2f} seconds\n")

    console.print(
        "[yellow]Tip:[/yellow] Try changing the multiplier (e.g. *50, *100) "
        "or using skewed duplication to simulate real-world usage patterns."
    )
