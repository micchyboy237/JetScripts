# File: embed_client_demo.py
from __future__ import annotations

import time

from jet.adapters.llama_cpp.parallel_embeddings import embed_batch
from rich.console import Console

console = Console()


def timed_batch_embed(
    texts: list[str],
    max_workers: int,
    batch_size: int,
) -> tuple[list[list[float]], float]:
    """Run batch embed and return results + wall-clock time in seconds."""
    start = time.perf_counter()
    embeddings = embed_batch(texts, max_workers=max_workers, batch_size=batch_size)
    elapsed = time.perf_counter() - start
    return embeddings, elapsed


# === Real-world example data ===
# Simulate 16 support articles / product FAQs (short-medium length)
SAMPLE_DOCS = [
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


if __name__ == "__main__":
    console.rule("llama.cpp Embedding Server Performance Demo")

    console.print(
        "\n[bold cyan]Scenario:[/bold cyan] Indexing 16 customer support articles "
        "for semantic search in a helpdesk RAG system.\n"
        "Goal: Minimize total time to generate embeddings for vector DB insertion.\n"
    )

    console.print(
        "[green]Running concurrent version (max_workers=4) – leverages --parallel 4 + cont-batching[/green]"
    )
    _, time_para = timed_batch_embed(SAMPLE_DOCS, max_workers=4, batch_size=2)
    throughput_para = len(SAMPLE_DOCS) / time_para if time_para > 0 else 0
    console.print(
        f"→ Concurrent: {time_para:.2f} seconds | ~{throughput_para:.1f} docs/sec\n"
    )

    # console.print(
    #     "[yellow]Running sequential version (max_workers=1) – simulates --parallel 1[/yellow]"
    # )
    # _, time_seq = timed_batch_embed(SAMPLE_DOCS, max_workers=1)
    # throughput_seq = len(SAMPLE_DOCS) / time_seq if time_seq > 0 else 0
    # console.print(
    #     f"→ Sequential: {time_seq:.2f} seconds | ~{throughput_seq:.1f} docs/sec\n"
    # )

    # speedup = time_seq / time_para if time_para > 0 else 1
    # console.print(
    #     f"[bold]Speedup factor:[/bold] {speedup:.1f}x faster with parallel requests!"
    # )
    # console.print(
    #     "→ With --parallel + --cont-batching, GPU batches tokens from multiple requests → higher throughput under load."
    # )
