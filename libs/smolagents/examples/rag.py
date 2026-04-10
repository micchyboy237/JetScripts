import asyncio
import logging
import os
import time
from typing import List

# ChromaDB
import chromadb
import numpy as np
import requests
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

# Crawl4AI imports
from crawl4ai import (
    AdaptiveConfig,
    AdaptiveCrawler,  # For intelligent "until satisfied" crawling
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    DefaultMarkdownGenerator,
    PruningContentFilter,
)
from pydantic import BaseModel

# Rich for beautiful logging & progress
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# ========================= CONFIG =========================
QUERY = "machine learning tutorials with code examples"
SEED_URL = "https://example.com/blog"  # ← Change this

MAX_DEPTH = 4
MAX_PAGES = 80
RELEVANCE_THRESHOLD = 0.78

LLAMA_CPP_EMBED_URL = os.getenv("LLAMA_CPP_EMBED_URL")
if not LLAMA_CPP_EMBED_URL:
    raise ValueError("Set LLAMA_CPP_EMBED_URL (e.g. http://shawn-pc.local:8081)")

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, show_time=True, rich_tracebacks=True)],
)
logger = logging.getLogger("crawl_pipeline")


# ========================= FIXED LLAMA.CPP EMBEDDING FUNCTION =========================
class LlamaCppEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def __call__(self, input: Documents) -> Embeddings:
        if isinstance(input, str):
            input = [input]

        embeddings: List[List[float]] = []
        for text in input:
            emb = None
            # Try legacy endpoint first (common for llama.cpp --embeddings)
            try:
                payload = {"content": text}
                resp = self.session.post(
                    f"{self.base_url}/embedding", json=payload, timeout=60
                )
                resp.raise_for_status()
                data = resp.json()
                emb = data.get("embedding") or data.get("data", data)
            except Exception:
                # Fallback to OpenAI-compatible endpoint
                try:
                    payload = {"input": text, "model": "llama-cpp-embedding"}
                    resp = self.session.post(
                        f"{self.base_url}/v1/embeddings", json=payload, timeout=60
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    emb = data["data"][0]["embedding"]
                except Exception as e:
                    logger.warning(
                        f"Embedding failed for text (first 100 chars): {text[:100]}... | Error: {e}"
                    )
                    emb = [0.0] * 384  # Adjust to your model's dimension if known

            embeddings.append(emb)
        return embeddings

    def encode(self, texts: List[str], normalize: bool = True):
        emb_list = self(texts)
        emb_array = np.array(emb_list, dtype=np.float32)
        if normalize and emb_array.size > 0:
            norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
            emb_array = emb_array / (norms + 1e-8)
        return emb_array


embed_function = LlamaCppEmbeddingFunction(LLAMA_CPP_EMBED_URL)
query_embedding = embed_function.encode([QUERY])[0]


def compute_relevance(markdown: str) -> float:
    if not markdown or len(markdown.strip()) < 50:
        return 0.0
    page_embedding = embed_function.encode([markdown])[0]
    return float(np.dot(query_embedding, page_embedding))


class CrawlResult(BaseModel):
    url: str
    markdown: str
    relevance_score: float = 0.0


# ========================= MAIN PIPELINE =========================
async def main():
    start_time = time.time()
    console.print(
        Panel(
            f"[bold cyan]Starting Adaptive Deep Search[/bold cyan]\nQuery: [yellow]{QUERY}[/yellow]\nSeed: [blue]{SEED_URL}[/blue]\nEmbed URL: [green]{LLAMA_CPP_EMBED_URL}[/green]",
            title="Crawl4AI + llama.cpp",
        )
    )

    markdown_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.48, min_word_threshold=20)
    )

    browser_config = BrowserConfig(headless=True)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.ENABLED,
        markdown_generator=markdown_generator,
        remove_overlay_elements=True,
        exclude_social_media_links=True,
        word_count_threshold=80,
    )

    all_results: List[CrawlResult] = []
    satisfied = False
    pages_processed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        overall_task = progress.add_task("[cyan]Adaptive crawling...", total=MAX_PAGES)

        async with AsyncWebCrawler(config=browser_config) as crawler:
            adaptive_config = AdaptiveConfig(
                confidence_threshold=RELEVANCE_THRESHOLD - 0.1,
                max_pages=MAX_PAGES,
                # strategy="statistical",   # or "embedding" if you want to try it
            )

            adaptive = AdaptiveCrawler(crawler, config=adaptive_config)

            # Run intelligent adaptive digest (best for "until satisfied")
            result_state = await adaptive.digest(
                start_url=SEED_URL,
                query=QUERY,
            )

            # AdaptiveCrawler returns a CrawlState; extract results from it
            result_list = getattr(result_state, "results", []) or []

            for result in result_list:
                if hasattr(result, "markdown") and result.markdown:
                    score = compute_relevance(result.markdown)
                    crawl_res = CrawlResult(
                        url=result.url, markdown=result.markdown, relevance_score=score
                    )
                    all_results.append(crawl_res)

                    pages_processed += 1
                    progress.update(overall_task, completed=pages_processed)

                    status = "✅ SATISFIED" if score >= RELEVANCE_THRESHOLD else "📄"
                    logger.info(f"{status} [{score:.3f}] {result.url}")

                    if score >= RELEVANCE_THRESHOLD and not satisfied:
                        logger.info(
                            "[bold green]Query satisfaction threshold reached![/bold green]"
                        )
                        satisfied = True

    # ========================= SUMMARY =========================
    elapsed = time.time() - start_time
    all_results.sort(key=lambda x: x.relevance_score, reverse=True)
    top_pages = all_results[:10]

    table = Table(title="Crawl Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")
    table.add_row("Pages Crawled", str(len(all_results)))
    table.add_row(
        "Query Satisfied",
        "[bold green]Yes[/bold green]" if satisfied else "[red]No[/red]",
    )
    table.add_row("Time Taken", f"{elapsed:.1f} s")
    table.add_row(
        "Best Relevance", f"{top_pages[0].relevance_score:.3f}" if top_pages else "N/A"
    )

    console.print(table)

    # Save top results
    os.makedirs("crawl_output", exist_ok=True)
    for i, page in enumerate(top_pages):
        with open(
            f"crawl_output/top_{i + 1}_{page.relevance_score:.3f}.md",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(
                f"# {page.url}\n\nRelevance: {page.relevance_score:.3f}\n\n{page.markdown}"
            )

    logger.info(
        Panel("[bold green]Crawl finished successfully![/bold green]", title="Done")
    )

    # Chroma indexing
    client = chromadb.PersistentClient(path="./crawl_chroma_db")
    collection = client.get_or_create_collection(
        name="crawl_results", embedding_function=embed_function
    )

    for page in all_results:
        collection.add(
            documents=[page.markdown],
            metadatas=[{"url": page.url, "relevance": page.relevance_score}],
            ids=[page.url.replace("/", "_").replace(":", "_").replace("?", "_")[:200]],
        )

    logger.info("✅ Results indexed in ./crawl_chroma_db/")


if __name__ == "__main__":
    asyncio.run(main())
