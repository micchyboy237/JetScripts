"""
RAG Web Demo - Long Context Lab
================================
Complete usage example: Scrape web pages → Index in HierarchicalMemory → Query → Save results

Features:
- Real embeddings (embeddinggemma)
- Text + source tracking
- Saves retrieval results as JSON
- Configurable via constants
- CPU-only, works on Mac M1 / Windows 10
"""

import json
import os
import shutil
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterator
from pathlib import Path

# --- External Dependencies (install once) ---
# pip install beautifulsoup4 playwright
# from sentence_transformers import SentenceTransformer
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding  # NEW

# === UPDATED IMPORTS ===
from jet.scrapers.playwright_utils import scrape_urls_sync
from bs4 import BeautifulSoup

from jet.search.searxng import search_searxng
from jet.utils.text import format_sub_source_dir
from jet.file.utils import save_file

# --- Import from long_context_lab.py (same directory) ---
from jet.libs.context_engineering.course._02_context_processing.labs.long_context_lab import (
    ContextProcessor,
    HierarchicalMemory
)

# ================================
# CONFIGURATION (Easy to tweak)
# ================================
EMBEDDER_MODEL = "embeddinggemma"        # Fast, 384-dim
CHUNK_SIZE_TOKENS = 256                   # Approx tokens per chunk
MAX_RETRIEVAL_TOKENS = 1024               # Context budget for LLM
SENTENCE_MIN_CHARS = 50                   # Filter noise

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

OUTPUT_DIR = Path(OUTPUT_DIR)
INPUT_URLS_FILE = OUTPUT_DIR / "urls.txt"
RESULTS_FILE = OUTPUT_DIR / "rag_retrieval_results.json"

# --- Add to CONFIGURATION section (after RESULTS_FILE) ---
CHUNKS_FILE = OUTPUT_DIR / "indexed_chunks.json"
MEMORY_FILE = OUTPUT_DIR / "memory_state.json"

# --- ADD CONSTANTS (after OUTPUT_DIR setup) ---
DEFAULT_SEARCH_QUERY = "long context attention mechanisms site:arxiv.org OR site:huggingface.co OR site:wikipedia.org"

# Ensure directories
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
INPUT_URLS_FILE.parent.mkdir(exist_ok=True)

# ================================
# PATCH: Enhanced Memory with Text & Source & Chunk Index
# ================================
@dataclass
class MemoryChunk:
    embedding: np.ndarray
    text: str
    source: str
    token_count: int
    chunk_index: int  # ← NEW: index within the source document

class EnhancedHierarchicalMemory(HierarchicalMemory):
    """Extends HierarchicalMemory to store raw text, source, and chunk index."""

    def __init__(self, d_model: int, short_term_size: int = 512,
                 medium_term_size: int = 1024, compression_ratio: int = 4):
        super().__init__(d_model, short_term_size, medium_term_size, compression_ratio)
        self.short_term: List[MemoryChunk] = []
        self.medium_term: List[MemoryChunk] = []
        self.long_term: List[MemoryChunk] = []
        # These will now be (768, 768) if d_model=768
        self.compress_medium = np.random.randn(d_model, d_model) * 0.02
        self.compress_long = np.random.randn(d_model, d_model) * 0.02

    def add_context(self, embedding: np.ndarray, text: str, source: str, chunk_index: int) -> Dict[str, int]:
        token_count = max(1, len(text.split()) // 4)  # Rough estimate
        chunk = MemoryChunk(embedding, text, source, token_count, chunk_index)  # include chunk_index
        self.short_term.append(chunk)

        # Promote to medium/long with compression
        while self._total_tokens(self.short_term) > self.short_term_size:
            oldest = self.short_term.pop(0)
            compressed_emb = self._compress_context(oldest.embedding, self.compress_medium)
            compressed_text = self._summarize_text(oldest.text)  # Optional: truncate
            self.medium_term.append(MemoryChunk(
                compressed_emb, compressed_text, oldest.source, oldest.token_count // self.compression_ratio, oldest.chunk_index
            ))

        while self._total_tokens(self.medium_term) > self.medium_term_size:
            oldest = self.medium_term.pop(0)
            compressed_emb = self._compress_context(oldest.embedding, self.compress_long)
            self.long_term.append(MemoryChunk(
                compressed_emb, oldest.text[:500], oldest.source, max(1, oldest.token_count // 4), oldest.chunk_index
            ))

        return {
            'short_term': self._total_tokens(self.short_term),
            'medium_term': self._total_tokens(self.medium_term),
            'long_term': self._total_tokens(self.long_term)
        }

    def retrieve_relevant(self, query_emb: np.ndarray, max_tokens: int = 256) -> Tuple[np.ndarray, str, List[Dict]]:
        all_chunks = []
        for chunk in self.short_term:
            score = self._compute_relevance(query_emb, chunk.embedding) * 1.0
            all_chunks.append((score, chunk, 'short'))
        for chunk in self.medium_term:
            score = self._compute_relevance(query_emb, chunk.embedding) * 0.7
            all_chunks.append((score, chunk, 'medium'))
        for chunk in self.long_term:
            score = self._compute_relevance(query_emb, chunk.embedding) * 0.4
            all_chunks.append((score, chunk, 'long'))

        all_chunks.sort(key=lambda x: x[0], reverse=True)

        selected_embs = []
        selected_texts = []
        selected_meta = []
        total_tokens = 0

        for score, chunk, level in all_chunks:
            if total_tokens + chunk.token_count <= max_tokens:
                selected_embs.append(chunk.embedding)
                selected_texts.append(chunk.text)
                selected_meta.append({
                    "source": chunk.source,
                    "level": level,
                    "score": float(score),
                    "tokens": chunk.token_count,
                    "chunk_index": chunk.chunk_index  # ← NOW INCLUDED
                })
                total_tokens += chunk.token_count
            else:
                break

        combined_emb = np.concatenate(selected_embs, axis=0) if selected_embs else np.zeros((0, self.d_model))
        combined_text = "\n\n".join(selected_texts)
        return combined_emb, combined_text, selected_meta

    def _total_tokens(self, memory_list: List[MemoryChunk]) -> int:
        return sum(chunk.token_count for chunk in memory_list)

    def _summarize_text(self, text: str) -> str:
        """Simple truncation for medium-term."""
        return text[:1000] + ("..." if len(text) > 1000 else "")

# ================================
# PATCH: ContextProcessor with Text & Chunk Index
# ================================
class RAGContextProcessor(ContextProcessor):
    def __init__(self, d_model: int = 768, mechanism: str = 'streaming'):  # ← d_model changed from 384 to 768
        super().__init__(d_model, mechanism)
        self.memory = EnhancedHierarchicalMemory(d_model)

    def process_chunk(self, embedding: np.ndarray, text: str, source: str = "", chunk_index: int = 0) -> np.ndarray:
        if self.memory.short_term:
            relevant_emb, _, _ = self.memory.retrieve_relevant(embedding, max_tokens=128)
            if relevant_emb.shape[0] > 0:
                embedding = np.concatenate([relevant_emb, embedding], axis=0)
        output, _ = self.attention.forward(embedding)
        self.memory.add_context(output, text, source, chunk_index)
        return output

# ================================
# Web Scraping & Chunking
# ================================

def extract_text_from_html(html: str) -> str:
    """Extract clean readable text from raw HTML (same logic as before, but reusable)."""
    soup = BeautifulSoup(html, 'html.parser')
    for script in soup(["script", "style", "nav", "footer"]):
        script.decompose()
    text = soup.get_text(separator=' ', strip=True)
    return text

def scrape_urls_playwright(urls: List[str]) -> Iterator[Tuple[str, str]]:
    """
    Scrape multiple URLs using Playwright (sync wrapper) and yield (url, cleaned_text).
    Falls back gracefully on failure.
    """
    print(f"Scraping {len(urls)} URLs using Playwright ({min(10, len(urls))} parallel)...")
    for result in scrape_urls_sync(
        urls=urls,
        num_parallel=10,
        timeout=15000,           # increased slightly for JS-heavy pages
        max_retries=2,
        wait_for_js=True,
        with_screenshot=True,    # not needed for RAG
        use_cache=True,
        show_progress=True
    ):
        url = result["url"]
        status = result["status"]
        html = result.get("html")
        screenshot = result.get("screenshot")

        if status == "completed" and html:
            text = extract_text_from_html(html)
            if text.strip():
                # --- NEW: create per-source sub-dir and save HTML + screenshot ---
                sub_dir = Path(OUTPUT_DIR) / format_sub_source_dir(url)
                sub_dir.mkdir(parents=True, exist_ok=True)

                if html:
                    save_file(html, sub_dir / "raw.html")
                if screenshot:
                    save_file(screenshot, sub_dir / "screenshot.png")

                yield url, text
                print(f"  ✓ Successfully scraped: {url[:70]}{'...' if len(url)>70 else ''}")
            else:
                print(f"  – Empty text after cleaning: {url}")
        else:
            print(f"  ✗ Failed to scrape: {url} [{status}]")
            yield url, ""  # still yield so indexing continues

def chunk_text(text: str, embedder: LlamacppEmbedding) -> List[Tuple[np.ndarray, str, int]]:
    """
    Chunk text into token-limited chunks using sentence boundaries.
    Now uses batched embedding via LlamacppEmbedding (one API call per document).
    Returns a list of (embedding, chunk_text, chunk_index).
    """
    # Step 1: Extract clean sentences
    raw_sentences = [s.strip() for s in text.split('.') if len(s.strip()) > SENTENCE_MIN_CHARS // 10]
    sentences = [s + "." for s in raw_sentences if s]

    if not sentences:
        return []

    # Step 2: Encode ALL sentences in ONE batched call
    print(f"   → Encoding {len(sentences)} sentences in one batch...")
    sentence_embeddings = embedder.encode(
        sentences,
        return_format="numpy",
        batch_size=64,
        show_progress=True
    )  # shape: (n_sentences, d_model)

    # Ensure correct shape
    if sentence_embeddings.ndim == 1:
        sentence_embeddings = sentence_embeddings.reshape(1, -1)

    # Step 3: Rebuild chunks using pre-computed embeddings
    chunks = []
    current_texts = []
    current_embs = []
    chunk_idx = 0  # ← track index per document

    for sent_text, sent_emb in zip(sentences, sentence_embeddings):
        temp_texts = current_texts + [sent_text]
        approx_tokens = sum(len(t.split()) for t in temp_texts) // 4 + 1

        if approx_tokens > CHUNK_SIZE_TOKENS and current_texts:
            chunk_text_str = " ".join(current_texts).strip()
            if chunk_text_str:
                chunk_emb = np.stack(current_embs, axis=0)
                chunks.append((chunk_emb, chunk_text_str, chunk_idx))  # ← include index
                chunk_idx += 1
            current_texts = [sent_text]
            current_embs = [sent_emb]
        else:
            current_texts.append(sent_text)
            current_embs.append(sent_emb)

    # Don't forget last chunk
    if current_texts:
        chunk_text_str = " ".join(current_texts).strip()
        if chunk_text_str:
            chunk_emb = np.stack(current_embs, axis=0)
            chunks.append((chunk_emb, chunk_text_str, chunk_idx))
            chunk_idx += 1

    print(f"   → Generated {len(chunks)} chunks")
    return chunks

# ================================
# Save Results & Index
# ================================

def save_retrieval_result(
    query: str,
    context_text: str,
    metadata: List[Dict],
    filepath: Path,
    extra_data: Dict | None = None
):
    result = {
        "query": query,
        "retrieved_context": context_text,
        "sources": metadata,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_sources": len(metadata)
    }
    if extra_data:
        result.update(extra_data)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {filepath}")

def save_indexed_chunks(chunks: List[Tuple[np.ndarray, str, str, int]], filepath: Path):
    serializable = []
    for emb, text, source, chunk_idx in chunks:
        serializable.append({
            "text": text,
            "source": source,
            "chunk_index": chunk_idx,          # ← added
            "embedding_shape": list(emb.shape),
            "token_estimate": max(1, len(text.split()) // 4)
        })
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"Indexed chunks saved to {filepath}")

def save_memory_state(memory: EnhancedHierarchicalMemory, filepath: Path):
    def chunk_to_dict(chunk: MemoryChunk, level: str):
        return {
            "text": chunk.text,
            "source": chunk.source,
            "token_count": chunk.token_count,
            "embedding_shape": list(chunk.embedding.shape),
            "level": level,
            "chunk_index": chunk.chunk_index
        }

    state = {
        "short_term": [chunk_to_dict(c, "short") for c in memory.short_term],
        "medium_term": [chunk_to_dict(c, "medium") for c in memory.medium_term],
        "long_term": [chunk_to_dict(c, "long") for c in memory.long_term],
        "stats": {
            "short_tokens": memory._total_tokens(memory.short_term),
            "medium_tokens": memory._total_tokens(memory.medium_term),
            "long_tokens": memory._total_tokens(memory.long_term)
        }
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    print(f"Memory state saved to {filepath}")

# ================================
# Main Demo
# ================================
def main():
    print("RAG Web Demo - Initializing...")

    # Prompt for search query
    search_query = input("\nEnter search query for URLs (press Enter for default): ").strip()
    if not search_query:
        search_query = DEFAULT_SEARCH_QUERY
        print(f"Using default query: {search_query}")

    print("\nSearching for URLs using SearXNG...")
    try:
        search_results = search_searxng(
            query=search_query,
            count=10,
            use_cache=True
        )
        urls = [result["url"] for result in search_results if result.get("url")]
        if not urls:
            print("No URLs found from search. Falling back to sample URLs.")
            urls = [
                "https://en.wikipedia.org/wiki/Attention_(machine_learning)",
                "https://arxiv.org/abs/2305.13245",
                "https://huggingface.co/docs/transformers/main/en/model_doc/longformer"
            ]
    except Exception as e:
        print(f"Search failed ({e}). Using sample URLs.")
        urls = [
            "https://en.wikipedia.org/wiki/Attention_(machine_learning)",
            "https://arxiv.org/abs/2305.13245",
            "https://huggingface.co/docs/transformers/main/en/model_doc/longformer"
        ]

    # Write URLs to file
    INPUT_URLS_FILE.write_text("\n".join(urls))
    print(f"Generated {len(urls)} URLs → {INPUT_URLS_FILE}")

    urls = [line.strip() for line in INPUT_URLS_FILE.read_text().splitlines() if line.strip()]
    print(f"Found {len(urls)} URLs to index")

    # 2. Initialize
    embedder = LlamacppEmbedding(
        model=EMBEDDER_MODEL,
        use_cache=True,
        verbose=True
    )
    # --- Optional: Check connection to embedding server
    try:
        test_emb = embedder.encode("test", show_progress=False)
        print(f" ✓ Connected to llama.cpp embedding server (dim={test_emb.shape[-1]})")
    except Exception as e:
        print(f" ✗ Could not connect to embedding server: {e}")
        print("    Make sure `llama-server --model embeddinggemma ... --port 8081` is running")
        exit(1)
    processor = RAGContextProcessor(d_model=768, mechanism='streaming')  # <- changed from 384 to 768

    # 3. Index Web Pages (collect all chunks with source and chunk index)
    print("\nIndexing web content...")
    all_indexed_chunks = []  # now stores (emb, text, url, chunk_idx)
    
    # NEW: Use Playwright-based scraper
    for url, raw_text in scrape_urls_playwright(urls):
        if not raw_text.strip():
            print(f"  Skipping {url} (no content)")
            continue

        print(f"  Chunking content from {url}")
        chunks = chunk_text(raw_text, embedder)
        print(f"    → {len(chunks)} chunks generated")

        for emb, text, chunk_idx in chunks:   # ← include chunk_idx
            all_indexed_chunks.append((emb, text, url, chunk_idx))
            # Routing chunk_index to add_context, but not directly through process_chunk (legacy keeps process_chunk call optional chunk_index)
            processor.memory.add_context(emb, text, url, chunk_idx)

        sub_dir = f"{OUTPUT_DIR}/{format_sub_source_dir(url)}"
        save_file({
            "url": url,
            "count": len(chunks),
        }, f"{sub_dir}/info.json")
        save_file([item[1] for item in chunks], f"{sub_dir}/chunks.json")
    
    save_file([item[1] for item in all_indexed_chunks], f"{OUTPUT_DIR}/all_chunks.json")

    # After indexing, save chunks and memory
    save_indexed_chunks(all_indexed_chunks, CHUNKS_FILE)
    save_memory_state(processor.memory, MEMORY_FILE)

    print("Indexing complete. Memory stats:")
    stats = {
        "short": processor.memory._total_tokens(processor.memory.short_term),
        "medium": processor.memory._total_tokens(processor.memory.medium_term),
        "long": processor.memory._total_tokens(processor.memory.long_term)
    }
    print(f"  Short: {stats['short']}, Medium: {stats['medium']}, Long: {stats['long']} tokens")

    # 4. Query & Retrieve
    # query = input("\nEnter your question: ").strip()
    # if not query:
    #     query = "What are efficient attention mechanisms for long sequences?"
    query = search_query

    print(f"\nRetrieving context for: '{query}'")
    query_emb = embedder.encode(query, return_format="numpy")
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    _, context_text, metadata = processor.memory.retrieve_relevant(query_emb, MAX_RETRIEVAL_TOKENS)

    # 5. Save Results (with extra data)
    result_file = RESULTS_FILE
    extra = {
        "indexed_chunks_file": str(CHUNKS_FILE),
        "memory_state_file": str(MEMORY_FILE),
        "total_indexed_chunks": len(all_indexed_chunks),
        "memory_stats": stats
    }
    save_retrieval_result(query, context_text, metadata, result_file, extra_data=extra)
    print("\nTop sources retrieved:")
    for item in metadata[:3]:
        print(f"  • [{item['level'].upper()}] {item['source'][:60]}... (score: {item['score']:.3f})")

    print(f"\nContext length: ~{len(context_text.split())} words")
    print("Use this context in your LLM prompt!")

if __name__ == "__main__":
    main()