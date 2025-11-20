"""
RAG Web Demo - Long Context Lab
================================
Complete usage example: Scrape web pages → Index in HierarchicalMemory → Query → Save results

Features:
- Real embeddings (all-MiniLM-L6-v2)
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
from typing import List, Tuple, Dict
from pathlib import Path

# --- External Dependencies (install once) ---
# pip install sentence-transformers beautifulsoup4 requests
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup

# --- ADD IMPORT ---
from jet.search.searxng import search_searxng

# --- Import from long_context_lab.py (same directory) ---
from jet.libs.context_engineering.course._02_context_processing.labs.long_context_lab import (
    ContextProcessor,
    HierarchicalMemory
)

# ================================
# CONFIGURATION (Easy to tweak)
# ================================
EMBEDDER_MODEL = "all-MiniLM-L6-v2"        # Fast, 384-dim
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
# PATCH: Enhanced Memory with Text & Source
# ================================
@dataclass
class MemoryChunk:
    embedding: np.ndarray
    text: str
    source: str
    token_count: int

class EnhancedHierarchicalMemory(HierarchicalMemory):
    """Extends HierarchicalMemory to store raw text and source."""

    def __init__(self, d_model: int, short_term_size: int = 512,
                 medium_term_size: int = 1024, compression_ratio: int = 4):
        super().__init__(d_model, short_term_size, medium_term_size, compression_ratio)
        self.short_term: List[MemoryChunk] = []
        self.medium_term: List[MemoryChunk] = []
        self.long_term: List[MemoryChunk] = []
        self.compress_medium = np.random.randn(d_model, d_model) * 0.02
        self.compress_long = np.random.randn(d_model, d_model) * 0.02

    def add_context(self, embedding: np.ndarray, text: str, source: str) -> Dict[str, int]:
        token_count = max(1, len(text.split()) // 4)  # Rough estimate
        chunk = MemoryChunk(embedding, text, source, token_count)
        self.short_term.append(chunk)

        # Promote to medium/long with compression
        while self._total_tokens(self.short_term) > self.short_term_size:
            oldest = self.short_term.pop(0)
            compressed_emb = self._compress_context(oldest.embedding, self.compress_medium)
            compressed_text = self._summarize_text(oldest.text)  # Optional: truncate
            self.medium_term.append(MemoryChunk(
                compressed_emb, compressed_text, oldest.source, oldest.token_count // self.compression_ratio
            ))

        while self._total_tokens(self.medium_term) > self.medium_term_size:
            oldest = self.medium_term.pop(0)
            compressed_emb = self._compress_context(oldest.embedding, self.compress_long)
            self.long_term.append(MemoryChunk(
                compressed_emb, oldest.text[:500], oldest.source, max(1, oldest.token_count // 4)
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
                    "tokens": chunk.token_count
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
# PATCH: ContextProcessor with Text
# ================================
class RAGContextProcessor(ContextProcessor):
    def __init__(self, d_model: int = 384, mechanism: str = 'streaming'):
        super().__init__(d_model, mechanism)
        self.memory = EnhancedHierarchicalMemory(d_model)

    def process_chunk(self, embedding: np.ndarray, text: str, source: str = "") -> np.ndarray:
        if self.memory.short_term:
            relevant_emb, _, _ = self.memory.retrieve_relevant(embedding, max_tokens=128)
            if relevant_emb.shape[0] > 0:
                embedding = np.concatenate([relevant_emb, embedding], axis=0)
        output, _ = self.attention.forward(embedding)
        self.memory.add_context(output, text, source)
        return output

# ================================
# Web Scraping & Chunking
# ================================
def scrape_url(url: str) -> str:
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style", "nav", "footer"]):
            script.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return text
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return ""

def chunk_text(text: str, embedder: SentenceTransformer) -> List[Tuple[np.ndarray, str]]:
    sentences = [s.strip() for s in text.split('.') if len(s) > SENTENCE_MIN_CHARS]
    chunks = []
    current_text = ""
    current_embs = []  # NEW: Collect per-sentence embeddings for true [seq_len, d_model]

    for sent in sentences:
        sent_with_dot = sent + "."
        if len(current_text.split()) + len(sent.split()) > CHUNK_SIZE_TOKENS:
            if current_text:
                # NEW: Encode per sentence, stack to 2D
                sent_texts = [s.strip() + "." for s in current_text.split('.') if s.strip()]
                if sent_texts:
                    embs = embedder.encode(sent_texts, convert_to_numpy=True)  # [num_sents, 384]
                    # Ensure 2D: If single, reshape
                    if embs.ndim == 1:
                        embs = embs.reshape(1, -1)
                    chunks.append((embs, current_text.strip()))
            current_text = sent_with_dot
        else:
            current_text += " " + sent_with_dot

    if current_text.strip():
        # NEW: Same stacking logic
        sent_texts = [s.strip() + "." for s in current_text.split('.') if s.strip()]
        if sent_texts:
            embs = embedder.encode(sent_texts, convert_to_numpy=True)
            if embs.ndim == 1:
                embs = embs.reshape(1, -1)
            chunks.append((embs, current_text.strip()))

    return chunks

# ================================
# Save Results & Index
# ================================

# --- Update save_retrieval_result to accept optional extra_data ---
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

# --- New: Save all indexed chunks ---
def save_indexed_chunks(chunks: List[Tuple[np.ndarray, str, str]], filepath: Path):
    serializable = []
    for emb, text, source in chunks:
        serializable.append({
            "text": text,
            "source": source,
            "embedding_shape": list(emb.shape),
            "token_estimate": max(1, len(text.split()) // 4)
        })
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"Indexed chunks saved to {filepath}")

# --- New: Save full memory state ---
def save_memory_state(memory: EnhancedHierarchicalMemory, filepath: Path):
    def chunk_to_dict(chunk: MemoryChunk, level: str):
        return {
            "text": chunk.text,
            "source": chunk.source,
            "token_count": chunk.token_count,
            "embedding_shape": list(chunk.embedding.shape),
            "level": level
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
    embedder = SentenceTransformer(EMBEDDER_MODEL)
    processor = RAGContextProcessor(d_model=384, mechanism='streaming')

    # 3. Index Web Pages (collect all chunks with source)
    print("\nIndexing web content...")
    all_indexed_chunks = []  # type: List[Tuple[np.ndarray, str, str]]
    for idx, url in enumerate(urls):
        print(f"  [{idx+1}/{len(urls)}] Scraping {url}")
        raw_text = scrape_url(url)
        if not raw_text:
            continue
        chunks = chunk_text(raw_text, embedder)
        # <<<--- INSERT DEBUG LINE HERE --->
        print(f"Sample chunk shape: {chunks[0][0].shape if chunks else 'None'}")  # e.g., (5, 384)
        # <<<-------------------------------->
        for emb, text in chunks:
            all_indexed_chunks.append((emb, text, url))
            processor.process_chunk(emb, text, source=url)

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
    query_emb = embedder.encode(query, convert_to_numpy=True).reshape(1, -1)
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