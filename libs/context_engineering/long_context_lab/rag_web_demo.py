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
import uuid
import numpy as np
import tiktoken
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterator
from pathlib import Path
from tqdm import tqdm

# --- External Dependencies (install once) ---
# pip install beautifulsoup4 playwright
# from sentence_transformers import SentenceTransformer
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding  # NEW

# === UPDATED IMPORTS ===
from jet.scrapers.playwright_utils import scrape_urls_sync
from bs4 import BeautifulSoup

from jet.search.searxng import search_searxng
from jet.wordnet.text_chunker import chunk_texts_with_data
from jet.code.extraction import extract_sentences
from jet._token.token_utils import token_counter
from jet.utils.text import format_sub_source_dir
from jet.file.utils import save_file

# --- Import from long_context_lab.py (same directory) ---
from jet.libs.context_engineering.course._02_context_processing.labs.long_context_lab import (
    ContextProcessor,
    HierarchicalMemory
)

# === LLM imports ===
from jet.adapters.llama_cpp.llm import LlamacppLLM
from jet.logger import logger

# ================================
# CONFIGURATION (Easy to tweak)
# ================================
EMBED_MODEL = "embeddinggemma"         # Fast, 768-dim
LLM_MODEL = "qwen3-instruct-2507:4b"       # or any model you have in llama-server --model

CHUNK_SIZE = 256                          # Approx tokens per chunk
CHUNK_OVERLAP = 32                        # Approx overlapped tokens between chunks
MAX_RETRIEVAL_TOKENS = 1024               # Context budget for LLM

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

OUTPUT_DIR = Path(OUTPUT_DIR)
INPUT_URLS_FILE = OUTPUT_DIR / "urls.txt"
LLM_DIR = OUTPUT_DIR / "llm"
RESULTS_DIR = OUTPUT_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "rag_retrieval_results.json"

# --- Add to CONFIGURATION section (after RESULTS_FILE) ---
CHUNKS_FILE = OUTPUT_DIR / "indexed_chunks.json"
MEMORY_FILE = OUTPUT_DIR / "memory_state.json"

# --- ADD CONSTANTS (after OUTPUT_DIR setup) ---
# DEFAULT_SEARCH_QUERY = "long context attention mechanisms site:arxiv.org OR site:huggingface.co OR site:wikipedia.org"
DEFAULT_SEARCH_QUERY = "top RAG context engineering tips reddit 2025"

# Ensure directories
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
LLM_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
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

    def _token_count(self, text: str, model: str = "gpt-4o") -> int:
        """Return token count for a single string."""
        enc = tiktoken.encoding_for_model(model)   # automatically picks cl100k_base or o200k_base
        return len(enc.encode(text))

    def add_context(self, embedding: np.ndarray, text: str, source: str, chunk_index: int) -> Dict[str, int]:
        token_count = self._token_count(text)
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
                    "tokens": chunk.token_count,  # Now accurate!
                    "chunk_index": chunk.chunk_index
                })
                total_tokens += chunk.token_count
            else:
                break  # Strictly enforce max_tokens

        combined_emb = np.concatenate(selected_embs, axis=0) if selected_embs else np.zeros((0, self.d_model))
        combined_text = "\n\n".join(selected_texts)
        
        # Optional: log how close we got
        print(f"Retrieved context: {total_tokens}/{max_tokens} tokens used")
        
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
        max_retries=1,
        wait_for_js=True,
        with_screenshot=False,   # not needed for RAG
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

# ================================
# Save Index & Memory
# ================================

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
# RAG Prompt Template
# ================================
def build_rag_prompt(query: str, context: str, metadata: List[Dict]) -> List[Dict[str, str]]:
    """Build a clean, effective RAG prompt for streaming LLMs."""
    sources = "\n".join(
        f"- [{m['level'].upper()}] {m['source']} (score: {m['score']:.3f})"
        for m in metadata[:10]
    )
    system_msg = """
You are an expert assistant. Answer the user's question using ONLY the information from the provided context.
If the context does not contain enough information to answer confidently, say so.
""".strip()

    prompt = f"""\
Context:
{context.strip()}

Sources:
{sources}

Question: {query.strip()}
Answer: """

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]

# ================================
# Main Demo
# ================================
def main():
    print("RAG Web Demo - Initializing...")
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
    
    INPUT_URLS_FILE.write_text("\n".join(urls))
    print(f"Generated {len(urls)} URLs → {INPUT_URLS_FILE}")
    
    urls = [line.strip() for line in INPUT_URLS_FILE.read_text().splitlines() if line.strip()]
    print(f"Found {len(urls)} URLs to index")
    
    embedder = LlamacppEmbedding(
        model=EMBED_MODEL,
        use_cache=True,
        verbose=True
    )
    
    try:
        test_emb = embedder.encode("test", show_progress=False)
        print(f" ✓ Connected to llama.cpp embedding server (dim={test_emb.shape[-1]})")
    except Exception as e:
        print(f" ✗ Could not connect to embedding server: {e}")
        print(" Make sure `llama-server --model embeddinggemma ... --port 8081` is running")
        exit(1)
    
    processor = RAGContextProcessor(d_model=768, mechanism='streaming')
    
    # Scrape URLs and collect documents
    all_docs = []
    for url, raw_text in scrape_urls_playwright(urls):
        if not raw_text.strip():
            print(f" Skipping {url} (no content)")
            continue
        all_docs.append({
            "id": str(uuid.uuid4()),
            "url": url,
            "text": raw_text,
        })
        sub_dir = f"{OUTPUT_DIR}/{format_sub_source_dir(url)}"
        save_file(raw_text, f"{sub_dir}/doc.md")
    
    print(f"\nScraped {len(all_docs)} documents. Starting chunking...")
    
    # Chunk all documents using chunk_texts_with_data
    doc_ids = [doc["id"] for doc in all_docs]
    doc_texts = [doc["text"] for doc in all_docs]
    
    chunks_with_data = chunk_texts_with_data(
        doc_texts,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        model=EMBED_MODEL,
        strict_sentences=True,
        ids=doc_ids,
    )
    
    # Filter out empty chunks
    chunks_with_data = [chunk for chunk in chunks_with_data if chunk["content"].strip()]

    chunk_texts = [chunk["content"] for chunk in chunks_with_data]
    chunk_embeddings = embedder.encode(
        chunk_texts,
        return_format="numpy",
        batch_size=128,
        show_progress=True
    )
    if chunk_embeddings.ndim == 1:
        chunk_embeddings = chunk_embeddings.reshape(1, -1)
    
    chunk_embeddings_mappings = {chunk["id"]: chunk_emb for chunk_emb, chunk in zip(chunk_embeddings, chunks_with_data)}
    
    print(f"Generated {len(chunks_with_data)} chunks from {len(all_docs)} documents")
    
    # Create document-to-chunk mapping
    chunk_doc_mappings = {}
    for chunk in chunks_with_data:
        doc_id = chunk["doc_id"]
        if doc_id not in chunk_doc_mappings:
            chunk_doc_mappings[doc_id] = []
        chunk_doc_mappings[doc_id].append(chunk)
    
    # Process each document's chunks
    all_indexed_sentences = []
    all_chunks = []
    
    for doc_idx, doc in enumerate(all_docs):
        doc_id = doc["id"]
        url = doc["url"]
        raw_text = doc["text"]
        
        # Get chunks for this document
        doc_chunks = chunk_doc_mappings.get(doc_id, [])
        if not doc_chunks:
            print(f" → No chunks found for {url}")
            continue
            
        print(f" → Processing {len(doc_chunks)} chunks for {url}")
        
        # Re-index chunks for this document (0-based for document)
        for chunk_idx, chunk in enumerate(doc_chunks):
            chunk["doc_chunk_index"] = chunk_idx  # Add document-specific index
            
        # Extract content and metadata for embedding
        doc_chunk_texts = [chunk["content"] for chunk in doc_chunks]
        doc_token_counts = [chunk["num_tokens"] for chunk in doc_chunks]
        
        # Move this out for only 1 call to encode
        # chunk_embeddings = embedder.encode(
        #     doc_chunk_texts,
        #     return_format="numpy",
        #     batch_size=128,
        #     show_progress=False
        # )
        
        # if chunk_embeddings.ndim == 1:
        #     chunk_embeddings = chunk_embeddings.reshape(1, -1)
        
        # Store embeddings and metadata for this document
        for emb_idx, chunk in enumerate(doc_chunks):
            chunk_emb = chunk_embeddings_mappings[chunk["id"]]
            all_indexed_sentences.append((
                chunk_emb, 
                chunk["content"], 
                url, 
                chunk["chunk_index"],  # Global chunk index from chunker
                chunk_idx             # Document-specific chunk index
            ))
            all_chunks.append(chunk["content"])
        
        # Save document-specific files
        sub_dir = f"{OUTPUT_DIR}/{format_sub_source_dir(url)}"
        save_file({
            "url": url,
            "doc_id": doc_id,
            "chunk_count": len(doc_chunks),
            "tokens": {
                "min": min(doc_token_counts) if doc_token_counts else 0,
                "ave": sum(doc_token_counts) // len(doc_token_counts) if doc_token_counts else 0,
                "max": max(doc_token_counts) if doc_token_counts else 0,
                "total": sum(doc_token_counts),
            },
        }, f"{sub_dir}/info.json")
        
        save_file(doc_chunks, f"{sub_dir}/chunks.json")
        save_file(doc_chunk_texts, f"{sub_dir}/sentences.json")
    
    # Save global files
    save_file(all_chunks, f"{OUTPUT_DIR}/all_chunks.json")
    
    # Add embeddings to memory (fixed accumulation)
    print(f"\nStoring {len(all_indexed_sentences)} chunks in hierarchical memory...")
    for idx, (chunk_emb, chunk_text, url, global_idx, doc_idx) in enumerate(
        tqdm(all_indexed_sentences, desc="Building memory")
    ):
        # Ensure embedding is 2D (1, embedding_dim)
        if chunk_emb.ndim == 1:
            chunk_emb = chunk_emb.reshape(1, -1)
        processor.memory.add_context(chunk_emb, chunk_text, url, doc_idx)
    
    save_memory_state(processor.memory, MEMORY_FILE)
    
    print("Indexing complete. Memory stats:")
    stats = {
        "short": processor.memory._total_tokens(processor.memory.short_term),
        "medium": processor.memory._total_tokens(processor.memory.medium_term),
        "long": processor.memory._total_tokens(processor.memory.long_term)
    }
    print(f" Short: {stats['short']}, Medium: {stats['medium']}, Long: {stats['long']} tokens")
    
    # Query processing (unchanged)
    query = search_query
    print(f"\nRetrieving context for: '{query}'")
    query_emb = embedder.encode(query, return_format="numpy")
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    
    _, context, metadata = processor.memory.retrieve_relevant(query_emb, MAX_RETRIEVAL_TOKENS)
    print(f"\nRetrieved ~{len(context.split())} words from memory. Generating answer...\n")
    
    logger.info("Initializing LLM for final answer generation...")
    llm = LlamacppLLM(
        model=LLM_MODEL,
        verbose=True,
    )

    llm_dir = LLM_DIR
    result_dir = RESULTS_DIR
    messages = build_rag_prompt(query, context, metadata)

    save_file(messages, llm_dir / "messages.json")
    save_file(metadata, llm_dir / "metadata.json")
    save_file(query, llm_dir / "query.md")
    save_file(context, llm_dir / "context.md")

    chunks = list(llm.chat(messages, temperature=0.1, stream=True))
    response = "".join(chunks)

    info = {
        "indexed_chunks_file": str(CHUNKS_FILE),
        "memory_state_file": str(MEMORY_FILE),
        "total_indexed_chunks": len(all_indexed_sentences),
        "memory_stats": stats,
        "llm_model": LLM_MODEL,
    }

    save_file(info, llm_dir / "info.json")
    save_file({
        "prompt": token_counter(messages, model=LLM_MODEL),
        "response": token_counter(response, model=LLM_MODEL),
    }, llm_dir / "tokens.json")

    print("Final Answer:")
    save_file(response, llm_dir / "response.md")
    
    print(f"\n\nGeneration complete! Full results saved to {llm_dir}")

if __name__ == "__main__":
    main()
