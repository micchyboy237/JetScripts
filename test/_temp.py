from typing import List, TypedDict
from jet.llm.models import OLLAMA_MODEL_NAMES
from jet.llm.utils.embeddings import generate_embeddings
from jet.file.utils import load_file, save_file
from jet.logger import logger

import numpy as np
import os
import shutil

import stanza

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

DEFAULT_MODEL_DIR = os.getenv(
    'STANZA_RESOURCES_DIR',
    os.path.join(os.path.expanduser("~/.cache"), "stanza_resources")
)

class ContextItem(TypedDict):
    doc_idx: int
    tokens: int
    text: str

class SearchResult(TypedDict):
    rank: int
    doc_index: int
    score: float
    text: str

def search(
    query: str,
    documents: List[str],
    model: str | OLLAMA_MODEL_NAMES = "all-minilm:33m",
    top_k: int = None
) -> List[SearchResult]:
    """Search for documents most similar to the query.

    If top_k is None, return all results sorted by similarity.
    """
    if not documents:
        return []
    vectors = generate_embeddings([query] + documents, model, use_cache=True)
    query_vector = vectors[0]
    doc_vectors = vectors[1:]
    similarities = np.dot(doc_vectors, query_vector) / (
        np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vector) + 1e-10
    )
    sorted_indices = np.argsort(similarities)[::-1]
    if top_k is not None:
        sorted_indices = sorted_indices[:top_k]
    return [
        {
            "rank": i + 1,
            "doc_index": int(sorted_indices[i]),
            "score": float(similarities[sorted_indices[i]]),
            "text": documents[sorted_indices[i]],
        }
        for i in range(len(sorted_indices))
    ]


if __name__ == "__main__":
    md_content = load_file("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/https_docs_tavily_com_documentation_api_reference_endpoint_crawl/markdown.md")
    model: OLLAMA_MODEL_NAMES = "embeddinggemma"

    # Search
    query = "How to change max depth?"
    # texts = [doc["text"] for doc in all_contexts]

    nlp = stanza.Pipeline('en', dir=DEFAULT_MODEL_DIR, processors='tokenize,pos', verbose=True, logging_level="DEBUG")
    doc = nlp(md_content)

    sentences = [sent.text.strip() for sent in doc.sentences]
    save_file(sentences, f"{OUTPUT_DIR}/sentences.json")

    search_results = search(query, sentences, model)
    save_file({
        "query": query,
        "count": len(search_results),
        "results": search_results,
    }, f"{OUTPUT_DIR}/search_results.json")