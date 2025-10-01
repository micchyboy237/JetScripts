from typing import Iterator, List, TypedDict
from jet.libs.stanza.utils import serialize_stanza_object
from jet.llm.models import OLLAMA_MODEL_NAMES
from jet.llm.utils.embeddings import generate_embeddings_stream
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.transformers.object import make_serializable
from jet.utils.class_utils import get_non_empty_primitive_attributes
from jet.code.markdown_utils._markdown_parser import derive_by_header_hierarchy
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

def compute_similarities(
    query_vector: np.ndarray,
    doc_vectors: np.ndarray
) -> np.ndarray:
    """Compute cosine similarity scores between query and document vectors."""
    return np.dot(doc_vectors, query_vector) / (
        np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vector) + 1e-10
    )

def search(
    query: str,
    documents: List[str],
    model: str | OLLAMA_MODEL_NAMES = "all-minilm:33m",
    top_k: int = None
) -> Iterator[List[SearchResult]]:
    """Search for documents most similar to the query.

    If top_k is None, return all results sorted by similarity.
    """
    if not documents:
        return []

    query_vector = None
    vectors_list: list[np.ndarray] = []

    # Stream embeddings batch by batch
    for vectors in generate_embeddings_stream([query] + documents, model, use_cache=True, show_progress=True):
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)

        if query_vector is None:
            # First element of first batch is the query
            query_vector = vectors[0]
            vectors_list.extend(vectors[1:])
        else:
            vectors_list.extend(vectors)

        # Compute similarity using the separated function
        similarities = compute_similarities(query_vector, vectors)
        sorted_indices = np.argsort(similarities)[::-1]

        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]

        yield [
            {
                "rank": i + 1,
                "doc_index": int(sorted_indices[i]),
                "score": float(similarities[sorted_indices[i]]),
                "text": documents[sorted_indices[i]],
            }
            for i in range(len(sorted_indices))
        ]

    # doc_vectors = np.array(vectors_list)

if __name__ == "__main__":
    md_content = load_file("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/https_docs_tavily_com_documentation_api_reference_endpoint_crawl/markdown.md")
    model: OLLAMA_MODEL_NAMES = "embeddinggemma"

    headers = derive_by_header_hierarchy(md_content, ignore_links=True)
    docs = [f"{header["header"]}\n\n{header["content"]}" for header in headers]

    # Search
    query = "How to change max depth?"
    # texts = [doc["text"] for doc in all_contexts]

    nlp = stanza.Pipeline('en', dir=DEFAULT_MODEL_DIR, processors='tokenize,pos', verbose=True, logging_level="DEBUG")
    doc_stream = nlp.stream(docs)

    doc_pos = []
    doc_info = []
    doc_dict = {}
    doc_texts = []
    for doc in doc_stream:
        pos = make_serializable(str(doc))
        doc_pos.append(pos)
        save_file(doc_pos, f"{OUTPUT_DIR}/pos.json")

        info = get_non_empty_primitive_attributes(doc)
        doc_info.append(info)
        save_file(doc_info, f"{OUTPUT_DIR}/info.json")
        
        data = serialize_stanza_object(doc)
        for key, value in data.items():
            doc_dict[key] = doc_dict.get(key, [])
            doc_dict[key].append(value)
            save_file(doc_dict[key], f"{OUTPUT_DIR}/{key}.json")

        texts = [sent["text"] for sent in data["sentences"]]
        doc_texts.append(texts)
        save_file(doc_texts, f"{OUTPUT_DIR}/texts.json")

    flattened_texts = [text for doc_text in doc_texts for text in doc_text]
    search_results = []
    for results in search(query, flattened_texts, model):
        search_results.extend(results)
        save_file({
            "query": query,
            "count": len(search_results),
            "results": search_results,
        }, f"{OUTPUT_DIR}/search_results.json")
