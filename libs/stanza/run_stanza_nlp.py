from typing import Iterator, List, TypedDict
from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.libs.llama_cpp.embeddings import LlamacppEmbedding
from jet.libs.stanza.utils import serialize_stanza_object
from jet.llm.models import OLLAMA_MODEL_NAMES
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.transformers.object import make_serializable
from jet.utils.class_utils import get_non_empty_primitive_attributes
from jet.wordnet.text_chunker import chunk_texts_with_data
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

def preprocess_text(text: str) -> str:
    """
    Preprocesses a single text by normalizing whitespace, converting to lowercase,
    and removing special characters.

    Args:
        text (str): Input text to preprocess.

    Returns:
        str: Preprocessed text.
    """
    # import re
    # # Convert to lowercase
    # text = text.lower()
    # # Remove special characters, keeping alphanumeric and spaces
    # text = re.sub(r'[^\w\s]', '', text)
    # # Normalize whitespace (replace multiple spaces with single space, strip)
    # text = re.sub(r'\s+', ' ', text).strip()
    return text

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

    # Initialize the embedding client
    embedder = LlamacppEmbedding(model=model)

    # Stream embeddings batch by batch
    for vectors in embedder.get_embeddings_stream([query] + documents, model, show_progress=True):
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
    html_string = load_file("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/https_docs_tavily_com_documentation_api_reference_endpoint_crawl/page.html")
    model = "embeddinggemma"
    query = "How to change max depth?"

    md_content = convert_html_to_markdown(html_string, ignore_links=True)
    chunks = chunk_texts_with_data(md_content, chunk_size=150, chunk_overlap=40)
    chunk_tokens = [chunk["num_tokens"] for chunk in chunks]
    save_file({
        "count": len(chunks),
        "tokens": {
            "min": min(chunk_tokens),
            "max": max(chunk_tokens),
            "sum": sum(chunk_tokens),
        },
        "texts": [{"doc_index": chunk["doc_index"], "tokens": chunk["num_tokens"], "text": chunk["content"]} for chunk in chunks],
    }, f"{OUTPUT_DIR}/chunks.json")

    docs = [chunk["content"] for chunk in chunks]
    save_file(docs, f"{OUTPUT_DIR}/docs.json")

    search_results = []
    for results in search(query, docs, model):
        search_results.extend(results)
        save_file({
            "query": query,
            "count": len(search_results),
            "results": search_results,
        }, f"{OUTPUT_DIR}/search_chunks_results.json")

    
    # Stanza NLP
    
    # docs = [doc["text"] for doc in all_contexts]

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

        sentences = [sent["text"] for sent in data["sentences"]]
        doc_texts.append(sentences)
        save_file(doc_texts, f"{OUTPUT_DIR}/sentences.json")

    flattened_texts = [text for doc_text in doc_texts for text in doc_text]
    search_results = []
    for results in search(query, flattened_texts, model):
        search_results.extend(results)
        save_file({
            "query": query,
            "count": len(search_results),
            "results": search_results,
        }, f"{OUTPUT_DIR}/search_sentences_results.json")
