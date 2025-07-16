import os
import shutil
from typing import TypedDict, List
import numpy as np
from jet.file.utils import load_file, save_file
from jet.models.embeddings.base import generate_embeddings
from jet.models.embeddings.chunking import chunk_headers_by_hierarchy
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.models.tokenizer.base import get_tokenizer_fn
from shared.data_types.job import JobData

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR)


class SearchResult(TypedDict):
    rank: int
    score: float
    job_title: str
    content: str


def chunk_text(text: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
    """Split text into chunks based on token count with overlap."""
    tokenizer = get_tokenizer_fn("mxbai-embed-large")
    tokens = tokenizer(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens))
        start += max_tokens - overlap

    return chunks


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def vector_search(query: str, texts: List[str], embed_model: EmbedModelType, top_k: int = 10) -> List[SearchResult]:
    """Perform vector search with chunking and return ranked results."""
    # Chunk texts if needed
    all_chunks = []
    chunk_to_doc = []

    for doc_idx, text in enumerate(texts):
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        chunk_to_doc.extend([(doc_idx, text) for _ in chunks])

    # Generate embeddings for query and all chunks
    embeddings = generate_embeddings(
        [query] + all_chunks,
        embed_model,
        return_format="numpy",
        show_progress=True
    )

    query_embedding = embeddings[0]
    chunk_embeddings = embeddings[1:]

    # Calculate similarities
    similarities = [
        (cosine_similarity(query_embedding, chunk_emb), doc_idx, orig_text)
        for chunk_emb, (doc_idx, orig_text) in zip(chunk_embeddings, chunk_to_doc)
    ]

    # Aggregate scores by document (take max score across chunks)
    doc_scores = {}
    for score, doc_idx, orig_text in similarities:
        if doc_idx not in doc_scores or score > doc_scores[doc_idx][0]:
            doc_scores[doc_idx] = (score, orig_text)

    # Sort by score and create results
    results = []
    for rank, (doc_idx, (score, content)) in enumerate(
        sorted(doc_scores.items(), key=lambda x: x[1][0], reverse=True)[
            :top_k], 1
    ):
        job_title = content.split("\n\n")[0].replace(
            "# Job Title\n\n", "").strip()
        results.append(SearchResult(
            rank=rank,
            score=float(score),
            job_title=job_title,
            content=content
        ))

    return results


if __name__ == "__main__":
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"

    data: list[JobData] = load_file(data_file)
    chunk_size = 150
    embed_model: EmbedModelType = "mxbai-embed-large"
    llm_model: LLMModelType = "qwen3-1.7b-4bit-dwq-053125"
    system = None

    texts = [
        "\n\n".join([
            f"# Job Title\n\n{item['title']}",
            f"## Details\n\n{item['details']}",
            *[
                f"## {key.replace('_', ' ').title()}\n\n" +
                "\n".join([f"- {value}" for value in item["entities"][key]])
                for key in item["entities"]
            ],
            f"## Tags\n\n" + "\n".join([f"- {tag}" for tag in item["tags"]]),
        ])
        for item in data
    ]
    save_file(texts, f"{OUTPUT_DIR}/docs.json")

    tokenizer = get_tokenizer_fn(embed_model)
    chunks = [chunk for text in texts for chunk in chunk_headers_by_hierarchy(
        text, chunk_size, tokenizer)]
    save_file(chunks, f"{OUTPUT_DIR}/chunks.json")
    # query = "React web"
    # search_results = vector_search(query, texts, embed_model)

    # save_file(search_results, f"{OUTPUT_DIR}/search_results.json")
