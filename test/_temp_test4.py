import logging
import time
from typing import List
import numpy as np
import mlx.core as mx
from tqdm import tqdm
from jet.file.utils import load_file, save_file
import os
from llama_cpp import Llama
from mlx_lm import load
from jet.models.embeddings.base import generate_embeddings
from jet.utils.commands import copy_to_clipboard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format query with task instruction."""
    return f'Instruct: {task_description}\nQuery: {query}'


def compute_similarity_scores(
    query_embedding: List[float],
    document_embeddings: List[List[float]]
) -> List[float]:
    """
    Compute cosine similarity scores between a query embedding and a list of document embeddings.

    Args:
        query_embedding: List of floats representing the query's embedding.
        document_embeddings: List of lists of floats representing document embeddings.

    Returns:
        List of cosine similarity scores, one for each document embedding.
    """
    if not query_embedding or not document_embeddings:
        return []

    # Convert to numpy for efficient computation
    query_array = np.array(query_embedding, dtype=np.float32)
    doc_array = np.array(document_embeddings, dtype=np.float32)

    # Normalize vectors for cosine similarity
    query_norm = np.linalg.norm(query_array)
    doc_norms = np.linalg.norm(doc_array, axis=1)
    if query_norm == 0 or np.any(doc_norms == 0):
        return [0.0] * len(document_embeddings)

    query_array /= query_norm
    doc_array /= doc_norms[:, np.newaxis]

    # Compute cosine similarity
    scores = np.dot(doc_array, query_array)

    return scores.tolist()


# Load the GGUF model (unchanged)
model_path = "/Users/jethroestrada/Downloads/Qwen3-Embedding-0.6B-f16.gguf"
model = Llama(
    model_path=model_path,
    embedding=True,
    n_ctx=512,
    n_threads=4,
    n_threads_batch=4,
    n_gpu_layers=0,
    n_batch=32,
    verbose=True
)

# Load documents (unchanged)
docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
output_dir = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

docs = load_file(docs_file)
documents = [
    "\n".join([
        doc["metadata"].get("parent_header") or "",
        doc["metadata"]["header"],
    ]).strip()
    for doc in docs
]
task = 'Given a web search query, retrieve relevant passages that answer the query'
query = "List all ongoing and upcoming isekai anime 2025."

queries = [get_detailed_instruct(task, query)]
documents = documents[:64]

# Load embedding model (unchanged)
model_name = "mlx-community/Qwen3-0.6B-4bit-DWQ-053125"
model, _ = load(model_name)

# Generate embeddings
query_embeddings = generate_embeddings(
    query, model_name, model=model, show_progress=True)
document_embeddings = generate_embeddings(
    documents, model_name, model=model, show_progress=True)

# Log embedding details for debugging
copy_to_clipboard([
    len(query_embeddings) if isinstance(query_embeddings,
                                        (list, mx.array)) else "not a list/array",
    len(query_embeddings[0]) if isinstance(query_embeddings,
                                           list) and query_embeddings else "no query embeddings",
    str(type(query_embeddings)),
    len(document_embeddings),
    len(document_embeddings[0]) if document_embeddings else 0,
    str(type(document_embeddings)),
    str(type(document_embeddings[0])
        ) if document_embeddings else "no document embeddings",
])

# Convert embeddings to compatible format
# query_embeddings might be an mx.array (e.g., shape [1, 151936]) or a list [array([...])]
if isinstance(query_embeddings, mx.array):
    # Convert to list and flatten if necessary
    query_embedding = mx.flatten(query_embeddings).tolist()
elif isinstance(query_embeddings, list) and query_embeddings:
    # Extract first element and flatten if necessary
    query_emb = query_embeddings[0]
    query_embedding.ndim > 0 and (
        query_embedding == query_embedding.flatten[0])
    query_embedding = mx.flatten(query_emb).tolist()
else:
    query_embedding = []

# document_embeddings is a list of mx.array, convert each to list
document_embeddings = [
    mx.flatten(doc_emb).tolist() for doc_emb in document_embeddings
]

# Log converted shapes for debugging
copy_to_clipboard([
    len(query_embedding),
    str(type(query_embedding[0])
        ) if query_embedding else "empty query_embedding",
    len(document_embeddings),
    len(document_embeddings[0]) if document_embeddings else 0,
    str(type(document_embeddings[0][0])
        ) if document_embeddings else "empty document_embeddings",
])

logger.info(
    f"\nComputing similarity scores for {len(document_embeddings)} docs...")
start_time = time.time()

# Compute similarity scores
scores = compute_similarity_scores(query_embedding, document_embeddings)

end_time = time.time()
execution_time = end_time - start_time

save_file(
    {"execution_time": f"{execution_time:.4f}s",
        "count": len(scores), "results": scores},
    f"{output_dir}/similarity.json"
)

# try:
#     # Encode queries and documents
#     query_embeddings, query_total_tokens = model.embed(
#         queries, normalize=True, return_count=True)
#     document_embeddings, documenty_total_tokens = model.embed(
#         documents, normalize=True, return_count=True)

#     # Compute similarity scores (queries vs documents)
#     scores = query_embeddings @ document_embeddings.T
#     print("Similarity matrix:")
#     print(scores.tolist())

# except Exception as e:
#     print(f"Error during embedding or similarity computation: {str(e)}")
# finally:
#     # Clean up model resources
#     model.close()


# tests/test_similarity_scores.py


class TestComputeSimilarityScores:
    def test_valid_embeddings(self):
        # Test with valid query and document embeddings
        query_embedding = [1.0, 0.0]
        document_embeddings = [[1.0, 0.0], [0.0, 1.0]]
        expected = [1.0, 0.0]  # Cosine similarity: [1.0, 0.0]
        result = compute_similarity_scores(
            query_embedding, document_embeddings)
        assert result == pytest.approx(
            expected, rel=1e-5), f"Expected {expected}, got {result}"

    def test_empty_embeddings(self):
        # Test with empty inputs
        query_embedding = []
        document_embeddings = []
        expected = []
        result = compute_similarity_scores(
            query_embedding, document_embeddings)
        assert result == expected, f"Expected {expected}, got {result}"

    def test_zero_norm_vectors(self):
        # Test with zero-norm query embedding
        query_embedding = [0.0, 0.0]
        document_embeddings = [[1.0, 0.0], [0.0, 1.0]]
        expected = [0.0, 0.0]
        result = compute_similarity_scores(
            query_embedding, document_embeddings)
        assert result == expected, f"Expected {expected}, got {result}"

    def test_single_document(self):
        # Test with single document embedding
        query_embedding = [1.0, 0.0]
        document_embeddings = [[0.0, 1.0]]
        expected = [0.0]
        result = compute_similarity_scores(
            query_embedding, document_embeddings)
        assert result == pytest.approx(
            expected, rel=1e-5), f"Expected {expected}, got {result}"
