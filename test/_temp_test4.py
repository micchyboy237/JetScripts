import logging
import time
from typing import List
from tqdm import tqdm
from jet.file.utils import load_file, save_file
import os
import numpy as np
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

    # Compute dot product (equivalent to cosine similarity for normalized vectors)
    scores = np.dot(doc_array, query_array)

    # Convert back to list format
    return scores.tolist()


# Load the GGUF model
model_path = "/Users/jethroestrada/Downloads/Qwen3-Embedding-0.6B-f16.gguf"
model = Llama(
    model_path=model_path,
    embedding=True,
    n_ctx=512,       # Balance capacity and memory
    n_threads=4,      # Suitable for Mac M1
    n_threads_batch=4,
    n_gpu_layers=0,   # CPU-only due to unsupported bf16 kernels
    n_batch=32,
    verbose=True     # Reduce logging
)


docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
output_dir = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

docs = load_file(docs_file)
documents = [
    "\n".join([
        doc["metadata"].get("parent_header") or "",
        doc["metadata"]["header"],
        # doc["metadata"]["content"]
    ]).strip()
    for doc in docs
]
task = 'Given a web search query, retrieve relevant passages that answer the query'
query = "List all ongoing and upcoming isekai anime 2025."

queries = [
    get_detailed_instruct(task, query),
    # get_detailed_instruct(task, 'Explain gravity')
]
documents = documents[:64]

model_name = "mlx-community/Qwen3-0.6B-4bit-DWQ-053125"
model, _ = load(model_name)
query_embeddings = generate_embeddings(
    queries, model_name, model=model, show_progress=True)
document_embeddings = generate_embeddings(
    documents, model_name, model=model, show_progress=True)

copy_to_clipboard([
    len(query_embeddings),
    str(type(query_embeddings[0][0])),
    len(document_embeddings),
    len(document_embeddings[0]),
    str(type(document_embeddings[0][0])),
])


logger.info(
    f"\nComputing similarity scores for {len(document_embeddings)} docs...")
# Start timing
start_time = time.time()

scores = compute_similarity_scores(query_embeddings, document_embeddings)

# End timing
end_time = time.time()
execution_time = end_time - start_time

save_file({"execution_time": f"{execution_time:.2f}s", "count": len(scores), "results": scores},
          f"{output_dir}/similarity_scores.json")

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
