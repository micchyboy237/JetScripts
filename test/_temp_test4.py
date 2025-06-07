import logging
from typing import List, Union
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from jet.file.utils import load_file, save_file
import os
import numpy as np
from llama_cpp import Llama
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def last_token_pool(embeddings):
    """Ensure embeddings are returned as-is (already fixed-size)."""
    embeddings = np.array(embeddings)
    return embeddings  # Expect 2D array (batch_size, embedding_dim)


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format query with task instruction."""
    return f'Instruct: {task_description}\nQuery: {query}'


def encode_with_padding(model: Llama, texts: List[str], max_length: int = 512, batch_size: int = 8) -> np.ndarray:
    """
    Encode texts with padding and return fixed-size embeddings in batches.

    Skips batching when there's only one input.

    Args:
        model: Llama model instance.
        texts: List of texts to encode.
        max_length: Maximum token length for padding/truncation.
        batch_size: Number of texts to process per batch.

    Returns:
        np.ndarray: Array of embeddings with shape (len(texts), embedding_dim).
    """
    if len(texts) == 1:
        logger.info("Single input detected. Skipping batch processing.")
        try:
            embedding = model.embed(texts[0])
            embedding = np.array(embedding)
            if len(embedding.shape) > 1:
                embedding = embedding[-1]
        except Exception as e:
            logger.error(f"Error embedding single input: {str(e)}")
            embedding = np.zeros(512, dtype=np.float32)  # fallback

        return np.expand_dims(embedding, axis=0)

    def tokenize_text(text: str) -> List[int]:
        tokens = model.tokenize(text.encode('utf-8'), add_bos=True)
        if len(tokens) > max_length:
            return tokens[:max_length]
        return tokens + [0] * (max_length - len(tokens))

    total_batches = (len(texts) + batch_size - 1) // batch_size
    embeddings = []

    for batch_idx in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
        current_batch = texts[batch_idx:batch_idx + batch_size]
        logger.info(
            f"Processing batch {batch_idx // batch_size + 1}/{total_batches} - {len(current_batch)} items")

        with ThreadPoolExecutor(max_workers=model.n_threads) as executor:
            batch_tokens = list(executor.map(tokenize_text, current_batch))

        batch_max_len = min(max_length, max(len(tokens)
                            for tokens in batch_tokens))
        logger.info(
            f"Batch {batch_idx // batch_size + 1}: max token length after padding = {batch_max_len}")

        batch_tokens = [
            tokens[:batch_max_len] + [0] * (batch_max_len - len(tokens))
            if len(tokens) < batch_max_len else tokens[:batch_max_len]
            for tokens in batch_tokens
        ]

        batch_embeddings = []
        for idx, text in enumerate(current_batch):
            try:
                embedding = model.embed(text)
                embedding = np.array(embedding)
                if len(embedding.shape) > 1:
                    embedding = embedding[-1]
                batch_embeddings.append(embedding)
            except Exception as e:
                logger.error(
                    f"Error embedding text at batch {batch_idx // batch_size + 1}, item {idx + 1}: {str(e)}")
                batch_embeddings.append(np.zeros_like(embedding))

        shapes = [e.shape for e in batch_embeddings]
        if len(set(shapes)) > 1:
            logger.warning(
                f"Batch {batch_idx // batch_size + 1}: Inconsistent embedding shapes: {shapes}")
            max_dim = max(s[0] for s in shapes)
            batch_embeddings = [
                np.pad(
                    e, (0, max_dim - e.shape[0]), mode='constant') if e.shape[0] < max_dim else e[:max_dim]
                for e in batch_embeddings
            ]

        logger.info(
            f"Batch {batch_idx // batch_size + 1}: Final embedding shape = {batch_embeddings[0].shape}")
        embeddings.extend(batch_embeddings)

    logger.info(f"Total embeddings generated: {len(embeddings)}")
    return np.array(embeddings)


# Load the GGUF model
model_path = "/Users/jethroestrada/Downloads/Qwen3-Embedding-0.6B-f16.gguf"
model = Llama(
    model_path=model_path,
    embedding=True,
    n_ctx=2048,       # Balance capacity and memory
    n_threads=4,      # Suitable for Mac M1
    n_gpu_layers=0,   # CPU-only due to unsupported bf16 kernels
    verbose=False     # Reduce logging
)

# # Task description
# task = 'Given a web search query, retrieve relevant passages that answer the query'

# # Example with one query and multiple documents
# queries = [
#     get_detailed_instruct(task, 'What is the capital of China?'),
#     # get_detailed_instruct(task, 'Explain gravity')
# ]
# documents = [
#     "The capital of China is Beijing.",
#     "China is a country in East Asia with a rich history.",
#     "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
# ]


docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
output_dir = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

docs = load_file(docs_file)
documents = [
    "\n".join([
        doc["metadata"].get("parent_header") or "",
        doc["metadata"]["header"],
        doc["metadata"]["content"]
    ]).strip()
    for doc in docs
]
task = 'Given a web search query, retrieve relevant passages that answer the query'
query = "List all ongoing and upcoming isekai anime 2025."

queries = [
    get_detailed_instruct(task, query),
    # get_detailed_instruct(task, 'Explain gravity')
]
documents = documents[:20]

try:
    # Encode queries and documents
    query_embeddings = encode_with_padding(model, queries, max_length=512)
    document_embeddings = encode_with_padding(model, documents, max_length=512)

    # Apply last-token pooling (no-op as embeddings are fixed-size)
    query_embeddings = last_token_pool(query_embeddings)
    document_embeddings = last_token_pool(document_embeddings)

    # Normalize embeddings
    query_embeddings = normalize(query_embeddings, norm='l2', axis=1)
    document_embeddings = normalize(document_embeddings, norm='l2', axis=1)

    # Compute similarity scores (queries vs documents)
    scores = query_embeddings @ document_embeddings.T
    print("Similarity matrix:")
    print(scores.tolist())

except Exception as e:
    print(f"Error during embedding or similarity computation: {str(e)}")
finally:
    # Clean up model resources
    model.close()
