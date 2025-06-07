import numpy as np
from llama_cpp import Llama
from sklearn.preprocessing import normalize


def last_token_pool(embeddings):
    """Ensure embeddings are returned as-is (already fixed-size)."""
    embeddings = np.array(embeddings)
    return embeddings  # Expect 2D array (batch_size, embedding_dim)


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format query with task instruction."""
    return f'Instruct: {task_description}\nQuery: {query}'


def encode_with_padding(model, texts, max_length=512):
    """Encode texts with padding and return fixed-size embeddings."""
    embeddings = []
    for text in texts:
        # Tokenize and pad/truncate to max_length
        tokens = model.tokenize(text.encode('utf-8'), add_bos=True)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens = tokens + [0] * (max_length - len(tokens))  # Pad with 0
        # Generate embedding and select last token's embedding
        embedding = model.embed(text)
        embedding = np.array(embedding)
        # Ensure fixed-size embedding (last token if multi-dimensional)
        if len(embedding.shape) > 1:
            embedding = embedding[-1]
        embeddings.append(embedding)
    # Verify shapes before returning
    shapes = [e.shape for e in embeddings]
    if len(set(shapes)) > 1:
        raise ValueError(f"Inconsistent embedding shapes: {shapes}")
    return embeddings


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

# Task description
task = 'Given a web search query, retrieve relevant passages that answer the query'

# Example with one query and multiple documents
queries = [
    get_detailed_instruct(task, 'What is the capital of China?'),
    # get_detailed_instruct(task, 'Explain gravity')
]
documents = [
    "The capital of China is Beijing.",
    "China is a country in East Asia with a rich history.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
]

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
