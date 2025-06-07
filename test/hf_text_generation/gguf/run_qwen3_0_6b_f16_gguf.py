import numpy as np
from llama_cpp import Llama
from sklearn.metrics.pairwise import cosine_similarity


def encode_with_padding(model, texts, max_length=512, prompt=None):
    """Encode texts with padding and return fixed-size embeddings."""
    embeddings = []
    for text in texts:
        if prompt:
            text = prompt.format(text)
        # Tokenize and pad/truncate to max_length
        tokens = model.tokenize(text.encode('utf-8'), add_bos=True)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens = tokens + [0] * (max_length - len(tokens))  # Pad with 0
        # Generate embedding and take the last token's embedding for consistency
        embedding = model.embed(text)
        # Ensure embedding is a 1D array of size n_embd (e.g., 1024)
        embedding = np.array(
            embedding)[-1] if len(np.array(embedding).shape) > 1 else np.array(embedding)
        embeddings.append(embedding)
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

# The queries and documents to embed
queries = [
    "What is the capital of China?",
    # "Explain gravity",
]
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

try:
    # Encode queries with a query prompt
    query_prompt = "query: {}"
    query_embeddings = encode_with_padding(
        model, queries, max_length=512, prompt=query_prompt)

    # Encode documents
    document_embeddings = encode_with_padding(model, documents, max_length=512)

    # Verify embedding shapes
    query_shapes = [np.array(e).shape for e in query_embeddings]
    doc_shapes = [np.array(e).shape for e in document_embeddings]
    if len(set(query_shapes + doc_shapes)) > 1:
        raise ValueError(
            f"Inconsistent embedding shapes: Queries {query_shapes}, Documents {doc_shapes}")

    # Convert embeddings to numpy arrays
    query_embeddings = np.array(query_embeddings)
    document_embeddings = np.array(document_embeddings)

    # Compute cosine similarity
    similarity = cosine_similarity(query_embeddings, document_embeddings)
    print("Similarity matrix:")
    print(similarity.tolist())

except Exception as e:
    print(f"Error during embedding or similarity computation: {str(e)}")
finally:
    # Clean up model resources
    model.close()
