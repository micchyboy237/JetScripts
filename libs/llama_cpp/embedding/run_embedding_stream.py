from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
import numpy as np

# Initialize the embedding client
embedder = LlamacppEmbedding(model="nomic-embed-text")

# Example 1: Streaming embeddings for a single text
print("Example 1: Streaming embedding for a single text")
single_text = "This is a sample text for embedding."
for batch_embeddings in embedder.get_embeddings_stream(
    inputs=single_text,
    return_format="numpy",
    batch_size=32,
    show_progress=True,
    use_cache=True,
):
    print(f"Received embedding shape: {batch_embeddings.shape}")
    print(f"First few values: {batch_embeddings.flatten()[:5]}")

# Example 2: Streaming embeddings for multiple texts
print("\nExample 2: Streaming embeddings for multiple texts")
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming the world.",
    "Embeddings are useful for text similarity tasks."
]
for batch_embeddings in embedder.get_embeddings_stream(
    inputs=texts,
    return_format="list",
    batch_size=2,
    show_progress=True,
    use_cache=True,
):
    print(f"Received batch with {len(batch_embeddings)} embeddings")
    for i, embedding in enumerate(batch_embeddings):
        print(f"Embedding {i+1} length: {len(embedding)}")

# Example 3: Processing streamed embeddings for similarity computation
print("\nExample 3: Computing cosine similarity with streamed embeddings")
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

reference_text = "Artificial intelligence is advancing rapidly."
reference_embedding = None
query_texts = [
    "AI is changing technology.",
    "The weather is nice today.",
    "Deep learning models are powerful."
]
for batch_embeddings in embedder.get_embeddings_stream(
    inputs=[reference_text] + query_texts,
    return_format="numpy",
    batch_size=2,
    show_progress=True,
    use_cache=True,
):
    if reference_embedding is None:
        reference_embedding = batch_embeddings[0]
        batch_embeddings = batch_embeddings[1:] if len(batch_embeddings) > 1 else []
    for i, embedding in enumerate(batch_embeddings):
        similarity = cosine_similarity(reference_embedding, embedding)
        print(f"Similarity between reference and query {i+1}: {similarity:.4f}")
