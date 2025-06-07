import os
import numpy as np
from llama_cpp import Llama


def initialize_llama_model(model_path):
    """Initialize the Llama model with optimized settings for Mac M1."""
    return Llama(
        model_path=model_path,
        n_gpu_layers=99,  # Offload all possible layers to Metal GPU
        flash_attn=True,  # Enable Flash Attention for faster computation
        n_ctx=2048,       # Increase context size for flexibility
        n_batch=512,      # Batch size for efficient processing
        n_threads=4,      # Use 4 threads for M1’s 4 performance cores
        use_mmap=True,    # Memory mapping for efficient model loading
        use_mlock=False,  # Disable memory locking to avoid overhead
        offload_kqv=True,  # Offload key-value cache to GPU
        embedding=True,   # Enable embedding mode for embed method
        verbose=True      # Enable logging for debugging
    )


def generate_embeddings(llama, texts, normalize=True):
    """
    Generate embeddings for a list of texts using Llama's embed method.
    Applies mean pooling if per-token embeddings are returned.

    Args:
        llama: Initialized Llama model instance
        texts: List of strings or single string to embed
        normalize: Whether to normalize the embeddings

    Returns:
        NumPy array of embeddings, total token count
    """
    if isinstance(texts, str):
        texts = [texts]

    embeddings = []
    total_tokens = 0
    expected_dim = llama.n_embd()  # Expected embedding dimension

    for text in texts:
        if not text.strip():  # Handle empty strings
            embeddings.append(np.zeros(expected_dim, dtype=np.float32))
            continue
        # Use Llama's embed method
        embedding, token_count = llama.embed(
            input=text,
            normalize=normalize,
            truncate=True,
            return_count=True
        )
        # Convert to NumPy array and apply mean pooling if needed
        embedding = np.array(embedding, dtype=np.float32)
        if embedding.ndim == 2:  # Per-token embeddings [seq_len, emb_dim]
            embedding = np.mean(embedding, axis=0)  # Mean pool to [emb_dim]

        # Verify embedding shape
        if embedding.shape != (expected_dim,):
            raise ValueError(
                f"Embedding for text '{text}' has shape {embedding.shape}, "
                f"expected ({expected_dim},)"
            )
        embeddings.append(embedding)
        total_tokens += token_count

    return np.array(embeddings), total_tokens


def main():
    # Set model path
    model_path = "/Users/jethroestrada/Downloads/Qwen3-Embedding-0.6B-f16.gguf"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Initialize Llama model with optimized settings
    llama = initialize_llama_model(model_path)

    # Example texts for embedding
    texts = [
        "Hello, world!",
        "This is a test sentence.",
        "Qwen model embedding generation."
    ]

    # Generate embeddings
    try:
        embeddings, total_tokens = generate_embeddings(llama, texts)

        # Print results
        print(f"Total tokens used: {total_tokens}")
        for text, emb in zip(texts, embeddings):
            print(f"Text: {text}")
            print(f"Embedding shape: {emb.shape}")
            print(f"Embedding (first 5 values): {emb[:5]}\n")

    finally:
        # Clean up
        llama.close()


if __name__ == "__main__":
    main()
