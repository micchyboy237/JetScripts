import json
from jet.logger import logger
import mlx_embeddings
import mlx.core as mx
import numpy as np


def load_model():
    """Load the pre-trained model and tokenizer."""
    return mlx_embeddings.load("mlx-community/all-MiniLM-L6-v2-4bit")


def generate_embeddings(model, tokenizer, texts):
    """Generate normalized embeddings for a list of texts."""
    output = mlx_embeddings.generate(model, tokenizer, texts=texts)
    return output.text_embeds


def compute_similarity(query_embedding, corpus_embeddings):
    """Compute cosine similarity between query and corpus embeddings."""
    return mx.matmul(query_embedding, corpus_embeddings.T)


def search(query, corpus_texts, model, tokenizer, top_k=3):
    """Perform semantic search and return top_k most similar texts."""
    # Generate embeddings for query and corpus
    query_embedding = generate_embeddings(model, tokenizer, [query])
    corpus_embeddings = generate_embeddings(model, tokenizer, corpus_texts)

    # Compute similarities
    similarities = compute_similarity(query_embedding, corpus_embeddings)[0]

    # Get top_k indices and scores
    similarities_np = np.array(similarities)
    top_k_indices = np.argsort(similarities_np)[::-1][:top_k]
    top_k_scores = similarities_np[top_k_indices]

    # Return results
    results = [
        {"text": corpus_texts[idx], "score": float(score)}
        for idx, score in zip(top_k_indices, top_k_scores)
    ]
    return results


def main():
    # Load model and tokenizer
    model, tokenizer = load_model()

    # Sample corpus
    corpus = [
        "I enjoy eating fresh fruits",
        "Grapes are my favorite snack",
        "I prefer vegetables over fruits",
        "Wine is made from grapes",
        "Healthy eating includes fruits and vegetables"
    ]

    # Sample query
    query = "I like grapes"
    top_k = 5

    # Perform search
    results = search(query, corpus, model, tokenizer, top_k=top_k)

    # Print results
    logger.newline()
    logger.log(f"Query:", query, colors=["GRAY", "INFO"])
    logger.gray("Top matching texts:")
    for i, result in enumerate(results, 1):
        score = result['score'] * 100
        logger.log(f"{i}.", f"{score:.2f}%", json.dumps(result['text'])[:100], colors=[
                   "DEBUG", "SUCCESS", "WHITE"])


if __name__ == "__main__":
    main()
