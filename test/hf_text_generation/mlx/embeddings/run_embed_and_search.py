import json
from jet.file.utils import save_file
from jet.llm.mlx.mlx_types import EmbedModelKey
from jet.llm.mlx.models import AVAILABLE_EMBED_MODELS, resolve_model
from jet.logger import logger
import mlx_embeddings
import mlx.core as mx
import numpy as np


def load_model(embed_model: EmbedModelKey):
    """Load the pre-trained model and tokenizer."""
    model = resolve_model(embed_model)
    return mlx_embeddings.load(model)


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
    import os

    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    os.makedirs(output_dir, exist_ok=True)

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

    # Iterate through all available models
    results_dict = {}
    for model_key in AVAILABLE_EMBED_MODELS.keys():
        results_dict[model_key] = []
        results = results_dict[model_key]

        logger.newline()

        # Load model and tokenizer
        logger.log(f"Testing model:", model_key, colors=["GRAY", "ORANGE"])
        model, tokenizer = load_model(model_key)

        # Perform search
        search_results = search(query, corpus, model, tokenizer, top_k=top_k)

        # Print search results
        logger.log("Query:", query, colors=["GRAY", "INFO"])
        logger.gray(f"Top matching texts (model: {model_key}):")
        for i, result in enumerate(search_results, 1):
            score = result['score'] * 100
            logger.log(f"{i}.", f"{score:.2f}%", json.dumps(result['text'])[:100], colors=[
                       "DEBUG", "SUCCESS", "WHITE"])

            results.append(result)
            save_file(
                results_dict, f"{output_dir}/mlx_embed_and_search_results.json")


if __name__ == "__main__":
    main()
