from typing import Dict, List, Literal, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from jet.logger import logger
from jet.models.embeddings.sentence_transformer_pooling import AttentionMode, PoolingMode, load_sentence_transformer


def initialize_model(
    pooling_mode: PoolingMode,
    attn_implementation: AttentionMode = "eager"
) -> SentenceTransformer:
    """Initialize SentenceTransformer with specified pooling mode and attention implementation."""

    model_kwargs = {
        "torch_dtype": torch.float16,
        "attn_implementation": attn_implementation
    }
    return load_sentence_transformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        pooling_mode=pooling_mode,
        attn_implementation=attn_implementation,
        model_kwargs=model_kwargs,
    )


def compute_sentence_embeddings(
    sentences: List[str],
    pooling_mode: PoolingMode,
    prompt_name: Optional[str] = None,
    attn_implementation: AttentionMode = "eager"
) -> np.ndarray:
    """Compute sentence embeddings using the specified pooling mode, prompt, and attention implementation."""
    model = initialize_model(pooling_mode, attn_implementation)

    # Apply prompt if specified
    if prompt_name == "description":
        prompt = "Product description: "
        sentences = [prompt + sentence for sentence in sentences]
        logger.debug(
            f"Applied prompt '{prompt}' to {len(sentences)} sentences")

    embeddings = model.encode(
        sentences,
        convert_to_numpy=True,
        prompt_name=None,  # Do not pass prompt_name to model.encode
        prompts=None       # Do not pass prompts dictionary
    )
    return embeddings


def semantic_similarity(
    sentences: List[str],
    pooling_mode: PoolingMode,
    attn_implementation: AttentionMode = "eager"
) -> float:
    """Calculate cosine similarity between two sentences."""
    if len(sentences) != 2:
        raise ValueError(
            "Exactly two sentences are required for similarity comparison.")
    embeddings = compute_sentence_embeddings(
        sentences, pooling_mode, attn_implementation=attn_implementation)
    model = initialize_model(pooling_mode, attn_implementation)
    return float(model.similarity(embeddings, embeddings)[0, 1])


def cluster_sentences(
    sentences: List[str],
    n_clusters: int,
    pooling_mode: PoolingMode,
    attn_implementation: AttentionMode = "eager"
) -> List[int]:
    """Cluster sentences based on their embeddings."""
    embeddings = compute_sentence_embeddings(
        sentences, pooling_mode, attn_implementation=attn_implementation)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(embeddings).tolist()


def semantic_search(
    query: str,
    tickets: List[str],
    pooling_mode: PoolingMode,
    attn_implementation: AttentionMode = "eager"
) -> List[int]:
    """Find top matching support tickets for a query based on cosine similarity."""
    logger.debug(
        f"Starting semantic search with query: {query}, tickets: {len(tickets)}")
    if not tickets:
        logger.warning("No tickets provided for semantic search.")
        return []

    all_sentences = [query] + tickets
    logger.debug(f"All sentences: {len(all_sentences)}")

    embeddings = compute_sentence_embeddings(
        all_sentences, pooling_mode, attn_implementation=attn_implementation)
    logger.debug(f"Embeddings shape: {embeddings.shape}")

    model = initialize_model(pooling_mode, attn_implementation)
    query_embedding = embeddings[0].reshape(1, -1)
    ticket_embeddings = embeddings[1:]
    logger.debug(
        f"Query embedding shape: {query_embedding.shape}, Ticket embeddings shape: {ticket_embeddings.shape}")

    similarities = model.similarity(query_embedding, ticket_embeddings)[0]
    logger.debug(
        f"Similarities shape: {similarities.shape}, values: {similarities}")

    # Convert PyTorch tensor to NumPy array before sorting
    similarities_np = similarities.cpu().numpy()
    ranked_indices = np.argsort(similarities_np)[::-1].tolist()
    logger.debug(f"Ranked indices: {ranked_indices}")

    return ranked_indices


def classify_sentiment(
    sentences: List[str],
    labels: List[str],
    pooling_mode: PoolingMode,
    attn_implementation: AttentionMode = "eager"
) -> List[str]:
    """Classify sentences as positive or negative sentiment."""
    embeddings = compute_sentence_embeddings(
        sentences, pooling_mode, attn_implementation=attn_implementation)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    classifier = LogisticRegression(random_state=42)
    classifier.fit(embeddings, encoded_labels)
    predictions = classifier.predict(embeddings)
    return label_encoder.inverse_transform(predictions).tolist()


def detect_intent(
    sentences: List[str],
    intents: List[str],
    pooling_mode: PoolingMode,
    attn_implementation: AttentionMode = "eager"
) -> List[str]:
    """Detect intents from sentences using a classifier."""
    embeddings = compute_sentence_embeddings(
        sentences, pooling_mode, attn_implementation=attn_implementation)
    label_encoder = LabelEncoder()
    encoded_intents = label_encoder.fit_transform(intents)
    classifier = LogisticRegression(random_state=42)
    classifier.fit(embeddings, encoded_intents)
    predictions = classifier.predict(embeddings)
    return label_encoder.inverse_transform(predictions).tolist()


def document_similarity(
    documents: List[str],
    pooling_mode: PoolingMode,
    attn_implementation: AttentionMode = "eager"
) -> List[float]:
    """Calculate pairwise cosine similarities between documents with a prompt and attention implementation."""
    logger.debug(f"Computing similarities for {len(documents)} documents")
    embeddings = compute_sentence_embeddings(
        documents,
        pooling_mode,
        prompt_name="description",
        attn_implementation=attn_implementation
    )
    logger.debug(f"Embeddings shape: {embeddings.shape}")
    model = initialize_model(pooling_mode, attn_implementation)
    similarities = model.similarity(embeddings, embeddings)
    result = []
    for i in range(len(documents)):
        for j in range(i + 1, len(documents)):
            result.append(float(similarities[i, j]))
    logger.debug(f"Computed {len(result)} pairwise similarities")
    return result


def main():
    # Use Case 1: Semantic Search with Mean Pooling
    query = "My order is delayed"
    support_tickets = [
        "Order not delivered on time.",
        "Product quality is excellent.",
        "Delivery was late by two days.",
        "Great customer service."
    ]
    print("Semantic Search Example (Mean Pooling, Eager):")
    ranked_indices = semantic_search(
        query, support_tickets, "mean_tokens", attn_implementation="eager")
    print(f"Query: {query}")
    print("Ranked tickets:", [support_tickets[i] for i in ranked_indices])

    # Use Case 2: Sentiment Analysis with Max Pooling
    reviews = [
        "This product is amazing and works perfectly!",
        "Terrible experience, very disappointing.",
        "Really happy with my purchase.",
        "The item broke after one use."
    ]
    sentiment_labels = ["positive", "negative", "positive", "negative"]
    print("\nSentiment Analysis Example (Max Pooling, Eager):")
    predicted_sentiments = classify_sentiment(
        reviews, sentiment_labels, "max_tokens", attn_implementation="eager")
    for review, sentiment in zip(reviews, predicted_sentiments):
        print(f"Review: {review} -> Sentiment: {sentiment}")

    # Use Case 3: Intent Detection with CLS Pooling
    user_inputs = [
        "I want to book a flight.",
        "Cancel my order please.",
        "Can you help me book a trip?",
        "I need to cancel my subscription."
    ]
    intents = ["book", "cancel", "book", "cancel"]
    print("\nIntent Detection Example (CLS Pooling, Eager):")
    predicted_intents = detect_intent(
        user_inputs, intents, "cls_token", attn_implementation="eager")
    for input_text, intent in zip(user_inputs, predicted_intents):
        print(f"Input: {input_text} -> Intent: {intent}")

    # Use Case 4: Document Similarity with Mean Sqrt Len Pooling and SDPA
    documents = [
        "This smartphone has a great camera and long battery life.",
        "The phone features an excellent camera and lasts all day.",
        "This laptop is lightweight and has a fast processor."
    ]
    print("\nDocument Similarity Example (Mean Sqrt Len Pooling, SDPA):")
    similarities = document_similarity(
        documents, "mean_sqrt_len_tokens", attn_implementation="sdpa")
    print("Pairwise similarities:")
    pairs = [(0, 1), (0, 2), (1, 2)]
    for (i, j), sim in zip(pairs, similarities):
        print(f"Doc {i} vs Doc {j}: {sim:.4f}")


if __name__ == "__main__":
    main()
