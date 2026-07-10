"""
01_demo_topic_extraction.py

Extract topics from text documents using BERTopic with local embeddings.
Uses reusable factory functions from jet.adapters.bertopic.factory.
"""

from typing import List

from jet.adapters.bertopic.factory import (
    Topic,
    TopicExtractionResult,
    create_bertopic_embedder,
    extract_topics,
    sanity_check_embedder,
)


def run_topic_extraction_demo(
    documents: List[str],
    min_topic_size: int = 3,
    top_n_words: int = 5,
) -> TopicExtractionResult:
    """
    Demonstrate topic extraction using the reusable factory functions.

    Args:
        documents: List of text documents to analyze
        min_topic_size: Minimum documents per topic
        top_n_words: Number of keywords to extract per topic

    Returns:
        Structured topic extraction results
    """
    # Create embedder using factory (reads config from environment)
    embedder = create_bertopic_embedder()

    # Verify the embedding server is working
    sanity_check_embedder(embedder)

    # Extract topics using the high-level function
    result = extract_topics(
        documents=documents,
        embedder=embedder,
        min_topic_size=min_topic_size,
        top_n_words=top_n_words,
        verbose=True,
    )

    # Display results
    print(f"\nExtracted {len(result['topics'])} topics from {len(documents)} documents")
    print(f"Embedding shape: {result['embeddings'].shape}")

    for topic in result["topics"]:
        print(f"\n  Topic {topic['topic_id']}: {topic['name']}")
        print(f"    Size: {topic['size']} documents")
        print(f"    Keywords: {', '.join(topic['keywords'])}")
        print(f"    Representative: {topic['representative_doc'][:100]}...")

    return result


# ---------------------------------------------------------------------------
# Example usage (if run directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example with sample documents
    sample_docs = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Deep learning uses neural networks with many layers to model complex patterns in data.",
        "Natural language processing helps computers understand and generate human language.",
        "Computer vision allows machines to interpret and analyze visual information from the world.",
        "Reinforcement learning trains agents to make decisions through trial and error interactions.",
        "Python is a popular programming language for data science and machine learning projects.",
        "JavaScript is widely used for web development and building interactive user interfaces.",
        "Docker containers help package applications with their dependencies for consistent deployment.",
        "Git is a version control system that tracks changes in source code during development.",
        "REST APIs enable communication between different software systems over HTTP protocols.",
    ]

    result = run_topic_extraction_demo(sample_docs)
