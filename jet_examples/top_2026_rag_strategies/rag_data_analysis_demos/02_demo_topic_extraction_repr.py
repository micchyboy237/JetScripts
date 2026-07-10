"""
02_demo_topic_extraction_repr.py

Extract topics from text documents using BERTopic with local embeddings.
Uses reusable factory functions from jet.adapters.bertopic.factory.
"""

from typing import List

from jet.adapters.bertopic.factory_with_repr import (
    Topic,
    TopicExtractionResult,
    create_bertopic_embedder,
    extract_topics,
    sanity_check_embedder,
)


def run_topic_extraction_demo(
    documents: List[str],
    min_topic_size: int = 3,
    top_n_words: int = 10,
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
    embedder = create_bertopic_embedder()
    sanity_check_embedder(embedder)

    result = extract_topics(
        documents=documents,
        embedder=embedder,
        min_topic_size=min_topic_size,
        top_n_words=top_n_words,
        remove_stop_words=True,
        use_keybert=True,
        verbose=True,
    )

    print(f"\n{'=' * 60}")
    print(f"Extracted {len(result['topics'])} topics from {len(documents)} documents")
    print(f"{'=' * 60}")

    for topic in result["topics"]:
        print(f"\n  Topic {topic['topic_id']}: {topic['name']}")
        print(f"    Size: {topic['size']} documents")
        print(f"    Keywords: {', '.join(topic['keywords'][:5])}")
        if len(topic["keywords"]) > 5:
            print(f"             {', '.join(topic['keywords'][5:10])}")
        print(f"    Representative: {topic['representative_doc'][:120]}...")

    return result


if __name__ == "__main__":
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
