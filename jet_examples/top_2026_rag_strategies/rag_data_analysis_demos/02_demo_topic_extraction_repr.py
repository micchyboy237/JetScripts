"""
02_demo_topic_extraction_repr.py

Enhanced topic extraction with improved representation models.
Demonstrates KeyBERT-inspired topic labeling and stop word removal
for cleaner, more meaningful topic keywords.

Uses reusable factory functions from jet.adapters.bertopic.factory.
"""

from jet.libs.bertopic.monkey_patches.add_check_array import init_patch

init_patch()

from typing import List, Optional

from jet.adapters.bertopic.factory_with_repr import (
    TopicExtractionResult,
    create_bertopic_embedder,
    extract_topics,
    sanity_check_embedder,
)


def run_topic_extraction_demo(
    documents: List[str],
    min_topic_size: int = 3,
    top_n_words: int = 10,
    n_representative_docs: Optional[int] = None,  # None = return all
) -> TopicExtractionResult:
    """
    Demonstrate topic extraction using the reusable factory functions.
    Args:
        documents: List of text documents to analyze
        min_topic_size: Minimum documents per topic
        top_n_words: Number of keywords to extract per topic
        n_representative_docs: Max representative docs per topic.
            None (default) returns all available.
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
        n_representative_docs=n_representative_docs,
    )
    print(f"\n{'=' * 60}")
    print(f"Extracted {len(result['topics'])} topics from {len(documents)} documents")
    print(f"Sorted by size (largest first):")
    print(f"{'=' * 60}")
    max_rep_docs_to_show = 5
    for topic in result["topics"]:
        print(f"\n  Topic {topic['topic_id']}: {topic['name']}")
        print(f"    Size: {topic['size']} documents")
        print(f"    Keywords: {', '.join(topic['keywords'][:5])}")
        if len(topic["keywords"]) > 5:
            print(f"             {', '.join(topic['keywords'][5:10])}")

        # Display up to max_rep_docs_to_show representative docs
        rep_docs = topic["representative_docs"]
        print(f"    Representative docs ({len(rep_docs)} total):")
        for i, doc in enumerate(rep_docs[:max_rep_docs_to_show], 1):
            print(f"      [{i}] {doc[:120]}{'...' if len(doc) > 120 else ''}")
        if len(rep_docs) > max_rep_docs_to_show:
            print(f"      ... and {len(rep_docs) - max_rep_docs_to_show} more")
    return result


if __name__ == "__main__":
    from mocks import DOCS

    sample_docs = DOCS

    result = run_topic_extraction_demo(sample_docs)
