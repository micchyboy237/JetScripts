"""
03_demo_topic_search.py
Demonstrates the reusable find_topics function: lightweight semantic
search over fitted BERTopic topic centroids (no per-document details).
For enriched results (keywords, size, representative docs), see
04_demo_topic_docs_search.py.
"""

from jet.libs.bertopic.monkey_patches.add_check_array import init_patch

init_patch()
import logging
from typing import List

from jet.adapters.bertopic.factory_with_repr import (
    create_bertopic_embedder,
    create_topic_model,
    find_topics,
    sanity_check_embedder,
)

logger = logging.getLogger(__name__)


def run_topic_search_demo(documents: List[str], query: str, top_n: int = 5):
    """
    Fit a topic model on `documents`, then search for topics matching `query`.

    Args:
        documents: Documents to fit the topic model on
        query: Search term to find matching topics for
        top_n: Number of topics to return
    """
    embedder = create_bertopic_embedder()
    sanity_check_embedder(embedder)
    topic_model = create_topic_model(embedder=embedder, min_topic_size=2, verbose=True)
    embeddings = embedder.embed(documents, verbose=True)
    _, _ = topic_model.fit_transform(documents, embeddings=embeddings)

    logger.info("%s\nSearching for: '%s'\n%s", "=" * 60, query, "=" * 60)
    similar_topics, similarities = find_topics(topic_model, query, top_n=top_n)

    print(f"\n{'=' * 60}\nfind_topics results for: '{query}'\n{'=' * 60}")
    for rank, (tid, sim) in enumerate(zip(similar_topics, similarities), start=1):
        print(f"  #{rank} Topic {tid}: similarity={sim:.4f}")

    return similar_topics, similarities


if __name__ == "__main__":
    from mocks import DOCS

    sample_docs = DOCS
    run_topic_search_demo(sample_docs, "artificial intelligence")
