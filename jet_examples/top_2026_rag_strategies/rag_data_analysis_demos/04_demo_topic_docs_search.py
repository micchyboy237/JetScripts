"""
04_demo_topic_docs_search.py
Demonstrates the reusable find_topics_with_data function: enriched topic
search that returns similarity, topic name, size, keywords, and
representative documents per matching topic.
"""

from jet.libs.bertopic.monkey_patches.add_check_array import init_patch

init_patch()
import logging
from typing import List

import pandas as pd
from jet.adapters.bertopic.factory_with_repr import (
    create_bertopic_embedder,
    create_topic_model,
    find_topics_with_data,
    sanity_check_embedder,
)

logger = logging.getLogger(__name__)


def run_topic_docs_search_demo(
    documents: List[str],
    query: str,
    top_n: int = 5,
    max_reps: int = 2,
) -> pd.DataFrame:
    """
    Fit a topic model on `documents`, then run an enriched topic search
    for `query`, returning similarity, topic size, keywords, and
    representative documents (capped at `max_reps`) per matching topic.

    Args:
        documents: Documents to fit the topic model on
        query: Search term to find matching topics for
        top_n: Number of topics to return
        max_reps: Max representative documents to include per topic

    Returns:
        DataFrame with columns: Topic, Name, Similarity, Size, Top_Words,
        Representative_Docs_Count, Representative_Docs
    """
    embedder = create_bertopic_embedder()
    sanity_check_embedder(embedder)
    topic_model = create_topic_model(embedder=embedder, min_topic_size=2, verbose=True)
    embeddings = embedder.embed(documents, verbose=True)
    _, _ = topic_model.fit_transform(documents, embeddings=embeddings)

    logger.info("%s\nSearching (with data) for: '%s'\n%s", "=" * 60, query, "=" * 60)
    rich_df = find_topics_with_data(
        topic_model,
        query,
        docs=documents,
        top_n=top_n,
        max_reps=max_reps,
    )

    print(f"\n{'=' * 60}\nfind_topics_with_data results for: '{query}'\n{'=' * 60}")
    print(rich_df.to_string(index=False))

    return rich_df


if __name__ == "__main__":
    from mocks import DOCS

    sample_docs = DOCS
    run_topic_docs_search_demo(sample_docs, "artificial intelligence")
