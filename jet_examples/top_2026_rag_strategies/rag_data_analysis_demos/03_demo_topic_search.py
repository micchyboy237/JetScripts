"""
03_demo_topic_search.py
Demonstrates reusable find_topics and find_topics_with_data.
"""

from jet.libs.bertopic.monkey_patches.add_check_array import init_patch

init_patch()

from typing import List

from jet.adapters.bertopic.factory_with_repr import (
    create_bertopic_embedder,
    create_topic_model,
    find_topics,
    find_topics_with_data,
    sanity_check_embedder,
)


def run_topic_search_demo(documents: List[str], query: str):
    embedder = create_bertopic_embedder()
    sanity_check_embedder(embedder)
    topic_model = create_topic_model(embedder=embedder, min_topic_size=2, verbose=True)
    _, _ = topic_model.fit_transform(documents)

    print(f"\n{'=' * 60}\nSearching for: '{query}'\n{'=' * 60}")
    topics, sims = find_topics(topic_model, query, top_n=5)
    rich_df = find_topics_with_data(
        topic_model, query, docs=documents, top_n=5, max_reps=2
    )
    print(rich_df)


if __name__ == "__main__":
    from mocks import DOCS

    sample_docs = DOCS

    run_topic_search_demo(sample_docs, "artificial intelligence")
