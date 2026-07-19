"""
04_demo_hierarchy.py
Demonstrates explore_hierarchy for topic relationships.
"""

from jet.libs.bertopic.monkey_patches.add_check_array import init_patch

init_patch()

from typing import List

from jet.adapters.bertopic.factory_with_repr import (
    create_bertopic_embedder,
    create_topic_model,
    explore_hierarchy,
    sanity_check_embedder,
)


def run_hierarchy_demo(documents: List[str]):
    embedder = create_bertopic_embedder()
    sanity_check_embedder(embedder)
    topic_model = create_topic_model(embedder=embedder, min_topic_size=2, verbose=True)
    _, _ = topic_model.fit_transform(documents)

    hier_df = explore_hierarchy(topic_model, documents, linkage="ward")
    print("\nFull hierarchy shape:", hier_df.shape)
    # Optional: topic_model.visualize_hierarchy(hier_df)  # if Plotly available


if __name__ == "__main__":
    from mocks import DOCS

    sample_docs = DOCS

    run_hierarchy_demo(sample_docs)
