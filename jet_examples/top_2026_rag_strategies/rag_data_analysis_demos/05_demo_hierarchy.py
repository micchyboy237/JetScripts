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

    # Compute embeddings explicitly and pass them into fit_transform so
    # BERTopic doesn't re-embed the documents internally (avoids a
    # duplicate embedding call to the llama.cpp server).
    embeddings = embedder.embed(documents, verbose=True)
    _, _ = topic_model.fit_transform(documents, embeddings=embeddings)

    hier_df = explore_hierarchy(topic_model, documents, linkage="ward")
    print("\nFull hierarchy shape:", hier_df.shape)


if __name__ == "__main__":
    from mocks import DOCS

    sample_docs = DOCS
    run_hierarchy_demo(sample_docs)
