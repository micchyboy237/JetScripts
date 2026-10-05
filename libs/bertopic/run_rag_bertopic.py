import os
import shutil

from jet.file.utils import save_file
from jet.libs.bertopic.examples.mock import load_sample_jobs
from jet.libs.bertopic.rag_bertopic import TopicRAG

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "generated",
    os.path.splitext(os.path.basename(__file__))[0],
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def run_example_rag_bertopic():
    """Demonstrates RAG retrieval with varying document sets and queries."""

    docs = load_sample_jobs()

    rag = TopicRAG(verbose=True)

    rag.fit_topics(docs)
    save_file(
        rag.model.get_topic_info().to_dict(orient="records"),
        f"{OUTPUT_DIR}/topic_info.json",
    )

    search_results = rag.retrieve_for_query("Top isekai anime 2025")
    save_file(search_results, f"{OUTPUT_DIR}/search_results.json")


if __name__ == "__main__":
    run_example_rag_bertopic()
